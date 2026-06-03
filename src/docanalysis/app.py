"""Flask application factory for the document-analysis API.

Wires the domain logic in :mod:`docanalysis.core` to HTTP routes and adds the
operational layer: API-key auth, Prometheus metrics, JSON request logging, and a
background cleanup thread. Route behaviour matches the reference contract.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime

import mlflow
from flask import Flask, jsonify, request

from .config import Config, load_config
from .core import (
    MODELS,
    LRUCache,
    analyze_chunk,
    cache_key,
    chunk_text,
    deliver_webhook,
    extract_text_from_pdf,
    extract_text_from_txt,
    get_db,
    init_db,
    log_stderr,
)
from .logging_config import configure_logging

_log = logging.getLogger("docanalysis")


def _run_analysis(cfg, cache, doc_id, content, model_name, analysis_type, context_window):
    """Chunk, log to MLFlow, persist results + a DB row, and cache. Returns the
    output dict shared by /analyze and /batch/analyze."""
    chunks = chunk_text(content, context_window)

    run = mlflow.start_run()
    mlflow.log_param("document_id", doc_id)
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("analysis_type", analysis_type)
    mlflow.log_param("num_chunks", len(chunks))

    results = []
    word_counts = []
    for idx, chunk in enumerate(chunks):
        word_counts.append(len(chunk.split()))
        results.append(analyze_chunk(idx, chunk, analysis_type))
    if word_counts:
        mlflow.log_metric("avg_chunk_words", sum(word_counts) / len(word_counts))
        mlflow.log_metric("max_chunk_words", max(word_counts))
    mlflow.end_run()

    analysis_id = str(uuid.uuid4())
    output = {
        "analysis_id": analysis_id,
        "document_id": doc_id,
        "model_name": model_name,
        "context_window": context_window,
        "analysis_type": analysis_type,
        "num_chunks": len(chunks),
        "mlflow_run_id": run.info.run_id,
        "results": results,
    }
    with open(cfg.results_dir / f"{analysis_id}.json", "w") as f:
        json.dump(output, f)

    conn = get_db(cfg.db_path)
    conn.execute(
        "INSERT INTO analyses (analysis_id, document_id, model_name, analysis_type, "
        "context_window, num_chunks, mlflow_run_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (analysis_id, doc_id, model_name, analysis_type, context_window, len(chunks), run.info.run_id),
    )
    conn.commit()
    conn.close()

    cache.put(cache_key(doc_id, model_name, analysis_type, context_window), output)
    return output


def create_app(cfg: Config | None = None) -> Flask:
    cfg = cfg or load_config()
    configure_logging(cfg.log_level)
    cfg.ensure_dirs()
    init_db(cfg.db_path)

    mlflow.set_tracking_uri(cfg.mlruns_uri)
    mlflow.set_experiment(cfg.experiment)

    cache = LRUCache(capacity=cfg.cache_capacity, ttl=cfg.cache_ttl)

    app = Flask(__name__)
    app.config["docanalysis"] = cfg

    # --- Prometheus metrics (registers /metrics) --------------------------- #
    try:
        from prometheus_flask_exporter import PrometheusMetrics

        PrometheusMetrics(app, group_by="endpoint")
    except Exception as exc:  # pragma: no cover - metrics are best-effort
        log_stderr(f"prometheus metrics unavailable: {exc}")

    _OPEN_PATHS = {"/health", "/metrics"}

    @app.before_request
    def _auth():
        if request.method == "OPTIONS" or request.path in _OPEN_PATHS:
            return None
        if not cfg.api_keys:
            return None
        presented = request.headers.get("X-API-Key", "")
        import hmac

        if presented and any(hmac.compare_digest(presented, k) for k in cfg.api_keys):
            return None
        return jsonify({"error": "Missing or invalid API key"}), 401

    @app.after_request
    def _log_request(response):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "path": request.path,
            "status_code": response.status_code,
        }
        try:
            with open(cfg.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:  # pragma: no cover
            log_stderr(f"Failed to write to log: {exc}")
        return response

    # --- routes ------------------------------------------------------------ #
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/documents", methods=["POST"])
    def upload_document():
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        if ".." in file.filename:
            return jsonify({"error": "Invalid filename"}), 400

        safe_filename = os.path.basename(file.filename)
        if not (safe_filename.endswith(".pdf") or safe_filename.endswith(".txt")):
            return jsonify({"error": "Unsupported file type"}), 400

        doc_id = str(uuid.uuid4())
        filepath = str(cfg.upload_dir / f"{doc_id}_{safe_filename}")
        conn = get_db(cfg.db_path)
        try:
            file.save(filepath)
            try:
                if safe_filename.endswith(".pdf"):
                    content = extract_text_from_pdf(filepath)
                else:
                    content = extract_text_from_txt(filepath)
            except Exception as exc:
                os.remove(filepath)
                return jsonify({"error": str(exc)}), 400

            word_count = len(content.split())
            created_at = datetime.utcnow().isoformat()
            conn.execute(
                "INSERT INTO documents (id, filename, filepath, word_count, content, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (doc_id, safe_filename, filepath, word_count, content, created_at),
            )
            conn.commit()
            return jsonify({"document_id": doc_id, "word_count": word_count}), 201
        except Exception as exc:
            conn.rollback()
            if os.path.exists(filepath):
                os.remove(filepath)
            log_stderr(f"Upload error: {exc}")
            return jsonify({"error": str(exc)}), 500
        finally:
            conn.close()

    @app.route("/documents/<doc_id>", methods=["GET"])
    def get_document(doc_id):
        conn = get_db(cfg.db_path)
        row = conn.execute(
            "SELECT id, word_count, filename, created_at FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        conn.close()
        if row is None:
            return jsonify({"error": "Document not found"}), 404
        return jsonify({
            "id": row["id"],
            "word_count": row["word_count"],
            "filename": row["filename"],
            "created_at": row["created_at"],
        })

    @app.route("/documents/<doc_id>", methods=["DELETE"])
    def delete_document(doc_id):
        conn = get_db(cfg.db_path)
        try:
            row = conn.execute("SELECT filepath FROM documents WHERE id = ?", (doc_id,)).fetchone()
            if row is None:
                conn.close()
                return jsonify({"error": "Document not found"}), 404
            filepath = row["filepath"]
            if os.path.exists(filepath):
                os.remove(filepath)
            analyses = conn.execute(
                "SELECT analysis_id FROM analyses WHERE document_id = ?", (doc_id,)
            ).fetchall()
            for a in analyses:
                result_path = cfg.results_dir / f"{a['analysis_id']}.json"
                if result_path.exists():
                    result_path.unlink()
            conn.execute("DELETE FROM analyses WHERE document_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()
            return "", 204
        except Exception as exc:
            conn.rollback()
            log_stderr(f"Delete error: {exc}")
            return jsonify({"error": str(exc)}), 500
        finally:
            conn.close()

    @app.route("/documents", methods=["GET"])
    def list_documents():
        try:
            limit = int(request.args.get("limit", 10))
            offset = int(request.args.get("offset", 0))
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid pagination parameters"}), 400
        if limit < 0 or offset < 0:
            return jsonify({"error": "limit and offset must be non-negative"}), 400

        sort_by = request.args.get("sort_by", "created_at")
        order = request.args.get("order", "desc")
        if sort_by not in ("word_count", "created_at"):
            return jsonify({"error": "Invalid sort_by"}), 400
        if order not in ("asc", "desc"):
            return jsonify({"error": "Invalid order"}), 400
        direction = "DESC" if order == "desc" else "ASC"

        conn = get_db(cfg.db_path)
        total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        rows = conn.execute(
            f"SELECT id, word_count, filename, created_at FROM documents "
            f"ORDER BY {sort_by} {direction} LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        conn.close()
        return jsonify({
            "documents": [
                {
                    "id": r["id"],
                    "word_count": r["word_count"],
                    "filename": r["filename"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ],
            "total": total,
        })

    @app.route("/models", methods=["GET"])
    def list_models():
        return jsonify({"models": MODELS})

    @app.route("/analyze", methods=["POST"])
    def analyze():
        data = request.get_json(silent=True) or {}
        doc_id = data.get("document_id")
        model_name = data.get("model_name")
        context_window = data.get("context_window")
        analysis_type = data.get("analysis_type", "summary")
        skip_cache = data.get("skip_cache", False)

        if not doc_id:
            return jsonify({"error": "document_id is required"}), 400

        conn = get_db(cfg.db_path)
        doc = conn.execute("SELECT id, content FROM documents WHERE id = ?", (doc_id,)).fetchone()
        conn.close()
        if doc is None:
            return jsonify({"error": "Document not found"}), 400
        if not model_name:
            return jsonify({"error": "model_name is required"}), 400

        model = next((m for m in MODELS if m["name"] == model_name), None)
        if not model:
            return jsonify({"error": "Model not found"}), 404
        if context_window is None:
            context_window = model["context_window"]
        if context_window > model["context_window"]:
            return jsonify({"error": "context_window exceeds model maximum"}), 400
        if analysis_type not in ("summary", "keywords", "sentiment"):
            return jsonify({"error": "Unsupported analysis_type"}), 400

        if not skip_cache:
            cached = cache.get(cache_key(doc_id, model_name, analysis_type, context_window))
            if cached:
                cached["cached"] = True
                return jsonify(cached), 200

        output = _run_analysis(cfg, cache, doc_id, doc["content"], model_name, analysis_type, context_window)
        return jsonify(output), 200

    @app.route("/batch/analyze", methods=["POST"])
    def batch_analyze():
        data = request.get_json(silent=True) or {}
        jobs = data.get("jobs")
        webhook_url = data.get("webhook_url")
        if not jobs or not isinstance(jobs, list):
            return jsonify({"error": "jobs array is required"}), 400

        batch_id = str(uuid.uuid4())
        results = []
        successful = 0
        failed = 0
        for job in jobs:
            try:
                doc_id = job.get("document_id")
                model_name = job.get("model_name")
                context_window = job.get("context_window")
                analysis_type = job.get("analysis_type", "summary")

                if not doc_id:
                    results.append({"error": "Invalid or missing document_id"})
                    failed += 1
                    continue
                conn = get_db(cfg.db_path)
                doc = conn.execute(
                    "SELECT id, content FROM documents WHERE id = ?", (doc_id,)
                ).fetchone()
                conn.close()
                if doc is None:
                    results.append({"error": "Invalid or missing document_id"})
                    failed += 1
                    continue
                model = next((m for m in MODELS if m["name"] == model_name), None)
                if not model:
                    results.append({"error": "Model not found"})
                    failed += 1
                    continue
                if context_window is None:
                    context_window = model["context_window"]
                if context_window > model["context_window"]:
                    results.append({"error": "context_window exceeds model maximum"})
                    failed += 1
                    continue
                if analysis_type not in ("summary", "keywords", "sentiment"):
                    results.append({"error": "Unsupported analysis_type"})
                    failed += 1
                    continue

                output = _run_analysis(
                    cfg, cache, doc_id, doc["content"], model_name, analysis_type, context_window
                )
                results.append(
                    {"analysis_id": output["analysis_id"], "mlflow_run_id": output["mlflow_run_id"]}
                )
                successful += 1
            except Exception as exc:
                results.append({"error": str(exc)})
                failed += 1

        response_data = {
            "batch_id": batch_id,
            "status": "completed",
            "total_jobs": len(jobs),
            "successful_jobs": successful,
            "failed_jobs": failed,
            "results": results,
        }
        if webhook_url:
            threading.Thread(target=deliver_webhook, args=(webhook_url, response_data), daemon=True).start()
        return jsonify(response_data), 200

    @app.route("/compare", methods=["POST"])
    def compare_analyses():
        data = request.get_json(silent=True) or {}
        id1 = data.get("analysis_id_1")
        id2 = data.get("analysis_id_2")
        if not id1 or not id2:
            return jsonify({"error": "Both analysis_id_1 and analysis_id_2 are required"}), 400

        path1 = cfg.results_dir / f"{id1}.json"
        path2 = cfg.results_dir / f"{id2}.json"
        if not path1.exists() or not path2.exists():
            return jsonify({"error": "One or both analyses not found"}), 404

        data1 = json.loads(path1.read_text())
        data2 = json.loads(path2.read_text())

        kw1, kw2 = set(), set()
        for r in data1.get("results", []):
            if "keywords" in r:
                kw1.update(r["keywords"])
        for r in data2.get("results", []):
            if "keywords" in r:
                kw2.update(r["keywords"])
        common_keywords = list(kw1 & kw2)

        s1 = [r.get("sentiment", "") for r in data1.get("results", [])]
        s2 = [r.get("sentiment", "") for r in data2.get("results", [])]
        min_len = min(len(s1), len(s2))
        matches = sum(1 for i in range(min_len) if s1[i] == s2[i])
        agreement = matches / min_len if min_len > 0 else 0.0
        chunk_delta = abs(data1.get("num_chunks", 0) - data2.get("num_chunks", 0))

        return jsonify({
            "common_keywords": common_keywords,
            "sentiment_agreement": agreement,
            "chunk_count_delta": chunk_delta,
        }), 200

    @app.route("/results/<analysis_id>", methods=["GET"])
    def get_results(analysis_id):
        filepath = cfg.results_dir / f"{analysis_id}.json"
        if not filepath.exists():
            return jsonify({"error": "Analysis not found"}), 404
        data = json.loads(filepath.read_text())
        chunk_index = request.args.get("chunk_index", type=int)
        if chunk_index is not None:
            if chunk_index < 0 or chunk_index >= len(data["results"]):
                return jsonify({"error": "chunk_index out of bounds"}), 400
            data["results"] = [data["results"][chunk_index]]
        return jsonify(data)

    _start_cleanup_thread(cfg, cache)
    return app


def _start_cleanup_thread(cfg: Config, cache: LRUCache) -> None:
    def worker():
        while True:
            try:
                time.sleep(3600)
                purged = cache.purge_expired()
                if purged:
                    log_stderr(f"Cleaned up {purged} expired cache entries")
                _cleanup_old_logs(cfg)
            except Exception as exc:  # pragma: no cover
                log_stderr(f"Cleanup worker error: {exc}")

    threading.Thread(target=worker, daemon=True).start()


def _cleanup_old_logs(cfg: Config) -> None:
    cutoff = time.time() - 7 * 24 * 60 * 60
    removed = 0
    if cfg.logs_dir.exists():
        for f in cfg.logs_dir.iterdir():
            try:
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink()
                    removed += 1
            except Exception:
                pass
    if removed:
        log_stderr(f"Cleaned up {removed} old log files")
