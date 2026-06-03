"""Behavioural contract tests for the document-analysis API."""
from __future__ import annotations

import io
import os

import requests
from conftest import SAMPLE_DIR, upload_txt


# --- health + documents ------------------------------------------------------ #
def test_health(base):
    r = requests.get(f"{base}/health", timeout=10)
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_upload_txt(base):
    doc_id, wc = upload_txt(base)
    assert doc_id and wc > 0


def test_upload_pdf(base):
    with open(SAMPLE_DIR / "sample.pdf", "rb") as f:
        r = requests.post(f"{base}/documents", files={"file": ("sample.pdf", f, "application/pdf")}, timeout=15)
    assert r.status_code == 201
    assert r.json()["word_count"] > 0


def test_upload_stored_on_disk(base, data_dir):
    upload_txt(base, name="stored_test.txt")
    assert any("stored_test" in n for n in os.listdir(data_dir / "uploads"))


def test_db_created(base, data_dir):
    upload_txt(base)
    assert (data_dir / "documents.db").exists()


def test_path_traversal_rejected(base):
    r = requests.post(
        f"{base}/documents",
        files={"file": ("../../etc/passwd.txt", b"hello", "text/plain")},
        timeout=10,
    )
    assert r.status_code == 400


def test_filename_sanitized(base, data_dir):
    r = requests.post(
        f"{base}/documents",
        files={"file": ("deep/path/file.txt", b"some words here", "text/plain")},
        timeout=10,
    )
    # No "../" so it's accepted, but path components are stripped.
    assert r.status_code == 201
    assert any("file.txt" in n and "deep" not in n for n in os.listdir(data_dir / "uploads"))


def test_unsupported_type_rejected(base):
    r = requests.post(f"{base}/documents", files={"file": ("x.docx", b"data", "application/octet-stream")}, timeout=10)
    assert r.status_code == 400


def test_corrupt_pdf_rejected(base):
    r = requests.post(
        f"{base}/documents",
        files={"file": ("broken.pdf", io.BytesIO(b"%PDF-1.4 not really a pdf"), "application/pdf")},
        timeout=10,
    )
    assert r.status_code == 400


def test_get_document(base):
    doc_id, wc = upload_txt(base)
    r = requests.get(f"{base}/documents/{doc_id}", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == doc_id and body["word_count"] == wc
    assert "filename" in body and "created_at" in body


def test_get_document_404(base):
    assert requests.get(f"{base}/documents/nope-123", timeout=10).status_code == 404


def test_delete_document(base):
    doc_id, _ = upload_txt(base)
    assert requests.delete(f"{base}/documents/{doc_id}", timeout=10).status_code == 204
    assert requests.get(f"{base}/documents/{doc_id}", timeout=10).status_code == 404


def test_delete_document_404(base):
    assert requests.delete(f"{base}/documents/nope-123", timeout=10).status_code == 404


def test_delete_cascades_results(base, data_dir):
    doc_id, _ = upload_txt(base)
    analysis = requests.post(
        f"{base}/analyze",
        json={"document_id": doc_id, "model_name": "context-analyzer-v1", "analysis_type": "summary"},
        timeout=15,
    ).json()
    result_file = data_dir / "results" / f"{analysis['analysis_id']}.json"
    assert result_file.exists()
    requests.delete(f"{base}/documents/{doc_id}", timeout=10)
    assert not result_file.exists()


# --- listing / pagination / sorting ----------------------------------------- #
def test_list_documents_shape(base):
    upload_txt(base)
    r = requests.get(f"{base}/documents", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert "documents" in body and "total" in body
    assert all("id" in d for d in body["documents"])
    assert len(body["documents"]) <= 10  # default limit


def test_list_pagination_and_sort(base):
    for _ in range(3):
        upload_txt(base)
    r = requests.get(f"{base}/documents?limit=2&offset=0&sort_by=word_count&order=asc", timeout=10)
    assert r.status_code == 200
    assert len(r.json()["documents"]) <= 2


def test_list_invalid_params(base):
    assert requests.get(f"{base}/documents?sort_by=bogus", timeout=10).status_code == 400
    assert requests.get(f"{base}/documents?order=sideways", timeout=10).status_code == 400
    assert requests.get(f"{base}/documents?limit=-1", timeout=10).status_code == 400
    assert requests.get(f"{base}/documents?offset=-5", timeout=10).status_code == 400


# --- models ------------------------------------------------------------------ #
def test_models(base):
    r = requests.get(f"{base}/models", timeout=10)
    assert r.status_code == 200
    models = r.json()["models"]
    assert len(models) == 3
    for m in models:
        assert {"name", "context_window", "version"} <= set(m)


# --- analyze ----------------------------------------------------------------- #
def test_analyze_returns_run_and_chunks(base):
    doc_id, _ = upload_txt(base)
    r = requests.post(
        f"{base}/analyze",
        json={"document_id": doc_id, "model_name": "summary-extractor", "analysis_type": "keywords"},
        timeout=15,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mlflow_run_id"]
    assert body["num_chunks"] >= 1
    assert len(body["results"]) == body["num_chunks"]
    for res in body["results"]:
        assert "chunk_index" in res and "chunk_text" in res and "keywords" in res


def test_analyze_validation(base):
    doc_id, _ = upload_txt(base)
    # missing document_id
    assert requests.post(f"{base}/analyze", json={"model_name": "summary-extractor"}, timeout=10).status_code == 400
    # unknown document
    assert requests.post(
        f"{base}/analyze", json={"document_id": "ghost", "model_name": "summary-extractor"}, timeout=10
    ).status_code == 400
    # unknown model -> 404
    assert requests.post(
        f"{base}/analyze", json={"document_id": doc_id, "model_name": "nope"}, timeout=10
    ).status_code == 404
    # invalid analysis_type
    assert requests.post(
        f"{base}/analyze",
        json={"document_id": doc_id, "model_name": "summary-extractor", "analysis_type": "vibes"},
        timeout=10,
    ).status_code == 400
    # context_window too large
    assert requests.post(
        f"{base}/analyze",
        json={"document_id": doc_id, "model_name": "context-analyzer-v1", "context_window": 999999},
        timeout=10,
    ).status_code == 400


def test_analyze_caching(base):
    doc_id, _ = upload_txt(base)
    payload = {"document_id": doc_id, "model_name": "context-analyzer-v1", "analysis_type": "summary"}
    first = requests.post(f"{base}/analyze", json=payload, timeout=15).json()
    second = requests.post(f"{base}/analyze", json=payload, timeout=15).json()
    assert second.get("cached") is True
    assert second["analysis_id"] == first["analysis_id"]
    # skip_cache bypasses and produces a fresh analysis_id
    third = requests.post(f"{base}/analyze", json={**payload, "skip_cache": True}, timeout=15).json()
    assert third.get("cached") is not True
    assert third["analysis_id"] != first["analysis_id"]


# --- results ----------------------------------------------------------------- #
def test_results_full_and_chunk(base):
    doc_id, _ = upload_txt(base)
    analysis = requests.post(
        f"{base}/analyze",
        json={"document_id": doc_id, "model_name": "summary-extractor", "analysis_type": "summary"},
        timeout=15,
    ).json()
    aid = analysis["analysis_id"]
    full = requests.get(f"{base}/results/{aid}", timeout=10)
    assert full.status_code == 200 and "results" in full.json()
    one = requests.get(f"{base}/results/{aid}?chunk_index=0", timeout=10)
    assert one.status_code == 200 and len(one.json()["results"]) == 1
    assert requests.get(f"{base}/results/{aid}?chunk_index=9999", timeout=10).status_code == 400
    assert requests.get(f"{base}/results/missing", timeout=10).status_code == 404


# --- batch ------------------------------------------------------------------- #
def test_batch_analyze_mixed(base):
    doc_id, _ = upload_txt(base)
    jobs = [
        {"document_id": doc_id, "model_name": "summary-extractor", "analysis_type": "summary"},
        {"document_id": "ghost", "model_name": "summary-extractor"},  # fails
        {"document_id": doc_id, "model_name": "summary-extractor", "analysis_type": "keywords"},
    ]
    r = requests.post(f"{base}/batch/analyze", json={"jobs": jobs}, timeout=30)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["total_jobs"] == 3
    assert body["successful_jobs"] == 2 and body["failed_jobs"] == 1
    assert len(body["results"]) == 3
    assert "error" in body["results"][1]  # failed slot preserved in order


def test_batch_empty_rejected(base):
    assert requests.post(f"{base}/batch/analyze", json={"jobs": []}, timeout=10).status_code == 400
    assert requests.post(f"{base}/batch/analyze", json={}, timeout=10).status_code == 400


# --- compare ----------------------------------------------------------------- #
def test_compare(base):
    doc_id, _ = upload_txt(base)
    a1 = requests.post(
        f"{base}/analyze",
        json={"document_id": doc_id, "model_name": "summary-extractor", "analysis_type": "keywords"},
        timeout=15,
    ).json()["analysis_id"]
    a2 = requests.post(
        f"{base}/analyze",
        json={"document_id": doc_id, "model_name": "summary-extractor", "analysis_type": "keywords", "skip_cache": True},
        timeout=15,
    ).json()["analysis_id"]
    r = requests.post(f"{base}/compare", json={"analysis_id_1": a1, "analysis_id_2": a2}, timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert "common_keywords" in body
    assert 0.0 <= body["sentiment_agreement"] <= 1.0
    assert body["chunk_count_delta"] == 0


def test_compare_validation(base):
    assert requests.post(f"{base}/compare", json={"analysis_id_1": "x"}, timeout=10).status_code == 400
    assert requests.post(
        f"{base}/compare", json={"analysis_id_1": "x", "analysis_id_2": "y"}, timeout=10
    ).status_code == 404
