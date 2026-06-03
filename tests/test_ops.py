"""Operational-layer tests: request logging, MLFlow tracking, metrics, auth.

The auth test boots its own server with API keys configured.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time

import requests
from conftest import PROJECT_ROOT, SAMPLE_DIR, upload_txt


def test_request_logging(base, data_dir):
    requests.get(f"{base}/health", timeout=10)
    requests.get(f"{base}/models", timeout=10)
    log_path = data_dir / "logs" / "api.log"
    assert log_path.exists()
    lines = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    assert lines
    last = lines[-1]
    assert {"timestamp", "method", "path", "status_code"} <= set(last)


def test_mlflow_run_logged(base, data_dir):
    doc_id, _ = upload_txt(base)
    requests.post(
        f"{base}/analyze",
        json={"document_id": doc_id, "model_name": "summary-extractor", "analysis_type": "summary"},
        timeout=15,
    )
    # The file-based MLFlow store should now exist with the experiment.
    assert (data_dir / "mlruns").exists()


def test_metrics_endpoint(base):
    requests.get(f"{base}/health", timeout=10)
    r = requests.get(f"{base}/metrics", timeout=10)
    assert r.status_code == 200
    assert "flask_http_request" in r.text


def test_auth_enforced_when_keys_set(tmp_path):
    port = _free_port()
    proc = _boot(tmp_path, port, DOCANALYSIS_API_KEYS="secret1,secret2")
    try:
        base = f"http://127.0.0.1:{port}"
        # health + metrics stay open
        assert requests.get(f"{base}/health", timeout=10).status_code == 200
        assert requests.get(f"{base}/metrics", timeout=10).status_code == 200
        # models requires a key now
        assert requests.get(f"{base}/models", timeout=10).status_code == 401
        assert requests.get(f"{base}/models", headers={"X-API-Key": "wrong"}, timeout=10).status_code == 401
        ok = requests.get(f"{base}/models", headers={"X-API-Key": "secret2"}, timeout=10)
        assert ok.status_code == 200
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _boot(data_dir, port, **env_extra):
    import os

    env = {
        **os.environ,
        "DOCANALYSIS_DATA_DIR": str(data_dir),
        "DOCANALYSIS_SAMPLE_DIR": str(SAMPLE_DIR),
        "DOCANALYSIS_PORT": str(port),
        "GIT_PYTHON_REFRESH": "quiet",
        **env_extra,
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "docanalysis"],
        cwd=PROJECT_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 40
    while time.time() < deadline:
        try:
            if requests.get(f"{base}/health", timeout=1).status_code == 200:
                return proc
        except requests.RequestException:
            time.sleep(0.3)
    proc.terminate()
    raise RuntimeError("server did not start")
