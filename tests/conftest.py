"""Shared fixtures.

A single app instance is booted as a subprocess against a temporary data
directory and reused for the session; each test gets clean uploads via unique
filenames. The bundled ``sample_data`` is used as input.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_ROOT / "sample_data"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start(data_dir: Path, port: int, **env_extra) -> subprocess.Popen:
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
        cwd=PROJECT_ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    port = _free_port()
    proc = _start(data_dir, port)
    try:
        yield {"base": f"http://127.0.0.1:{port}", "data_dir": data_dir}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def base(server):
    return server["base"]


@pytest.fixture
def data_dir(server):
    return server["data_dir"]


def upload_txt(base, name="sample.txt"):
    """Upload the sample txt under a unique-ish name; return (doc_id, word_count)."""
    with open(SAMPLE_DIR / "sample.txt", "rb") as f:
        resp = requests.post(f"{base}/documents", files={"file": (name, f, "text/plain")}, timeout=15)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    return body["document_id"], body["word_count"]
