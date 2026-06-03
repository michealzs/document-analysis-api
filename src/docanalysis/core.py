"""Domain logic: models, caching, chunking, analysis, extraction, DB, webhooks.

This is a faithful refactor of the reference implementation — the behaviour is
unchanged, only reorganised into importable, testable units and decoupled from
hard-coded paths.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

MODELS = [
    {"name": "context-analyzer-v1", "context_window": 2048, "version": "1.0.0"},
    {"name": "long-doc-processor", "context_window": 4096, "version": "2.1.0"},
    {"name": "summary-extractor", "context_window": 8192, "version": "1.5.2"},
]


# --------------------------------------------------------------------------- #
# TTL LRU cache
# --------------------------------------------------------------------------- #
class LRUCache:
    def __init__(self, capacity: int = 100, ttl: int = 300) -> None:
        self.cache: OrderedDict = OrderedDict()
        self.capacity = capacity
        self.ttl = ttl
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key not in self.cache:
                return None
            entry = self.cache[key]
            if time.time() - entry["timestamp"] > self.ttl:
                del self.cache[key]
                return None
            self.cache.move_to_end(key)
            return entry["data"]

    def put(self, key, value) -> None:
        with self._lock:
            self.cache[key] = {"data": value, "timestamp": time.time()}
            self.cache.move_to_end(key)
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def purge_expired(self) -> int:
        now = time.time()
        with self._lock:
            expired = [k for k, e in self.cache.items() if now - e["timestamp"] > self.ttl]
            for k in expired:
                del self.cache[k]
        return len(expired)


def cache_key(document_id, model_name, analysis_type, context_window) -> str:
    raw = f"{document_id}:{model_name}:{analysis_type}:{context_window}"
    return hashlib.md5(raw.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Chunking + analysis
# --------------------------------------------------------------------------- #
def _sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def chunk_text(text: str, context_window: int) -> list[str]:
    """Group sentences into chunks under ``context_window`` words, never
    splitting a sentence; an over-long sentence becomes its own chunk."""
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in _sentences(text):
        words = len(sentence.split())
        if count > 0 and count + words > context_window:
            chunks.append(" ".join(current))
            current = [sentence]
            count = words
        else:
            current.append(sentence)
            count += words
    if current:
        chunks.append(" ".join(current))
    return chunks


def analyze_chunk(idx: int, chunk: str, analysis_type: str) -> dict:
    if analysis_type == "summary":
        return {"chunk_index": idx, "chunk_text": chunk, "summary": chunk[:100]}
    if analysis_type == "keywords":
        words = chunk.split()
        return {
            "chunk_index": idx,
            "chunk_text": chunk,
            "keywords": list({w.lower() for w in words if len(w) > 5})[:10],
        }
    # sentiment
    return {"chunk_index": idx, "chunk_text": chunk, "sentiment": "neutral"}


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #
def extract_text_from_pdf(filepath: str) -> str:
    from PyPDF2 import PdfReader

    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    if not text.strip():
        raise ValueError("Could not extract text from PDF")
    return text


def extract_text_from_txt(filepath: str) -> str:
    with open(filepath, encoding="utf-8") as f:
        return f.read()


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #
def init_db(db_path: str | Path) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                analysis_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                context_window INTEGER NOT NULL,
                num_chunks INTEGER NOT NULL,
                mlflow_run_id TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at)")
        conn.commit()
    finally:
        conn.close()


def get_db(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# --------------------------------------------------------------------------- #
# Webhook delivery
# --------------------------------------------------------------------------- #
def deliver_webhook(webhook_url: str, payload: dict) -> bool:
    import requests

    delays = [1, 2, 4, 8]
    for attempt, delay in enumerate(delays):
        try:
            resp = requests.post(webhook_url, json=payload, timeout=10)
            if resp.status_code < 500:
                return True
        except Exception:
            pass
        if attempt < len(delays) - 1:
            time.sleep(delay)
    return False


def log_stderr(msg: str) -> None:
    print(msg, file=sys.stderr)
