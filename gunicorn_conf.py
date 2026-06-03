"""Gunicorn config for the document-analysis API.

MLFlow + SQLite writes are not safe to share across forked workers without care,
and the in-process cache/cleanup thread are per-worker. Workers default to
``WEB_CONCURRENCY`` (or 2); for heavy concurrency, front with a shared DB
(set a server-side store) and an external cache.
"""
from __future__ import annotations

import os

bind = f"0.0.0.0:{os.environ.get('DOCANALYSIS_PORT', '5000')}"
workers = int(os.environ.get("WEB_CONCURRENCY", "2"))
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("DOCANALYSIS_LOG_LEVEL", "info").lower()
