"""Runtime configuration, all env-driven so the service runs anywhere.

The original task hard-coded ``/app`` paths. Here every path derives from a
single ``DOCANALYSIS_DATA_DIR`` (default ``./data``), so the same code runs in
dev, tests (temp dir), and Docker (a mounted volume) unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path


class Config:
    def __init__(self) -> None:
        self.data_dir = Path(os.environ.get("DOCANALYSIS_DATA_DIR", "data")).resolve()
        self.upload_dir = self.data_dir / "uploads"
        self.results_dir = self.data_dir / "results"
        self.logs_dir = self.data_dir / "logs"
        self.db_path = self.data_dir / "documents.db"
        self.mlruns_uri = os.environ.get(
            "DOCANALYSIS_MLRUNS_URI", f"file://{self.data_dir / 'mlruns'}"
        )
        self.experiment = os.environ.get("DOCANALYSIS_EXPERIMENT", "document-analysis")
        # Sample documents bundled with the package; overridable for Docker.
        self.sample_dir = Path(
            os.environ.get("DOCANALYSIS_SAMPLE_DIR", self.data_dir / "sample_data")
        )

        # Cache TTL (seconds) and capacity.
        self.cache_ttl = int(os.environ.get("DOCANALYSIS_CACHE_TTL", "300"))
        self.cache_capacity = int(os.environ.get("DOCANALYSIS_CACHE_CAPACITY", "100"))

        # Ops: API keys (comma-separated; empty ⇒ auth open) and optional throttle.
        _keys = os.environ.get("DOCANALYSIS_API_KEYS", "")
        self.api_keys = [k.strip() for k in _keys.split(",") if k.strip()]
        self.rate_limit = os.environ.get("DOCANALYSIS_RATE_LIMIT")  # e.g. "120/min"
        self.log_level = os.environ.get("DOCANALYSIS_LOG_LEVEL", "INFO").upper()

    @property
    def log_path(self) -> Path:
        return self.logs_dir / "api.log"

    def ensure_dirs(self) -> None:
        for d in (self.upload_dir, self.results_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    return Config()
