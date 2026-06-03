"""Document-analysis API: a Flask service with MLFlow tracking."""
from .app import create_app
from .config import Config, load_config

__version__ = "0.1.0"
__all__ = ["create_app", "Config", "load_config"]
