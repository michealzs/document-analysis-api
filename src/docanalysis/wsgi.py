"""WSGI entrypoint for gunicorn: ``gunicorn docanalysis.wsgi:app``."""
from .app import create_app

app = create_app()
