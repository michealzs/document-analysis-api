"""Dev entrypoint: ``python -m docanalysis`` runs the Flask dev server.

Use gunicorn (``docanalysis.wsgi:app``) in production.
"""
from __future__ import annotations

import os

from .app import create_app


def main() -> None:
    app = create_app()
    host = os.environ.get("DOCANALYSIS_HOST", "0.0.0.0")
    port = int(os.environ.get("DOCANALYSIS_PORT", "5000"))
    app.run(host=host, port=port)


if __name__ == "__main__":
    main()
