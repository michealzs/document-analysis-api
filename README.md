# docanalysis

A Flask document-analysis API with **MLFlow tracking**. Upload PDF/TXT
documents, then run chunked analysis (summary, keywords, sentiment) against a
choice of context-window models — with result caching, batch processing,
comparison, and a full operational layer (auth, metrics, structured logging,
Docker, monitoring, CI/CD).

Every analysis is logged as an MLFlow run (parameters + metrics) under a
file-based tracking store.

## Install

```bash
pip install -e ".[dev]"          # dev = tests + lint
```

This installs the `docanalysis` console script.

## Run

```bash
# dev server
docanalysis                       # or: python -m docanalysis
# production
gunicorn docanalysis.wsgi:app -c gunicorn_conf.py
```

The service listens on `0.0.0.0:5000`. All paths derive from
`DOCANALYSIS_DATA_DIR` (default `./data`): uploads, results, logs, the SQLite
DB, and the MLFlow store all live under it.

## Endpoints

| Method & path | Purpose |
| --- | --- |
| `GET /health` | Liveness probe |
| `GET /metrics` | Prometheus metrics |
| `POST /documents` | Upload a PDF/TXT (`multipart` `file`) → `{document_id, word_count}` |
| `GET /documents/<id>` | Document metadata |
| `GET /documents` | List with `limit`/`offset`/`sort_by`/`order` → `{documents, total}` |
| `DELETE /documents/<id>` | Delete document + cascade its analyses |
| `GET /models` | The three built-in models |
| `POST /analyze` | Analyze a document (`summary`/`keywords`/`sentiment`); cached |
| `GET /results/<analysis_id>` | Full results, or one chunk via `?chunk_index=` |
| `POST /batch/analyze` | Run a `jobs` array sequentially; optional `webhook_url` |
| `POST /compare` | Compare two analyses (common keywords, sentiment agreement, chunk delta) |

```bash
curl -F "file=@sample_data/sample.txt" localhost:5000/documents
curl -s localhost:5000/analyze -H 'content-type: application/json' \
  -d '{"document_id":"<id>","model_name":"context-analyzer-v1","analysis_type":"keywords"}'
```

## Configuration

| Env var | Default | Meaning |
| --- | --- | --- |
| `DOCANALYSIS_DATA_DIR` | `./data` | Root for uploads/results/logs/db/mlruns |
| `DOCANALYSIS_SAMPLE_DIR` | `<data>/sample_data` | Sample documents |
| `DOCANALYSIS_MLRUNS_URI` | `file://<data>/mlruns` | MLFlow tracking URI |
| `DOCANALYSIS_API_KEYS` | — | Comma-separated keys; unset ⇒ auth open |
| `DOCANALYSIS_CACHE_TTL` | `300` | Cache TTL (seconds) |
| `DOCANALYSIS_LOG_LEVEL` | `INFO` | Log level |
| `WEB_CONCURRENCY` | `2` | Gunicorn workers (Docker) |

### Auth

Set `DOCANALYSIS_API_KEYS` to require an `X-API-Key` header on every endpoint
except `/health` and `/metrics`. Unset ⇒ open (dev).

## Tests

```bash
pytest
```

The suite boots the app against a temporary data dir and exercises the full
HTTP contract (upload, analysis, caching, batch, compare, pagination, error
handling) plus the ops layer (health, metrics, auth).

## Docker & monitoring

```bash
docker build -t docanalysis .
docker run -p 5000:5000 -v docanalysis_data:/app/data docanalysis

# full stack with Prometheus + Grafana
docker compose --profile monitoring up -d --build
```

- Metrics: http://localhost:5000/metrics
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin) — auto-provisioned dashboard

## License

MIT.
