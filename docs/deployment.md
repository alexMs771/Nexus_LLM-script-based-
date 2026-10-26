# Deployment profiles

## Local

```powershell
python -m training.prepare_data
python -m training.train_scratch --out-dir artifacts/scratch
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The default bind address is loopback. DDL is validated but never executed by this service. Ollama is the default inference provider; install Ollama, pull the configured model, and keep the local Ollama server running.

## Hosted

Put the API behind HTTPS, authentication, a reverse proxy, request/rate limits, and a private inference network. Mount versioned model artifacts read-only. Use `GET /api/v1/ready` for readiness and retain only privacy-safe request ids, latency, model version, and validation outcomes.

## Production SaaS

Keep tenant authorization outside the model, use per-tenant schema metadata, least-privileged credentials, secret management, quotas, audit logs with redaction, artifact checksums, backups, rollbackable model versions, and disposable database tests. Do not expose a raw Ollama or PyTorch port publicly. Execution is intentionally absent from the API.
