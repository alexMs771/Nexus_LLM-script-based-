# SQL DDL SLM

An auditable, CPU-first research service for English-to-DDL generation. The project trains a small decoder-only Transformer from random initialization, evaluates it on held-out DDL and scope cases, and keeps an Ollama pretrained model as a separately measured baseline.

This is not a general chatbot and it does not execute generated SQL.

The local `scratch` serving path requires the trained checkpoint. For common natural-language table requests it combines the local checkpoint selection with a deterministic, dialect-aware schema planner and SQL/DBML validation, so requests such as `Create a table employee` can produce safe PostgreSQL DDL by default. The optional `ollama` path is separate and is used only when explicitly selected in an API request.

## What is implemented

- PostgreSQL-first DDL scope covering tables, schemas, indexes, views, `ALTER TABLE`, identifiers, types, defaults, nullability, primary/composite keys, foreign keys, unique/check constraints, and dialect conversion for PostgreSQL, SQLite, and MySQL.
- Independent hybrid scope gate with five outcomes: `IN_SCOPE`, `NEEDS_CLARIFICATION`, `UNSUPPORTED_SQL_OPERATION`, `OUT_OF_SCOPE`, and `RESTRICTED_OR_DESTRUCTIVE`.
- Separately trainable character n-gram intent classifier, deterministic safety/operation policy, diverse cross-domain examples, and an unseen-domain intent benchmark.
- Lossless byte tokenizer and corpus-trained byte-BPE tokenizer with serialized artifacts.
- Custom causal self-attention Transformer, next-token loss, clipping, checkpointing, validation loss/perplexity, resume-ready metadata, and autoregressive generation.
- SQLGlot syntax/policy validation after generation. Destructive operations are flagged and execution is disabled by default.
- Shared held-out benchmark and metric report for the scratch model. Ollama evaluation is opt-in and records when the service is unavailable rather than fabricating a score.
- FastAPI endpoints under `/api/v1`, plus `/generate` and `/health` compatibility routes.

## Reproduce the local pilot

```powershell
cd D:\Nexus_LLM\sql-llm
.\.venv\Scripts\python.exe -m training.prepare_data
.\.venv\Scripts\python.exe -m training.train_intent_classifier
.\.venv\Scripts\python.exe -m training.tokenizer_compare
.\.venv\Scripts\python.exe -m training.train_scratch --out-dir artifacts\runs\pilot
.\.venv\Scripts\python.exe -m training.evaluate
.\.venv\Scripts\python.exe -m pytest -q
```

The default pilot is deliberately small for a 16 GB, CPU-only machine. It writes a verified experiment under `artifacts/runs/pilot`. To make it the API model, train to `artifacts/scratch` or copy a verified run there only after reviewing its report. Large checkpoints are ignored by Git.

## Run the API

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Useful requests:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/intent/classify -Method Post -ContentType 'application/json' -Body '{"requirement":"What is Python?"}'
Invoke-RestMethod http://127.0.0.1:8000/api/v1/ddl/validate -Method Post -ContentType 'application/json' -Body '{"dialect":"postgresql","sql":"CREATE TABLE customers (id BIGINT PRIMARY KEY);"}'
```

`POST /api/v1/ddl/generate` returns structured rejection or clarification responses without calling a model when the scope gate blocks the request. When generation runs, the response includes the model id/version, validation status, warnings, referenced objects, request id, and latency.

## Ollama baseline

The baseline is not used for scratch training. Install Ollama separately, verify the model/license for the environment, pull `qwen2.5-coder:0.5b`, then run:

```powershell
python -m training.ollama_baseline --variant minimal
python -m training.ollama_baseline --variant common
```

If Ollama is not running, the report says `blocked`; it does not claim a result. The optional 1.5B candidate is a separate hardware experiment and is not enabled by default.

## Project map

```text
app/                 API, scope gate, validator, model adapters
training/            dataset preparation, tokenizers, scratch training, evaluation
data/examples/       hand-reviewed seed records
data/processed/      generated splits, manifest, reports
artifacts/            local checkpoints; large files are ignored
docs/                 feasibility, API, deployment, and data notes
tests/                regression and safety tests
```

See [docs/feasibility.md](docs/feasibility.md), [docs/api.md](docs/api.md), and [docs/deployment.md](docs/deployment.md) for the limits and security assumptions. Customer schemas must not enter the corpus without explicit authorization.
