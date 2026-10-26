# Own NL-to-SQL model build

The repository now has two deliberately separate paths:

- The existing `/api/v1/ddl/generate` route remains for phase-1 compatibility.
- `/generate-sql` is the learned path. It loads only a local PyTorch checkpoint from `artifacts/models/own-sql-tiny`; it has no schema compiler or Ollama fallback.

## Reproducible workflow

1. Preserve `data/raw` and prepare records from the downloaded sources:

```powershell
.\.venv\Scripts\python.exe -m training.prepare_nl2sql
```

This creates `data/processed/nl2sql/{train,val,test,rejected}.jsonl`, a manifest, and a quality report. Rows are accepted only when the source contains both a natural-language question and a SQL answer. Missing schemas remain missing.

2. Run a bounded smoke training job:

```powershell
.\.venv\Scripts\python.exe -m training.train_lm `
  --config training/configs/tiny.json `
  --limit 128 --val-limit 64 --max-steps 10 `
  --out-dir artifacts/models/smoke-own-sql
```

3. Train the real run on GPU/Colab after reviewing the smoke output:

```powershell
.\.venv\Scripts\python.exe -m training.train_lm `
  --config training/configs/small.json `
  --out-dir artifacts/models/own-sql-tiny
```

The trainer starts from random weights, uses the byte tokenizer, masks the prompt portion of the NL-to-SQL loss, streams JSONL records, and writes resumable checkpoints. Continue an interrupted run with `--resume`.

For optional self-supervised domain pretraining before NL-to-SQL fine-tuning, first prepare `general_text.jsonl`, run `train_lm.py --task causal` into an artifact directory, then point the same directory at the NL-to-SQL files and continue with `--resume`. The trainer checks that architecture and tokenizer fields are compatible before resuming.

4. Evaluate the held-out split:

```powershell
.\.venv\Scripts\python.exe -m training.evaluate_nl2sql `
  --model-dir artifacts/models/own-sql-tiny
```

The report includes syntax accuracy, normalized exact match, source-level breakdowns, schema coverage, and failure samples. Execution accuracy is intentionally `null` until disposable database execution tests are added.

5. Start the service:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Postman request: `POST http://127.0.0.1:8001/generate-sql`, raw JSON:

```json
{
  "question": "List employees hired after 2020",
  "schema": "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, hire_date DATE);",
  "dialect": "postgresql",
  "max_new_tokens": 128,
  "temperature": 0
}
```

Before training, the endpoint returns `503` by design. That prevents an untrained or hardcoded result from being presented as a learned model.
