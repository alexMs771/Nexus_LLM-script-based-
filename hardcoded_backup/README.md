# Hardcoded & Pilot Prototype Archive

This folder archives the deterministic, hardcoded, synthetic, and pilot prototype assets developed before transitioning to the full-scale neural LLM pipeline.

---

## How to Run & Demonstrate to Your Lead

You have three ready-to-run ways to showcase the hardcoded prototype results at any time:

### 1. Interactive CLI Console & Single Query Demo
Run a single query directly:
```powershell
python hardcoded_backup/demo.py "Create an employee table with id, name, and salary in PostgreSQL"
```

Or enter interactive mode (shows menu with presets and custom input prompt):
```powershell
python hardcoded_backup/demo.py
```

Run against the 9 pre-configured sample queries:
```powershell
python hardcoded_backup/demo.py --samples
```

### 2. Standalone Web UI & API Server (Port 8001)
To show a visual demo in the browser or via Swagger API:
```powershell
python hardcoded_backup/serve.py --port 8001
```
- **Web UI**: Open [http://127.0.0.1:8001](http://127.0.0.1:8001) in any browser (features dark mode, preset buttons, instant AST plan viewer, and DDL syntax validation).
- **Swagger Docs**: [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)
- **REST Endpoint**: `POST http://127.0.0.1:8001/api/v1/generate`

### 3. Automated Benchmark Report
Run an automated 10-point test suite covering PostgreSQL, MySQL, SQLite, DBML, Alter Table, Index, and Safety Scope Gate:
```powershell
python hardcoded_backup/benchmark.py
```
Outputs a clean ASCII table showing 100% syntax validity and ~4ms execution latency.

---

## Archived Components

1. **`app/services/schema_compiler.py`**:
   - Deterministic rule-based schema compiler that generated DDL via regular expressions, predefined templates, and heuristic AST builders.
2. **`app/services/scope_gate.py`**:
   - Heuristic / keyword-based scope classifier.
3. **`app/services/request_engine.py`**:
   - Router containing the bypass hook where `model_name == 'scratch'` diverted requests directly to `schema_compiler.py` instead of the neural model.
4. **`training/synthetic_data.py`**:
   - Handcrafted synthetic data generator relying on 5 fixed templates and 6 entity categories.
5. **`training/prepare_data.py`**:
   - Initial data preparation script that only ingested the synthetic records and hand-curated seed examples.
6. **`data/examples/`**:
   - Initial hand-written seed datasets (`ddl_examples.jsonl`, `invalid_ddl.jsonl`).
7. **`data/processed_pilot/`**:
   - Initial pilot dataset splits (`train.jsonl`, `val.jsonl`, `test.jsonl`, `manifest.json`, etc.).
8. **`artifacts/runs/pilot/`**:
   - Initial 588 KB pilot model weights, training history, and tokenizer.

---
*Archived on: 2026-10-24*
*Ready for Phase 2: Full-Scale Neural LLM Pipeline with SQaLe, Spider, SQL Create Context, and DB Documentation.*