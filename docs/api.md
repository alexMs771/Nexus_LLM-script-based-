# API contract

- `POST /api/v1/ddl/generate` - scope-gated generation and post-generation validation. Omitted dialect defaults to PostgreSQL; supported formats are `postgresql`, `mysql`, `sqlite`, and `dbml`.
- `POST /api/v1/ddl/explain` - validate and summarize supplied DDL.
- `POST /api/v1/ddl/validate` - dialect-aware syntax and policy checks.
- `POST /api/v1/ddl/convert` - PostgreSQL, SQLite, and MySQL conversion through SQLGlot.
- `POST /api/v1/intent/classify` - independent five-class scope classification.
- `GET /api/v1/models` - configured model inventory.
- `GET /api/v1/sessions/{session_id}` - inspect stored session state.
- `DELETE /api/v1/sessions/{session_id}` - clear stored session memory.
- `GET /api/v1/health` and `GET /api/v1/ready` - liveness/readiness.
- `POST /generate-sql` (alias `/api/v1/sql/generate`) - learned local NL-to-SQL model. Body fields are `question`, optional `schema`, `dialect`, `max_new_tokens`, and `temperature`.
- `GET /api/v1/sql/model-info` - reports whether the own checkpoint is installed.

`OUT_OF_SCOPE`, `NEEDS_CLARIFICATION`, `UNSUPPORTED_SQL_OPERATION`, and `RESTRICTED_OR_DESTRUCTIVE` responses contain no generated DDL. The legacy `/generate` and `/health` routes remain as compatibility aliases.

The `scratch` option uses the validated local typed schema planner first for common table, multi-table, index, and `ADD COLUMN` requests. Requests outside those deterministic patterns fall through to the scratch checkpoint, so advanced generation still requires a trained/installed checkpoint. The `ollama` option calls the configured Ollama model only when explicitly selected.

For `/api/v1/ddl/generate`, use the compact `result` object for new clients. It contains the operation, object name(s), columns, relationships, rendered DDL, and validation details. The older top-level fields such as `generated_ddl`, `sql`, and `validation_status` remain for compatibility.

Follow-up memory is enabled by sending the same `session_id` on each request. The service stores the latest objects, dialect, selected model, resolved requirement, and turn history in SQLite at `SQL_LLM_SESSION_DB_PATH` (or the default `artifacts/sessions.sqlite3`). For example:

```json
{
  "session_id": "customer-design-001",
  "model": "scratch",
  "requirement": "Create a customer table with name and email"
}
```

Then send:

```json
{
  "session_id": "customer-design-001",
  "requirement": "Now add phone and date of birth columns to it"
}
```

The second request resolves `it` to `customer`, remembers the previous dialect/model when omitted, and generates `ALTER TABLE` without executing it against a database.

The learned `/generate-sql` route is intentionally independent of the phase-1 compiler and Ollama adapter. It returns `503` until a checkpoint exists at `SQL_LLM_OWN_MODEL_DIR` (default `artifacts/models/own-sql-tiny`); it never substitutes a hardcoded SQL response.
