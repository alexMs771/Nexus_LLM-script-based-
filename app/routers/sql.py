from __future__ import annotations

import time
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from app.config import settings
from app.schemas import OwnSQLRequest, OwnSQLResponse
from app.security import require_api_key
from app.services.learned_sql import get_own_sql_model
from app.services.validator import validate_sql
from training.sql_format import build_sql_prompt


router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/generate-sql", response_model=OwnSQLResponse)
@router.post("/api/v1/sql/generate", response_model=OwnSQLResponse)
def generate_sql(payload: OwnSQLRequest) -> OwnSQLResponse:
    started = time.perf_counter()
    request_id = payload.request_id or str(uuid4())
    try:
        model = get_own_sql_model()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "The own NL-to-SQL checkpoint is not available. Prepare data and run "
                "training/train_lm.py before calling /generate-sql."
            ),
        ) from exc
    prompt = build_sql_prompt(payload.question, payload.schema_context, payload.dialect)
    generated = model.generate(prompt, max_new_tokens=payload.max_new_tokens, temperature=payload.temperature)
    result = validate_sql(generated, payload.dialect, allow_destructive=False, schema_context=payload.schema_context)
    return OwnSQLResponse(
        status="generated" if result.valid_syntax else "invalid",
        generated_sql=result.sql or generated.strip(),
        dialect=payload.dialect,
        validation_status=result.validation_status,
        warnings=result.warnings,
        model_id="own-sql",
        model_version=model.version or "unversioned-checkpoint",
        request_id=request_id,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
    )


@router.get("/api/v1/sql/model-info")
def sql_model_info() -> dict[str, object]:
    try:
        model = get_own_sql_model()
    except FileNotFoundError:
        return {"id": "own-sql", "provider": "local_pytorch", "available": False, "artifact_dir": str(settings.own_model_dir)}
    return {
        "id": "own-sql",
        "provider": "local_pytorch",
        "available": True,
        "version": model.version,
        "artifact_dir": str(settings.own_model_dir),
    }
