from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.config import settings
from app.schemas import (
    DDLConvertRequest,
    DDLGenerationRequest,
    DDLExplainRequest,
    DDLValidationRequest,
    GenerationResponse,
    IntentRequest,
    IntentResponse,
    SQLGenerationRequest,
    SQLGenerationResponse,
)
from app.services.model_router import model_inventory
from app.services.request_engine import classify_payload, generate_ddl
from app.services.session_store import delete_session, get_session
from app.services.validator import convert_sql, validate_sql
from app.security import require_api_key

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/api/v1/intent/classify", response_model=IntentResponse)
def classify(payload: IntentRequest):
    return classify_payload(payload.requirement)


@router.post("/api/v1/ddl/generate", response_model=GenerationResponse)
def generate(payload: DDLGenerationRequest):
    return generate_ddl(payload)


@router.post("/api/v1/ddl/validate")
def validate(payload: DDLValidationRequest):
    result = validate_sql(payload.sql, payload.dialect, allow_destructive=payload.allow_destructive, schema_context=payload.schema_context)
    return {
        "valid_syntax": result.valid_syntax,
        "validation_status": result.validation_status,
        "dialect": payload.dialect,
        "normalized_sql": result.sql,
        "statement_types": result.statement_types,
        "referenced_objects": result.referenced_objects,
        "destructive": result.destructive,
        "policy_allowed": result.policy_allowed,
        "warnings": result.warnings,
        "explanation": result.explanation,
    }


@router.post("/api/v1/ddl/convert")
def convert(payload: DDLConvertRequest):
    try:
        output = convert_sql(payload.sql, payload.source_dialect, payload.target_dialect)
    except Exception as exc:
        # The response intentionally avoids parser internals and stack traces.
        raise HTTPException(status_code=422, detail="The SQL could not be converted for the requested dialect.") from exc
    return {"source_dialect": payload.source_dialect, "target_dialect": payload.target_dialect, "sql": output}


@router.post("/api/v1/ddl/explain")
def explain(payload: DDLExplainRequest):
    if payload.requirement and not payload.sql:
        decision = classify_payload(payload.requirement)
        return {
            "status": decision["status"], "intent": decision["intent"],
            "explanation": "A database-structure explanation was requested. Supply SQL to receive statement-level details.",
            "message": decision["message"], "warnings": [],
        }
    if not payload.sql:
        raise HTTPException(status_code=422, detail="Supply either SQL or a DDL explanation requirement.")
    result = validate_sql(payload.sql, payload.dialect, allow_destructive=False)
    if not result.valid_syntax or not result.policy_allowed:
        return {"status": "validation_failed", "message": "The supplied SQL is not valid for the requested dialect.", "warnings": result.warnings}
    return {
        "status": "accepted",
        "dialect": payload.dialect,
        "statement_types": result.statement_types,
        "referenced_objects": result.referenced_objects,
        "explanation": "This DDL changes database structure. Review the statements and validate them in a disposable database before applying them.",
        "warnings": result.warnings,
    }


@router.get("/api/v1/models")
def models():
    return {"models": model_inventory(), "default_model": settings.default_model}


@router.get("/api/v1/sessions/{session_id}")
def session(session_id: str):
    try:
        state = get_session(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return state


@router.delete("/api/v1/sessions/{session_id}")
def clear_session(session_id: str):
    try:
        deleted = delete_session(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"status": "deleted", "session_id": session_id}


@router.post("/generate", response_model=SQLGenerationResponse)
def legacy_generate(payload: SQLGenerationRequest):
    result = generate_ddl(payload, legacy=True)
    return SQLGenerationResponse(
        model=payload.model,
        sql=result.get("generated_ddl") or result.get("sql"),
        explanation=result.get("explanation", ""),
        valid_syntax=result.get("validation_status") == "valid",
        warnings=result.get("warnings", []),
        status=result["status"],
        intent=result["intent"],
        code=result["code"],
        request_id=result["request_id"],
        session_id=result.get("session_id"),
    )
