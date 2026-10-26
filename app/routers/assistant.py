from __future__ import annotations

import re
import time
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from app.config import settings
from app.models.ollama_model import OllamaDDLModel, OllamaUnavailable
from app.schemas import AssistantEnvelope, AssistantRequest, AssistantResponseData, AssistantSummary
from app.security import require_api_key
from app.services.learned_ddl import get_own_ddl_model
from app.services.scope_gate import classify_request
from app.services.session_store import record_turn, resolve_followup
from app.services.validator import validate_sql
from app.services.schema_compiler import resolve_dialect
from training.sql_format import build_ddl_prompt


router = APIRouter(dependencies=[Depends(require_api_key)])


def _summary_text(objects: list[str], sql: str | None, status: str) -> str:
    if objects:
        return f"{status}. Database objects: {', '.join(objects)}."
    if sql:
        return f"{status}. Generated DDL was returned and validated."
    return f"{status}. No DDL was generated."


def _compact_warnings(warnings: list[str] | None) -> list[str]:
    """Keep diagnostics useful without returning parser dumps to clients."""
    compact: list[str] = []
    for warning in warnings or []:
        cleaned = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(warning))
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned and cleaned not in compact:
            compact.append(cleaned[:240] + ("…" if len(cleaned) > 240 else ""))
    return compact[:2]


def _base_response(*, status: str, code: str, intent: str, dialect: str, model: str, request_id: str, latency_ms: float, message: str, sql: str | None = None, objects: list[str] | None = None, validation: dict | None = None, warnings: list[str] | None = None, version: str | None = None) -> AssistantResponseData:
    return AssistantResponseData(
        status=status,
        code=code,
        intent=intent,
        dialect=dialect,
        sql=sql,
        operation="DDL" if sql else None,
        object_names=objects or [],
        validation=validation or {},
        warnings=_compact_warnings(warnings),
        message=message,
        model=model,
        model_version=version,
        request_id=request_id,
        latency_ms=latency_ms,
    )


@router.post("/api/v1/generate", response_model=AssistantEnvelope)
def generate_assistant(payload: AssistantRequest) -> AssistantEnvelope:
    started = time.perf_counter()
    request_id = payload.request_id or str(uuid4())
    session_id = payload.session_id or f"session-{uuid4().hex[:16]}"
    context = resolve_followup(session_id, payload.prompt, "generate_ddl")
    resolved_prompt = context.resolved_requirement or payload.prompt
    dialect, _ = resolve_dialect(payload.dialect, resolved_prompt)
    decision = classify_request(resolved_prompt)
    selected_model = payload.model or settings.assistant_default_model

    if not decision.accepted:
        status = "clarification_required" if decision.intent == "NEEDS_CLARIFICATION" else "rejected"
        message = decision.message
        response = _base_response(
            status=status,
            code=decision.intent,
            intent=decision.intent,
            dialect=dialect,
            model=selected_model,
            request_id=request_id,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            message=message,
        )
        turn = record_turn(
            session_id=session_id,
            request_id=request_id,
            requirement=payload.prompt,
            resolved_requirement=resolved_prompt,
            response=response.model_dump(),
            dialect=dialect,
            model=selected_model,
        )
        return AssistantEnvelope(
            response=response,
            summary=AssistantSummary(
                session_id=session_id,
                turn=turn,
                memory_used=context.memory_used,
                remember=_summary_text([], None, message),
                active_objects=context.table_names,
            ),
        )

    prompt = build_ddl_prompt(
        requirement=resolved_prompt,
        dialect=dialect,
        task="modify_ddl" if context.memory_used and re.search(r"\b(add|alter|modify|change|rename|remove)\b", resolved_prompt, re.I) else "generate_ddl",
        schema=payload.schema_context or context.schema_context,
        memory_summary=(f"Existing objects: {', '.join(context.table_names)}" if context.table_names else ""),
    )
    try:
        if selected_model == "slm":
            try:
                model = get_own_ddl_model()
            except FileNotFoundError as exc:
                raise HTTPException(status_code=503, detail="The own DDL SLM is not trained yet. Run training.prepare_ddl and training.train_lm with --task ddl.") from exc
            raw = model.generate(prompt, max_new_tokens=payload.max_new_tokens, temperature=payload.temperature)
            version = model.version or "own-ddl-unversioned"
        else:
            model = OllamaDDLModel(settings.ollama_url, settings.ollama_model, settings.ollama_timeout_seconds)
            raw = model.generate(prompt, max_new_tokens=payload.max_new_tokens)
            version = model.version
    except OllamaUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    result = validate_sql(raw, dialect, allow_destructive=False, schema_context=payload.schema_context or context.schema_context)
    status = "accepted" if result.valid_syntax and result.policy_allowed else "validation_failed"
    response = _base_response(
        status=status,
        code="DDL_GENERATED" if result.valid_syntax else "VALIDATION_FAILED",
        intent=decision.intent,
        dialect=dialect,
        model=selected_model,
        request_id=request_id,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        message="DDL generated by the selected model." if result.valid_syntax else "The model output was not valid DDL.",
        # Never expose malformed model text as executable SQL.  The raw text
        # remains available through validation warnings and the request id for
        # diagnostics, while clients receive a stable nullable SQL field.
        sql=result.sql if result.valid_syntax and result.policy_allowed else None,
        objects=result.referenced_objects,
        validation={"status": result.validation_status, "syntax_valid": result.valid_syntax, "policy_allowed": result.policy_allowed},
        warnings=result.warnings,
        version=version,
    )
    turn = record_turn(
        session_id=session_id,
        request_id=request_id,
        requirement=payload.prompt,
        resolved_requirement=resolved_prompt,
        response=response.model_dump(),
        table_names=result.referenced_objects,
        dialect=dialect,
        model=selected_model,
    )
    return AssistantEnvelope(
        response=response,
        summary=AssistantSummary(
            session_id=session_id,
                turn=turn,
                memory_used=context.memory_used,
                remember=_summary_text(result.referenced_objects, result.sql, status),
                active_objects=result.referenced_objects or context.table_names,
            ),
    )


@router.get("/api/v1/assistant/model-info")
def assistant_model_info() -> dict[str, object]:
    try:
        model = get_own_ddl_model()
    except FileNotFoundError:
        return {"id": "slm", "provider": "local_pytorch", "available": False, "artifact_dir": str(settings.own_ddl_model_dir)}
    return {"id": "slm", "provider": "local_pytorch", "available": True, "version": model.version, "artifact_dir": str(settings.own_ddl_model_dir)}
