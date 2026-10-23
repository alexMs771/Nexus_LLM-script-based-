from __future__ import annotations

import time
import uuid

from fastapi import HTTPException

from app.config import settings
from app.prompts import build_prompt
from app.models.ollama_model import OllamaUnavailable
from app.prompts import build_prompt
from app.services.model_router import select_model
from app.services.schema_compiler import compile_table_request, resolve_dialect
from app.services.scope_gate import ScopeDecision, classify_request
from app.services.validator import ValidationResult, validate_sql


def _decision_payload(decision: ScopeDecision) -> tuple[str, str]:
    if decision.intent == "NEEDS_CLARIFICATION":
        return "clarification_required", decision.message
    if decision.intent in {"OUT_OF_SCOPE", "UNSUPPORTED_SQL_OPERATION", "RESTRICTED_OR_DESTRUCTIVE"}:
        return "rejected", decision.message
    return "accepted", decision.message


def _decision_code(decision: ScopeDecision) -> str:
    return "UNSUPPORTED_OPERATION" if decision.intent == "UNSUPPORTED_SQL_OPERATION" else decision.intent


def generate_ddl(payload, *, legacy: bool = False) -> dict:
    started = time.perf_counter()
    request_id = payload.request_id or str(uuid.uuid4())
    decision = classify_request(payload.requirement)
    dialect, dialect_source = resolve_dialect(payload.dialect, payload.requirement)
    status, message = _decision_payload(decision)
    base = {
        "status": status,
        "code": _decision_code(decision),
        "intent": decision.intent,
        "generated_ddl": None,
        "sql": None,
        "explanation": "",
        "dialect": dialect,
        "referenced_objects": [],
        "validation_status": "not_run",
        "warnings": [],
        "model_id": None,
        "model_version": None,
        "request_id": request_id,
        "latency_ms": 0.0,
        "message": message,
        "metadata": {"matched_signals": list(decision.matched_signals), "scope_confidence": decision.confidence},
    }
    if not decision.accepted:
        base["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return base

    model_name = payload.model or settings.default_model
    model = select_model(model_name)
    generation_method = "scratch_autoregressive"
    compiled = None
    if model_name == "scratch":
        # The local checkpoint remains the selected model and is required to be
        # present.  For table requests, a typed local plan is used first so the
        # endpoint returns executable DDL even while the tiny scratch decoder is
        # still being improved.  More advanced requests continue to the model.
        compiled = compile_table_request(payload.requirement, dialect)
    if compiled is not None:
        raw = compiled.output
        generation_method = compiled.source
    else:
        prompt = build_prompt(payload.requirement, dialect, payload.task, payload.schema_context, intent=decision.intent)
        try:
            raw = model.generate(prompt, max_new_tokens=payload.max_new_tokens)
        except TypeError:
            raw = model.generate(prompt)
        except OllamaUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    result: ValidationResult = validate_sql(
        raw,
        dialect,
        allow_destructive=settings.allow_destructive,
        schema_context=payload.schema_context,
    )
    base.update(
        {
            "status": "accepted" if result.valid_syntax and result.policy_allowed else "validation_failed",
            "code": "DDL_GENERATED" if result.valid_syntax else "VALIDATION_FAILED",
            "generated_ddl": result.sql if result.valid_syntax and result.policy_allowed else None,
            "sql": result.sql if result.valid_syntax else raw,
            "explanation": result.explanation,
            "referenced_objects": result.referenced_objects,
            "validation_status": result.validation_status,
            "warnings": result.warnings,
            "model_id": model_name,
            "model_version": "scratch-local-hybrid" if compiled is not None else getattr(model, "version", model_name),
        }
    )
    base["metadata"].update(
        {
            "generation_method": generation_method,
            "dialect_source": dialect_source,
            "compiler_confidence": compiled.confidence if compiled is not None else None,
        }
    )
    base["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return base


def classify_payload(requirement: str) -> dict:
    decision = classify_request(requirement)
    status, message = _decision_payload(decision)
    return {
        "status": status,
        "intent": decision.intent,
        "code": _decision_code(decision),
        "message": message,
        "confidence": decision.confidence,
        "matched_signals": list(decision.matched_signals),
    }
