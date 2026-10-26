from __future__ import annotations

import re
import time
import uuid

from fastapi import HTTPException

from app.config import settings
from app.prompts import build_prompt
from app.models.ollama_model import OllamaUnavailable
from app.services.model_router import select_model
from app.services.schema_compiler import compile_ddl_request, resolve_dialect
from app.services.session_store import record_turn, resolve_followup
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


def _effective_task(requested_task: str, requirement: str, memory_used: bool) -> str:
    """Infer a modification task for natural follow-ups that omit task metadata."""
    if requested_task != "generate_ddl":
        return requested_task
    if re.search(r"\b(?:add|alter|modify|change)\b", requirement, re.IGNORECASE) and (
        re.search(r"\b(?:column|columns|field|fields|index|indexes)\b", requirement, re.IGNORECASE)
        or memory_used
    ):
        return "modify_ddl"
    return requested_task


def _structured_result(
    *,
    dialect: str,
    model_name: str,
    generation_method: str,
    compiled,
    validation: ValidationResult,
) -> dict:
    plan = dict(compiled.plan) if compiled is not None else {}
    object_type = str(plan.pop("object_type", "ddl"))
    operation = str(plan.pop("operation", "ddl"))
    name = plan.pop("name", None)
    columns = plan.pop("columns", [])
    relationships = plan.pop("relationships", [])
    return {
        "operation": operation,
        "object_type": object_type,
        "name": name,
        "dialect": dialect,
        "ddl": validation.sql if validation.valid_syntax and validation.policy_allowed else None,
        "columns": columns,
        "relationships": relationships,
        "validation": {
            "status": validation.validation_status,
            "syntax_valid": validation.valid_syntax,
            "policy_allowed": validation.policy_allowed,
            "warnings": validation.warnings,
        },
        "details": {
            **plan,
            "model": model_name,
            "generation_method": generation_method,
        },
    }


def generate_ddl(payload, *, legacy: bool = False) -> dict:
    started = time.perf_counter()
    request_id = payload.request_id or str(uuid.uuid4())
    session_context = resolve_followup(payload.session_id, payload.requirement, payload.task)
    requirement = session_context.resolved_requirement or payload.requirement
    task = _effective_task(payload.task, requirement, session_context.memory_used)
    classification_requirement = requirement
    if task == "modify_ddl" and session_context.memory_used and not re.search(
        r"\b(?:table|column|field|index|schema|constraint|view)\b", requirement, re.IGNORECASE
    ):
        # "add phone" is a valid follow-up when the previous turn established
        # the table; add the implied object only for scope classification.
        classification_requirement = f"{requirement} column"
    decision = classify_request(classification_requirement)
    dialect, dialect_source = resolve_dialect(payload.dialect, requirement)
    if payload.dialect is None and dialect_source == "default" and session_context.previous_dialect:
        dialect = session_context.previous_dialect
        dialect_source = "session"
    model_name = payload.model or session_context.previous_model or settings.default_model
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
        "session_id": session_context.session_id,
        "conversation_turn": 0,
        "result": None,
        "metadata": {
            "matched_signals": list(decision.matched_signals),
            "scope_confidence": decision.confidence,
            "memory_used": session_context.memory_used,
            "resolved_requirement": requirement,
            "previous_requirement": session_context.previous_requirement,
            "effective_task": task,
        },
    }
    if not decision.accepted:
        if session_context.session_id:
            base["conversation_turn"] = record_turn(
                session_id=session_context.session_id,
                request_id=request_id,
                requirement=payload.requirement,
                resolved_requirement=requirement,
                response=base,
                dialect=dialect,
                model=model_name,
            )
        base["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return base

    model = None
    generation_method = "scratch_autoregressive"
    compiled = None
    if model_name == "scratch":
        # Use the typed local plan before loading a checkpoint.  Common table
        # requests are deterministic and should remain available even when the
        # optional scratch checkpoint has not been trained or installed yet.
        compiled = compile_ddl_request(requirement, dialect, task)
    if compiled is not None:
        raw = compiled.output
        generation_method = compiled.source
    else:
        # Advanced scratch requests and all Ollama requests require their
        # selected inference backend at this point.
        model = select_model(model_name)
        context_parts = [part for part in (payload.schema_context, session_context.schema_context) if part]
        effective_schema_context = "\n".join(context_parts)
        prompt = build_prompt(requirement, dialect, task, effective_schema_context, intent=decision.intent)
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
        schema_context="\n".join(part for part in (payload.schema_context, session_context.schema_context) if part),
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
            "result": _structured_result(
                dialect=dialect,
                model_name=model_name,
                generation_method=generation_method,
                compiled=compiled,
                validation=result,
            ),
        }
    )
    base["metadata"].update(
        {
            "generation_method": generation_method,
            "dialect_source": dialect_source,
            "compiler_confidence": compiled.confidence if compiled is not None else None,
        }
    )
    if session_context.session_id:
        base["conversation_turn"] = record_turn(
            session_id=session_context.session_id,
            request_id=request_id,
            requirement=payload.requirement,
            resolved_requirement=requirement,
            response=base,
            table_names=compiled.table_names if compiled is not None else result.referenced_objects,
            plan=compiled.plan if compiled is not None else {},
            dialect=dialect,
            model=model_name,
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
