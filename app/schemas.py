from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


IntentName = Literal[
    "IN_SCOPE",
    "NEEDS_CLARIFICATION",
    "UNSUPPORTED_SQL_OPERATION",
    "OUT_OF_SCOPE",
    "RESTRICTED_OR_DESTRUCTIVE",
]
DialectName = Literal["postgresql", "sqlite", "mysql", "dbml"]
ModelName = Literal["scratch", "ollama"]


class HealthResponse(BaseModel):
    status: str
    model: str
    loaded_models: list[str]
    version: str = "1.0.0"


class ModelInfo(BaseModel):
    id: str
    provider: str
    available: bool
    version: str = ""
    details: str = ""


class IntentRequest(BaseModel):
    requirement: str = Field(min_length=1, max_length=4000)


class IntentResponse(BaseModel):
    status: Literal["accepted", "clarification_required", "rejected"]
    intent: IntentName
    code: str
    message: str
    confidence: float
    matched_signals: list[str] = []


class DDLGenerationRequest(BaseModel):
    requirement: str = Field(min_length=1, max_length=4000)
    # Omitted dialect is resolved from the request; PostgreSQL is the fallback.
    dialect: DialectName | None = None
    model: ModelName | None = None
    task: Literal["generate_ddl", "explain_ddl", "modify_ddl"] = "generate_ddl"
    schema_context: str = Field(default="", max_length=12000)
    max_new_tokens: int = Field(default=192, ge=8, le=512)
    request_id: str | None = Field(default=None, max_length=100)
    session_id: str | None = Field(default=None, max_length=100, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,99}$")


class DDLValidationRequest(BaseModel):
    sql: str = Field(min_length=1, max_length=20000)
    dialect: DialectName = "postgresql"
    schema_context: str = Field(default="", max_length=12000)
    allow_destructive: bool = False


class DDLConvertRequest(BaseModel):
    sql: str = Field(min_length=1, max_length=20000)
    source_dialect: DialectName = "postgresql"
    target_dialect: DialectName = "postgresql"


class DDLExplainRequest(BaseModel):
    sql: str | None = Field(default=None, max_length=20000)
    requirement: str | None = Field(default=None, max_length=4000)
    dialect: DialectName = "postgresql"


class StructuredGenerationResult(BaseModel):
    """Compact, stable result view for clients such as Postman and frontends."""

    operation: str
    object_type: str
    name: str | None = None
    dialect: DialectName
    ddl: str | None = None
    columns: list[Any] = Field(default_factory=list)
    relationships: list[dict[str, Any]] = Field(default_factory=list)
    validation: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)


class GenerationResponse(BaseModel):
    status: Literal["accepted", "clarification_required", "rejected", "validation_failed"]
    code: str
    intent: IntentName
    generated_ddl: str | None = None
    sql: str | None = None
    explanation: str = ""
    dialect: DialectName
    referenced_objects: list[str] = []
    validation_status: Literal["valid", "invalid", "not_run"]
    warnings: list[str] = []
    model_id: str | None = None
    model_version: str | None = None
    request_id: str
    latency_ms: float
    message: str = ""
    session_id: str | None = None
    conversation_turn: int = 0
    result: StructuredGenerationResult | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SQLGenerationRequest(DDLGenerationRequest):
    """Backward-compatible alias for the original endpoint."""
    requirement: str = Field(min_length=5, max_length=4000)
    model: Literal["scratch", "ollama"] = "scratch"


class SQLGenerationResponse(BaseModel):
    model: str
    sql: str | None = None
    explanation: str = ""
    valid_syntax: bool
    warnings: list[str]
    status: str = "accepted"
    intent: IntentName = "IN_SCOPE"
    code: str = "IN_SCOPE"
    request_id: str = ""
    session_id: str | None = None


class OwnSQLRequest(BaseModel):
    """Request for the trained, local NL-to-SQL model only."""

    model_config = ConfigDict(populate_by_name=True)

    question: str = Field(min_length=1, max_length=12000)
    schema_context: str = Field(default="", max_length=30000, alias="schema")
    dialect: Literal["postgresql", "mysql", "sqlite"] = "postgresql"
    max_new_tokens: int = Field(default=256, ge=8, le=1024)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    request_id: str | None = Field(default=None, max_length=100)


class OwnSQLResponse(BaseModel):
    status: Literal["generated", "invalid"]
    generated_sql: str
    dialect: Literal["postgresql", "mysql", "sqlite"]
    validation_status: Literal["valid", "invalid"]
    warnings: list[str] = Field(default_factory=list)
    model_id: str
    model_version: str
    request_id: str
    latency_ms: float


AssistantModelName = Literal["slm", "ollama"]


class AssistantRequest(BaseModel):
    """Stable conversational contract for the learned DDL assistant."""

    prompt: str = Field(min_length=1, max_length=12000)
    model: AssistantModelName = "slm"
    session_id: str | None = Field(default=None, max_length=100, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,99}$")
    dialect: DialectName | None = None
    schema_context: str = Field(default="", max_length=30000, alias="schema")
    max_new_tokens: int = Field(default=256, ge=8, le=1024)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    request_id: str | None = Field(default=None, max_length=100)


class AssistantResponseData(BaseModel):
    """Compact public response for the learned assistant endpoint."""

    status: Literal["accepted", "clarification_required", "rejected", "validation_failed"]
    code: str
    intent: IntentName
    dialect: DialectName
    sql: str | None = None
    operation: str | None = None
    object_names: list[str] = Field(default_factory=list)
    validation: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    message: str = ""
    model: str


class AssistantSummary(BaseModel):
    """Minimal conversational state exposed to clients."""

    session_id: str
    turn: int
    memory_used: bool
    remember: str
    active_objects: list[str] = Field(default_factory=list)


class AssistantEnvelope(BaseModel):
    response: AssistantResponseData
    summary: AssistantSummary
    model_config = ConfigDict(populate_by_name=True)
