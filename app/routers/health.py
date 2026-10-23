from fastapi import APIRouter

from app.schemas import HealthResponse
from app.config import settings
from app.services.model_router import health_model_label, loaded_models, ollama_available

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=health_model_label(),
        loaded_models=sorted(loaded_models()),
    )


@router.get("/api/v1/health", response_model=HealthResponse)
def api_health_check() -> HealthResponse:
    return health_check()


@router.get("/api/v1/ready")
def readiness_check():
    scratch_ready = bool(loaded_models().get("scratch"))
    ollama_ready = ollama_available() if settings.default_model == "ollama" else False
    return {
        "ready": ollama_ready if settings.default_model == "ollama" else scratch_ready,
        "default_model": settings.default_model,
        "loaded_models": sorted(loaded_models()),
        "ollama_available": ollama_ready,
    }
