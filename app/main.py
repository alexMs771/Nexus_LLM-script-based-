from fastapi import FastAPI

from app.routers import assistant, generate, health, sql

app = FastAPI(
    title="SQL LLM API",
    description="API for a SQL-focused language model supporting scratch training and pretrained fine-tuning.",
    version="1.0.0",
)

app.include_router(health.router)
app.include_router(generate.router)
app.include_router(sql.router)
app.include_router(assistant.router)


@app.get("/", include_in_schema=False)
def root() -> dict[str, object]:
    return {
        "service": "SQL DDL SLM API",
        "status": "ok",
        "docs": "/docs",
        "health": "/api/v1/health",
        "readiness": "/api/v1/ready",
        "models": "/api/v1/models",
        "learned_sql": "/generate-sql",
        "assistant": "/api/v1/generate",
    }
