import os
from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.config import settings
from app.services.minio_storage import minio_storage

router = APIRouter()


class HealthCheckResponse(BaseModel):
    """Schema for health check status response."""
    status: str = Field(default="healthy", description="Operational status of the API")
    project_name: str = Field(description="Name of the running project")
    version: str = Field(description="Current application version")
    environment: str = Field(description="Deployment environment (development, staging, production)")
    database: str = Field(description="PostgreSQL connectivity status (connected or error details)")
    minio: str = Field(description="MinIO object storage status (connected or disconnected)")
    gemini: str = Field(default="unconfigured", description="Gemini AI configuration status ('configured' or 'unconfigured')")
    timestamp: str = Field(description="Current server UTC timestamp in ISO 8601 format")


class GeminiConfigResponse(BaseModel):
    """Schema for Gemini health and configuration status without exposing secrets."""
    status: str = Field(description="Configuration status: 'configured' or 'unconfigured'")
    configured: bool = Field(description="Whether a valid GEMINI_API_KEY is configured")
    engine: str = Field(description="Configured semantic engine identifier")
    model: str = Field(description="Configured Gemini model name")


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Service Health Check",
    description="Returns operational status including PostgreSQL, MinIO, and Gemini configuration verification.",
)
def check_health(db: Session = Depends(get_db)) -> HealthCheckResponse:
    """Perform a health check verifying API, PostgreSQL, MinIO storage, and Gemini status."""
    # 1. Check PostgreSQL
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        db_status = f"disconnected ({str(exc).splitlines()[0]})"

    # 2. Check Storage (MinIO or local filesystem fallback)
    if minio_storage.check_health():
        storage_status = "minio (connected)"
        storage_ok = True
    else:
        storage_status = "local_filesystem (connected)"
        storage_ok = True

    # 3. Check Gemini configuration without exposing secret key
    gemini_key = os.environ.get("GEMINI_API_KEY") or getattr(settings, "GEMINI_API_KEY", None)
    gemini_configured = bool(gemini_key and str(gemini_key).strip())
    gemini_status = "configured" if gemini_configured else "unconfigured"

    # Overall system health
    is_healthy = (db_status == "connected") and storage_ok
    overall_status = "healthy" if is_healthy else "degraded"

    return HealthCheckResponse(
        status=overall_status,
        project_name=settings.PROJECT_NAME,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        database=db_status,
        minio=storage_status,
        gemini=gemini_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get(
    "/health/gemini",
    response_model=GeminiConfigResponse,
    summary="Gemini Semantic Engine Configuration Check",
    description="Reports whether Gemini is configured without exposing API keys.",
)
def check_gemini_config() -> GeminiConfigResponse:
    """Check whether Gemini is configured without exposing secrets."""
    gemini_key = os.environ.get("GEMINI_API_KEY") or getattr(settings, "GEMINI_API_KEY", None)
    configured = bool(gemini_key and str(gemini_key).strip())
    engine = os.environ.get("SEMANTIC_ENGINE") or getattr(settings, "SEMANTIC_ENGINE", "gemini")
    model = os.environ.get("SEMANTIC_MODEL_NAME") or getattr(settings, "SEMANTIC_MODEL_NAME", "gemini-3.1-flash-lite")

    return GeminiConfigResponse(
        status="configured" if configured else "unconfigured",
        configured=configured,
        engine=str(engine),
        model=str(model),
    )
