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
    timestamp: str = Field(description="Current server UTC timestamp in ISO 8601 format")


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Service Health Check",
    description="Returns operational status including PostgreSQL and MinIO connectivity verification.",
)
def check_health(db: Session = Depends(get_db)) -> HealthCheckResponse:
    """Perform a health check verifying API, PostgreSQL, and MinIO storage status."""
    # 1. Check PostgreSQL
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        db_status = f"disconnected ({str(exc).splitlines()[0]})"

    # 2. Check MinIO Object Storage
    minio_status = "connected" if minio_storage.check_health() else "disconnected"

    # Overall system health
    is_healthy = (db_status == "connected") and (minio_status == "connected")
    overall_status = "healthy" if is_healthy else "degraded"

    return HealthCheckResponse(
        status=overall_status,
        project_name=settings.PROJECT_NAME,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        database=db_status,
        minio=minio_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
