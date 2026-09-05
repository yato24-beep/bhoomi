from app.config import settings


def test_root_endpoint(client):
    """Verify that the root endpoint returns 200 and points to docs and health check."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["documentation"] == "/docs"
    assert data["health_check"] == "/health"


def test_health_check_endpoint(client):
    """Verify that the root /health endpoint returns healthy status and metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["project_name"] == settings.PROJECT_NAME
    assert data["version"] == settings.VERSION
    assert data["environment"] == settings.ENVIRONMENT
    assert data["database"] == "connected"
    assert data["minio"] == "connected"
    assert "timestamp" in data


def test_versioned_health_check_endpoint(client):
    """Verify that the versioned /api/v1/health endpoint also returns healthy status."""
    response = client.get(f"{settings.API_V1_STR}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert data["minio"] == "connected"
