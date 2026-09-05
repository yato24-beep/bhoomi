import io
import time
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

import app.main
import app.api.v1.documents
import app.api.v1.health
import app.services.minio_storage
import app.workers.tasks
from app.main import app as fastapi_app
from app.db.base import Base
from app.api.deps import get_db
from app.workers.celery_app import celery_app
from app.core.security import get_password_hash, create_access_token
from app.models.user import User, UserRole

# Use in-memory SQLite database with StaticPool for test isolation
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Patch main engine and worker sessionmaker during tests
app.main.engine = test_engine
app.main.SessionLocal = TestingSessionLocal
app.workers.tasks.SessionLocal = TestingSessionLocal

# Configure Celery in EAGER mode for instantaneous synchronous test execution
celery_app.conf.update(
    task_always_eager=True,
    task_eager_propagates=True,
)


@pytest.fixture(autouse=True)
def mock_sleep_in_tests(monkeypatch):
    """Bypass time.sleep in all tests for instantaneous test execution."""
    monkeypatch.setattr(time, "sleep", lambda x: None)


class MockMinIOObject(io.BytesIO):
    """Mock MinIO object that supports read(), seek(), and stream() methods."""
    
    def stream(self, chunk_size: int = 32768):
        self.seek(0)
        while True:
            chunk = self.read(chunk_size)
            if not chunk:
                break
            yield chunk


class MockMinIOStorageService:
    """Mock MinIO client for in-memory, zero-dependency unit tests."""

    def __init__(self):
        self.files = {}
        self.bucket_name = "documents"

    def ensure_bucket_exists(self) -> bool:
        return True

    def check_health(self) -> bool:
        return True

    def upload_file(self, file_stream, filename, file_hash, file_size, content_type="application/octet-stream") -> str:
        file_stream.seek(0)
        data = file_stream.read()
        file_stream.seek(0)
        key = f"uploads/{file_hash[:16]}_{filename}"
        self.files[key] = (data, content_type)
        return key

    def get_file_object(self, object_name: str):
        if object_name not in self.files:
            raise FileNotFoundError(f"Object {object_name} not found")
        data, _ = self.files[object_name]
        return MockMinIOObject(data)

    def get_presigned_download_url(self, object_name: str, expires_seconds: int = 3600) -> str:
        return f"http://localhost:9000/{self.bucket_name}/{object_name}?token=mock_presigned_token"

    def delete_file(self, object_name: str) -> None:
        self.files.pop(object_name, None)


# Instantiate and patch mock MinIO across modules
mock_minio = MockMinIOStorageService()
app.services.minio_storage.minio_storage = mock_minio
app.api.v1.documents.minio_storage = mock_minio
app.api.v1.health.minio_storage = mock_minio
app.main.minio_storage = mock_minio
app.workers.tasks.minio_storage = mock_minio


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create all database tables before tests and tear down after."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session():
    """Yields a clean database session for each test."""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def client(db_session):
    """FastAPI TestClient with overridden get_db dependency pointing to the test DB."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    fastapi_app.dependency_overrides[get_db] = override_get_db
    with TestClient(fastapi_app) as test_client:
        yield test_client
    fastapi_app.dependency_overrides.clear()


def create_test_user(db_session, email: str, role: str) -> User:
    stmt = select(User).where(User.email == email)
    existing = db_session.execute(stmt).scalar_one_or_none()
    if existing:
        return existing

    user = User(
        email=email,
        hashed_password=get_password_hash("testpassword123"),
        full_name=f"Test {role.capitalize()}",
        role=role,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def admin_headers(db_session):
    user = create_test_user(db_session, "admin_test@doc.com", UserRole.ADMIN.value)
    token = create_access_token(subject=user.id, role=user.role, email=user.email)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def officer_headers(db_session):
    user = create_test_user(db_session, "officer_test@doc.com", UserRole.OFFICER.value)
    token = create_access_token(subject=user.id, role=user.role, email=user.email)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def reviewer_headers(db_session):
    user = create_test_user(db_session, "reviewer_test@doc.com", UserRole.REVIEWER.value)
    token = create_access_token(subject=user.id, role=user.role, email=user.email)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def viewer_headers(db_session):
    user = create_test_user(db_session, "viewer_test@doc.com", UserRole.VIEWER.value)
    token = create_access_token(subject=user.id, role=user.role, email=user.email)
    return {"Authorization": f"Bearer {token}"}
