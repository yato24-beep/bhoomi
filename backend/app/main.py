import sys
from pathlib import Path

# Ensure backend directory and repository root are on sys.path
backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
for p in (str(backend_dir), str(repo_root)):
    if p not in sys.path:
        sys.path.insert(0, p)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.api.v1.api import api_router
from app.api.v1.health import router as health_router
from app.config import settings
from app.db.base import Base
from app.db.session import engine, SessionLocal
from app.services.minio_storage import minio_storage
from app.core.security import get_password_hash
from app.models.user import User, UserRole
import app.models  # Registers all models with Base.metadata

# Import OCR router from Person B
try:
    from src.api.router import router as ocr_router
except ImportError:
    ocr_router = None


def seed_initial_users():
    """Seed initial demonstration users for all RBAC roles on fresh startup."""
    db: Session = SessionLocal()
    try:
        stmt = select(User).limit(1)
        existing = db.execute(stmt).scalar_one_or_none()
        if not existing:
            demo_users = [
                User(
                    email="admin@docplatform.com",
                    hashed_password=get_password_hash("admin123"),
                    full_name="System Administrator",
                    role=UserRole.ADMIN.value,
                    is_active=True,
                ),
                User(
                    email="officer@docplatform.com",
                    hashed_password=get_password_hash("officer123"),
                    full_name="Ingestion Officer",
                    role=UserRole.OFFICER.value,
                    is_active=True,
                ),
                User(
                    email="reviewer@docplatform.com",
                    hashed_password=get_password_hash("reviewer123"),
                    full_name="Quality Reviewer",
                    role=UserRole.REVIEWER.value,
                    is_active=True,
                ),
                User(
                    email="viewer@docplatform.com",
                    hashed_password=get_password_hash("viewer123"),
                    full_name="Guest Stakeholder",
                    role=UserRole.VIEWER.value,
                    is_active=True,
                ),
            ]
            db.add_all(demo_users)
            db.commit()
            print("[INFO] Initial RBAC users seeded: admin@, officer@, reviewer@, viewer@ (password: <role>123)")
    except Exception as exc:
        print(f"[WARN] User seeding notice: {exc}")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    print(f"[STARTUP] Starting {settings.PROJECT_NAME} (v{settings.VERSION}) [{settings.ENVIRONMENT}]")
    
    # 1. Initialize database tables & seed initial users (PostgreSQL or SQLite fallback)
    try:
        Base.metadata.create_all(bind=engine)
        print("[OK] Database tables verified/created successfully.")
        seed_initial_users()
    except Exception as e:
        print(f"[WARN] Database initialization notice ({e}). Ensure database is accessible.")

    # 2. Verify / initialize MinIO storage bucket (or local filesystem fallback)
    try:
        if minio_storage.ensure_bucket_exists():
            print(f"[OK] Storage bucket '{settings.MINIO_BUCKET_NAME}' verified/ready.")
        else:
            print(f"[INFO] Storage operating in local filesystem fallback mode.")
    except Exception as e:
        print(f"[WARN] Storage initialization notice ({e}).")

    yield
    print(f"[SHUTDOWN] Shutting down {settings.PROJECT_NAME}")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Unified Land Record Digitization & Document Processing Platform. "
        "Integrates document ingestion, asynchronous extraction, RBAC authentication, "
        "and multimodal printed & handwritten OCR (PaddleOCR & TrOCR)."
    ),
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Set up CORS middleware
origins = list(settings.BACKEND_CORS_ORIGINS) if settings.BACKEND_CORS_ORIGINS else ["*"]
if "*" not in origins:
    origins.extend(["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000", "http://127.0.0.1:8000"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Direct root health check (/health)
app.include_router(health_router, prefix="", tags=["Health"])

# Mount Person B Multimodal OCR API routes (/api/ocr/...)
if ocr_router:
    app.include_router(ocr_router, prefix="", tags=["OCR"])

# Mount versioned API routes (/api/v1/...)
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", summary="Root Index", tags=["General"])
async def root():
    """Root endpoint welcoming clients and linking to documentation."""
    return JSONResponse(
        content={
            "message": f"Welcome to {settings.PROJECT_NAME} API",
            "version": settings.VERSION,
            "documentation": "/docs",
            "health_check": "/health",
            "ocr_health": "/api/ocr/health",
            "ocr_process": "/api/ocr/process",
            "documents_api": f"{settings.API_V1_STR}/documents",
            "auth_api": f"{settings.API_V1_STR}/auth",
        }
    )
