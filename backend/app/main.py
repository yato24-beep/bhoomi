import os
import sys
from pathlib import Path

# Ensure OpenMP and PaddleX OneDNN flags are set before any ML libraries initialize
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")

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

# Import OCR routers
try:
    from src.api.router import router as ocr_router
except ImportError:
    ocr_router = None

try:
    from src.api.person_c_router import router as person_c_router
except ImportError:
    person_c_router = None


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

    # 3. Startup Subsystem Diagnostics
    print("=" * 70)
    print("[STARTUP DIAGNOSTICS] Verifying core subsystems...")

    # (a) EasyOCR
    try:
        from src.handwriting.easyocr_recognizer import get_easyocr_recognizer
        _ = get_easyocr_recognizer()
        print("  [1/4] EasyOCR Printed Engine        : READY (languages: kannada, english)")
    except Exception as exc:
        print(f"  [1/4] EasyOCR Printed Engine        : WARNING ({exc})")

    # (b) TrOCR-12000
    try:
        from src.handwriting.trocr_12000_recognizer import (
            DEFAULT_CHECKPOINT_12000_PATH,
            DEFAULT_TOKENIZER_NAME,
            get_trocr_12000_recognizer,
        )
        trocr = get_trocr_12000_recognizer(auto_load=False)
        repo_id = os.environ.get("MODEL_REPO_ID", "local/unconfigured")
        print(
            f"  [2/4] TrOCR-12000 Kannada Engine   : CONFIGURED "
            f"(repo: {repo_id}, model_path: {trocr.model_path}, tokenizer: {DEFAULT_TOKENIZER_NAME})"
        )
    except Exception as exc:
        print(f"  [2/4] TrOCR-12000 Kannada Engine   : WARNING ({exc})")

    # (c) Gemini
    gemini_key = os.environ.get("GEMINI_API_KEY") or getattr(settings, "GEMINI_API_KEY", None)
    gemini_engine = os.environ.get("SEMANTIC_ENGINE", getattr(settings, "SEMANTIC_ENGINE", "gemini"))
    gemini_model = os.environ.get("SEMANTIC_MODEL_NAME", getattr(settings, "SEMANTIC_MODEL_NAME", "gemini-3.1-flash-lite"))
    if gemini_key:
        masked_key = gemini_key[:6] + "..." + gemini_key[-4:] if len(gemini_key) > 10 else "***"
        print(
            f"  [3/4] Gemini Semantic Engine       : CONFIGURED "
            f"(engine: {gemini_engine}, model: {gemini_model}, key: {masked_key})"
        )
    else:
        print("  [3/4] Gemini Semantic Engine       : UNCONFIGURED (GEMINI_API_KEY absent; deterministic fallback active)")

    # (d) Document Classification Gate
    try:
        from src.classification.document_gate import LandRecordGateClassifier
        gate = LandRecordGateClassifier()
        print("  [4/4] Document Classification Gate : READY (supported: Bhoomi RTC, Pahani, Mutation, Form 16, Index II)")
    except Exception as exc:
        print(f"  [4/4] Document Classification Gate : WARNING ({exc})")
    print("=" * 70)

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

# Set up CORS middleware with FRONTEND_ORIGIN support
origins = list(settings.BACKEND_CORS_ORIGINS) if settings.BACKEND_CORS_ORIGINS else []
fe_env = os.environ.get("FRONTEND_ORIGIN") or os.environ.get("FRONTEND_ORIGINS") or getattr(settings, "FRONTEND_ORIGIN", None)
if fe_env:
    for item in fe_env.split(","):
        item_clean = item.strip()
        if item_clean and item_clean not in origins:
            origins.append(item_clean)

if "*" not in origins:
    for default_origin in [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]:
        if default_origin not in origins:
            origins.append(default_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if "*" not in origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Direct root health check (/health)
app.include_router(health_router, prefix="", tags=["Health"])

# Mount Person B Multimodal OCR API routes (/api/ocr/...)
if ocr_router:
    app.include_router(ocr_router, prefix="", tags=["OCR"])

# Mount Person C direct API routes (/api/v1/...)
if person_c_router:
    app.include_router(person_c_router)

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
