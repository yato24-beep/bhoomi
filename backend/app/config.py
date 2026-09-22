from typing import List, Union, Optional, Set
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # General Project Information
    PROJECT_NAME: str = "Document Processing Platform"
    PROJECT_DESCRIPTION: str = "Asynchronous Document Ingestion and Processing API"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    DEMO_MODE: bool = False
    PORT: int = 8000
    ALLOW_LOCALHOST_CORS: bool = False
    
    # API Routing
    API_V1_STR: str = "/api/v1"

    # Security & Authentication (JWT)
    SECRET_KEY: str = "super_secret_jwt_key_for_document_processing_platform_123456"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # CORS (Cross-Origin Resource Sharing)
    FRONTEND_ORIGIN: Optional[str] = None
    BACKEND_CORS_ORIGINS: Union[List[str], str] = [
        "https://bhoomi-karnataka.vercel.app",
        "https://bhoomi.vercel.app",
        "https://bhoomi-seven.vercel.app",
        "https://bhoomi-yato24-beeps-projects.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]], info) -> List[str]:
        origins = []
        if isinstance(v, str):
            v_str = v.strip()
            if v_str.startswith("[") and v_str.endswith("]"):
                import json
                try:
                    origins = json.loads(v_str)
                except Exception:
                    origins = [i.strip() for i in v_str.split(",") if i.strip()]
            else:
                origins = [i.strip() for i in v_str.split(",") if i.strip()]
        elif isinstance(v, list):
            origins = list(v)

        # Incorporate FRONTEND_ORIGIN and FRONTEND_ORIGINS if set in environment
        import os
        fe_env = os.environ.get("FRONTEND_ORIGIN") or os.environ.get("FRONTEND_ORIGINS")
        if fe_env:
            for item in fe_env.split(","):
                item_clean = item.strip()
                if item_clean and item_clean not in origins:
                    origins.append(item_clean)

        # In production, disallow wildcard '*' and only allow localhost if explicitly requested
        env = (os.environ.get("ENVIRONMENT") or "development").lower()
        if env == "production":
            origins = [o for o in origins if o != "*"]
            allow_local = (os.environ.get("ALLOW_LOCALHOST_CORS") or "false").lower() in ("true", "1", "yes")
            if not allow_local:
                origins = [o for o in origins if not ("localhost" in o or "127.0.0.1" in o)]

        return origins

    # PostgreSQL Database Configuration
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres_password"
    POSTGRES_DB: str = "doc_platform"
    DATABASE_URL: Optional[str] = None

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: Optional[str], info) -> str:
        if isinstance(v, str) and v.strip():
            return v
        values = info.data
        server = values.get("POSTGRES_SERVER", "localhost")
        port = values.get("POSTGRES_PORT", 5432)
        user = values.get("POSTGRES_USER", "postgres")
        password = values.get("POSTGRES_PASSWORD", "postgres_password")
        db = values.get("POSTGRES_DB", "doc_platform")
        return f"postgresql+psycopg://{user}:{password}@{server}:{port}/{db}"

    # MinIO Object Storage (S3-Compatible) Configuration
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ROOT_USER: str = "minioadmin"
    MINIO_ROOT_PASSWORD: str = "minioadmin_password"
    MINIO_BUCKET_NAME: str = "documents"
    MINIO_SECURE: bool = False

    # Redis & Celery Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None

    @field_validator("CELERY_BROKER_URL", mode="before")
    @classmethod
    def assemble_celery_broker(cls, v: Optional[str], info) -> str:
        if isinstance(v, str) and v.strip():
            return v
        values = info.data
        host = values.get("REDIS_HOST", "localhost")
        port = values.get("REDIS_PORT", 6379)
        db = values.get("REDIS_DB", 0)
        return f"redis://{host}:{port}/{db}"

    @field_validator("CELERY_RESULT_BACKEND", mode="before")
    @classmethod
    def assemble_celery_backend(cls, v: Optional[str], info) -> str:
        if isinstance(v, str) and v.strip():
            return v
        values = info.data
        host = values.get("REDIS_HOST", "localhost")
        port = values.get("REDIS_PORT", 6379)
        db = values.get("REDIS_DB", 0)
        return f"redis://{host}:{port}/{db}"

    # Supported File Formats
    ALLOWED_EXTENSIONS: Set[str] = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif"}
    ALLOWED_MIME_TYPES: Set[str] = {
        "application/pdf",
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/tiff",
        "application/octet-stream",
    }

    # Multimodal OCR & TrOCR Model Configuration
    MODEL_REPO_ID: Optional[str] = None
    MODEL_CHECKPOINT_DIR: Optional[str] = None
    TROCR_MODEL_DIR: Optional[str] = None
    KANNADA_HANDWRITING_MODEL_PATH: Optional[str] = None
    TROCR_PRETRAINED_MODEL_PATH: str = "microsoft/trocr-small-handwritten"
    DEFAULT_DEVICE: str = "auto"
    OCR_DEVICE: str = "auto"
    DEFAULT_OCR_LANGUAGE: str = "kannada"
    OCR_CONFIDENCE_THRESHOLD: float = 0.60
    PRINTED_OCR_ENGINE: str = "easyocr"
    TRANSLATION_HIGH_CONFIDENCE_THRESHOLD: float = 0.85
    TRANSLATION_MEDIUM_CONFIDENCE_THRESHOLD: float = 0.60

    # Semantic Layer Configuration (Gemini / Local Multilingual AI)
    SEMANTIC_ENGINE: str = "gemini"
    GEMINI_API_KEY: Optional[str] = None
    SEMANTIC_MODEL_NAME: str = "gemini-3.1-flash-lite"
    SEMANTIC_TIMEOUT_SECONDS: int = 30
    SEMANTIC_TEMPERATURE: float = 0.0



settings = Settings()

# Synchronize model directory configuration to os.environ for seamless submodule access
import os

if settings.MODEL_REPO_ID and "MODEL_REPO_ID" not in os.environ:
    os.environ["MODEL_REPO_ID"] = str(settings.MODEL_REPO_ID)
if settings.PRINTED_OCR_ENGINE and "PRINTED_OCR_ENGINE" not in os.environ:
    os.environ["PRINTED_OCR_ENGINE"] = str(settings.PRINTED_OCR_ENGINE)
if settings.TROCR_MODEL_DIR and "TROCR_MODEL_DIR" not in os.environ:
    os.environ["TROCR_MODEL_DIR"] = str(settings.TROCR_MODEL_DIR)
if settings.MODEL_CHECKPOINT_DIR and "MODEL_CHECKPOINT_DIR" not in os.environ:
    os.environ["MODEL_CHECKPOINT_DIR"] = str(settings.MODEL_CHECKPOINT_DIR)
if settings.KANNADA_HANDWRITING_MODEL_PATH and "KANNADA_HANDWRITING_MODEL_PATH" not in os.environ:
    os.environ["KANNADA_HANDWRITING_MODEL_PATH"] = str(settings.KANNADA_HANDWRITING_MODEL_PATH)
if settings.SEMANTIC_ENGINE and "SEMANTIC_ENGINE" not in os.environ:
    os.environ["SEMANTIC_ENGINE"] = str(settings.SEMANTIC_ENGINE)
if settings.GEMINI_API_KEY and "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = str(settings.GEMINI_API_KEY)
if settings.SEMANTIC_MODEL_NAME and "SEMANTIC_MODEL_NAME" not in os.environ:
    os.environ["SEMANTIC_MODEL_NAME"] = str(settings.SEMANTIC_MODEL_NAME)


