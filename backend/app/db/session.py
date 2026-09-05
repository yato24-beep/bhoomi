from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from app.config import settings


def create_db_engine():
    """Create SQLAlchemy engine with quick connection check and automatic SQLite fallback."""
    if settings.DATABASE_URL and settings.DATABASE_URL.startswith("postgresql"):
        try:
            # Check PostgreSQL availability with a 2-second connection timeout
            pg_engine = create_engine(
                settings.DATABASE_URL,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                connect_args={"connect_timeout": 2},
            )
            with pg_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return pg_engine
        except Exception:
            print("[INFO] PostgreSQL not available on localhost:5432. Falling back to local SQLite database (doc_platform.db).")
            return create_engine("sqlite:///./doc_platform.db", connect_args={"check_same_thread": False})
    
    return create_engine(settings.DATABASE_URL or "sqlite:///./doc_platform.db", connect_args={"check_same_thread": False})


engine = create_db_engine()

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that provides a transactional database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
