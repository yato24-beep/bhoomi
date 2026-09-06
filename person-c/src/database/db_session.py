"""
src/database/db_session.py
Database engine and session management.
Supports PostgreSQL/PostGIS connection with fallback to local SQLite for offline development.
"""

import os
from typing import Generator
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import Base

# Read database URL from environment or default to local sqlite file
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # Use SQLite for standalone local execution
    DATABASE_URL = "sqlite:///./land_records.db"

# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

# Create session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initializes database tables if they do not exist."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info(f"Database schema initialized successfully using {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL}")
    except Exception as e:
        logger.warning(f"Could not initialize database tables: {e}")
        return

    # Real PostGIS geometry column + pgvector embeddings table.
    # No-op (logged, not raised) when running against SQLite - the existing
    # in-Python Shapely / TF-IDF fallbacks remain fully functional there.
    from src.database.postgis_ext import ensure_extensions, create_postgis_schema
    if ensure_extensions(engine):
        create_postgis_schema(engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency generator for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
