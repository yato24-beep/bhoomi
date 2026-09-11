"""
src/database/postgis_ext.py
Real PostgreSQL/PostGIS/pgvector schema bootstrap.

This module is intentionally kept separate from src/database/models.py so that the
core SQLAlchemy ORM models remain fully portable between SQLite (offline/dev) and
PostgreSQL (demo/production). PostGIS Geometry columns and pgvector Vector columns
are not addressed via the ORM layer here (to avoid needing SpatiaLite for local
SQLite development); instead this module issues raw DDL/DML against Postgres only,
and every call here is a no-op (with a clear log line) when the active engine is
not PostgreSQL.

Nothing in here is invoked unless init_db() detects a postgresql:// DATABASE_URL.
"""

from typing import Optional
from loguru import logger
from sqlalchemy import text
from sqlalchemy.engine import Engine

EMBEDDING_DIM = 128


def is_postgres(engine: Engine) -> bool:
    """True only for a real PostgreSQL connection (never SQLite)."""
    return engine.dialect.name == "postgresql"


def ensure_extensions(engine: Engine) -> bool:
    """
    Enables the postgis and vector extensions on the connected database.
    Requires the connecting role to have CREATE privilege (superuser or rds_superuser
    on managed Postgres). Safe to call repeatedly (IF NOT EXISTS).
    Returns True if both extensions are confirmed active, False otherwise.
    """
    if not is_postgres(engine):
        logger.info("[POSTGIS_EXT] Skipping extension setup - not connected to PostgreSQL (using SQLite fallback).")
        return False

    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        logger.info("[POSTGIS_EXT] postgis and vector extensions confirmed active.")
        return True
    except Exception as e:
        logger.warning(
            f"[POSTGIS_EXT] Could not create postgis/vector extensions ({e}). "
            f"Ask your DB admin to run: CREATE EXTENSION postgis; CREATE EXTENSION vector; "
            f"GIS/duplicate detection will fall back to in-Python Shapely/TF-IDF logic."
        )
        return False


def create_postgis_schema(engine: Engine) -> None:
    """
    Adds a real PostGIS geometry column to cadastral_parcels and creates a
    pgvector-backed document_embeddings table for semantic duplicate detection.
    Idempotent - safe to run on every startup.
    """
    if not is_postgres(engine):
        return

    try:
        with engine.begin() as conn:
            # Real PostGIS geometry column (SRID 4326 / WGS84), alongside the existing
            # portable geometry_geojson JSON column already on the ORM model.
            conn.execute(text(
                "ALTER TABLE cadastral_parcels "
                "ADD COLUMN IF NOT EXISTS geom geometry(Polygon, 4326);"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_cadastral_parcels_geom "
                "ON cadastral_parcels USING GIST (geom);"
            ))
            conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_cadastral_parcels_lookup "
                "ON cadastral_parcels (state, district, tehsil, village, khasra_number);"
            ))

            # pgvector table for semantic duplicate detection (real vector similarity,
            # replacing the in-memory TF-IDF fallback used under SQLite).
            conn.execute(text(
                f"CREATE TABLE IF NOT EXISTS document_embeddings ("
                f"  document_id VARCHAR(128) PRIMARY KEY, "
                f"  embedding vector({EMBEDDING_DIM}) NOT NULL, "
                f"  created_at TIMESTAMPTZ DEFAULT now()"
                f");"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_document_embeddings_cosine "
                "ON document_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
            ))
        logger.info("[POSTGIS_EXT] cadastral_parcels.geom and document_embeddings(pgvector) schema ready.")
    except Exception as e:
        logger.warning(f"[POSTGIS_EXT] Could not create PostGIS/pgvector schema objects: {e}")


def sync_parcel_geometry(engine: Engine, parcel_id: int, geojson_geometry: dict) -> bool:
    """
    Populates/updates the real PostGIS geom column for one cadastral_parcels row
    from a GeoJSON geometry dict. Returns True on success.
    """
    if not is_postgres(engine):
        return False
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE cadastral_parcels "
                    "SET geom = ST_SetSRID(ST_GeomFromGeoJSON(:geojson), 4326) "
                    "WHERE id = :pid;"
                ),
                {"geojson": _geojson_to_str(geojson_geometry), "pid": parcel_id},
            )
        return True
    except Exception as e:
        logger.warning(f"[POSTGIS_EXT] Failed to sync geometry for parcel id={parcel_id}: {e}")
        return False


def _geojson_to_str(geojson_geometry: dict) -> str:
    import json
    return json.dumps(geojson_geometry)
