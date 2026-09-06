"""
src/main.py
FastAPI application entry point.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.router import router as api_router
from src.database.db_session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes database schema on startup."""
    init_db()
    yield


app = FastAPI(
    title="Land Record AI/ML Processing Pipeline",
    description="Person C: Structured Extraction, Normalization, Rule & Cross-Record Validation, Cadastral GIS, Duplicate Detection, and Field-Level Confidence Scoring.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
