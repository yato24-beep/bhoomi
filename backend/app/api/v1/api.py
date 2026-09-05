from fastapi import APIRouter
from app.api.v1 import health, documents, auth

api_router = APIRouter()

# Health endpoints
api_router.include_router(health.router, tags=["Health"])

# Authentication & User endpoints
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])

# Documents & Extraction endpoints
api_router.include_router(documents.router, prefix="/documents", tags=["Documents"])
