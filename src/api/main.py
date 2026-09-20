"""FastAPI Backend for Land Record Digitization OCR & Handwriting Pipeline.

Exposes RESTful endpoints for:
1. Health Check (GET /health, GET /api/ocr/health)
2. Document & Crop OCR Processing (POST /api/ocr/process)

Reuses the existing production DocumentProcessingPipeline and LanguageScriptRouter
without duplicating any OCR or recognition logic.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict

from src.api.router import router as ocr_router, get_pipeline


def create_app() -> FastAPI:
    """Factory creating and configuring the standalone OCR FastAPI application instance."""
    app = FastAPI(
        title="Land Record Digitization OCR API",
        description=(
            "REST API for multimodal printed and handwritten land record OCR. "
            "Supports Printed Kannada (PaddleOCR), Handwritten Kannada (Fine-tuned TrOCR), "
            "Printed English (PaddleOCR), and Handwritten English (Pretrained TrOCR)."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Enable CORS for local development and web frontend integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", tags=["General"])
    async def root() -> Dict[str, Any]:
        """Root status and navigation helper."""
        return {
            "service": "Land Record Digitization OCR Service",
            "status": "operational",
            "docs": "/docs",
            "health": "/health",
            "process_endpoint": "POST /api/ocr/process",
        }

    @app.get("/health", tags=["Health"])
    async def health() -> Dict[str, Any]:
        from src.api.router import ocr_health_check
        return await ocr_health_check()

    from src.api.compat import router as compat_router

    app.include_router(ocr_router)
    app.include_router(compat_router)
    return app


# Module-level app instance for uvicorn (e.g. uvicorn src.api.main:app)
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
