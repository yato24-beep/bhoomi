"""FastAPI Backend for Land Record Digitization OCR & Handwriting Pipeline.

Exposes RESTful endpoints for:
1. Health Check (GET /health, GET /api/ocr/health)
2. Document & Crop OCR Processing (POST /api/ocr/process)

Reuses the existing production DocumentProcessingPipeline and LanguageScriptRouter
without duplicating any OCR or recognition logic.
"""

from datetime import datetime
import io
import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from src.integration.document_pipeline import DocumentProcessingPipeline
from src.integration.schemas import DocumentProcessingResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr_api")

# Singleton pipeline instance to preserve cache and avoid redundant allocations
_PIPELINE_INSTANCE: Optional[DocumentProcessingPipeline] = None


def get_pipeline() -> DocumentProcessingPipeline:
    """Returns the shared singleton instance of DocumentProcessingPipeline."""
    global _PIPELINE_INSTANCE
    if _PIPELINE_INSTANCE is None:
        _PIPELINE_INSTANCE = DocumentProcessingPipeline()
    return _PIPELINE_INSTANCE


def create_app() -> FastAPI:
    """Factory creating and configuring the FastAPI application instance."""
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
    @app.get("/api/ocr/health", tags=["Health"])
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint exposing supported modalities and pipeline readiness."""
        pipeline = get_pipeline()
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "service": "land-record-ocr-pipeline",
            "version": "1.0.0",
            "supported_modalities": {
                "printed_kannada": "PaddleOCR (ppocr_v4_kannada)",
                "handwritten_kannada": "TrOCR (fine-tuned Kannada checkpoint)",
                "printed_english": "PaddleOCR (ppocr_v4_en)",
                "handwritten_english": "TrOCR (microsoft/trocr-small-handwritten)",
            },
            "supported_languages": pipeline.router.list_supported_languages(),
            "default_language": pipeline.default_language,
            "confidence_threshold": pipeline.confidence_threshold,
        }

    @app.post(
        "/api/ocr/process",
        response_model=DocumentProcessingResponse,
        status_code=status.HTTP_200_OK,
        tags=["OCR"],
        summary="Process document image through multimodal OCR pipeline",
    )
    async def process_document_image(
        file: UploadFile = File(..., description="Document or crop image file (PNG, JPG, JPEG, TIFF, BMP, WEBP)"),
        language: Optional[str] = Form("kannada", description="Target language (e.g. 'kannada', 'english')"),
        is_handwritten: Optional[bool] = Form(None, description="Handwriting flag: True=Handwritten, False=Printed, None=Auto-infer"),
        document_id: Optional[str] = Form(None, description="Optional caller-provided document identifier"),
        page_number: int = Form(1, ge=1, description="1-indexed page number"),
        regions: Optional[str] = Form(None, description="Optional JSON array of candidate bounding box regions from layout analysis"),
        apply_preprocessing: bool = Form(True, description="Whether to apply deskew/denoise/contrast enhancement"),
        apply_normalization: bool = Form(True, description="Whether to apply conservative text normalization"),
    ) -> DocumentProcessingResponse:
        """Processes an uploaded document image and returns structured OCR results.

        Accepts an uploaded image and passes it directly to the existing
        `DocumentProcessingPipeline` and `LanguageScriptRouter`.
        """
        # Step 1: Validate and read uploaded file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file must have a valid filename.",
            )

        try:
            file_bytes = await file.read()
            if not file_bytes or len(file_bytes) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded file is empty (0 bytes).",
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error reading uploaded file: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read uploaded file: {str(e)}",
            )

        # Step 2: Validate image content via PIL
        try:
            image_obj = Image.open(io.BytesIO(file_bytes))
            image_obj.load()  # Force decode to verify image validity
        except Exception as e:
            logger.warning(f"Invalid image format for file '{file.filename}': {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid or unreadable image format for file '{file.filename}'. Error: {str(e)}",
            )

        # Step 3: Parse optional regions JSON if provided
        parsed_regions: Optional[List[Any]] = None
        if regions and regions.strip():
            try:
                parsed = json.loads(regions)
                if not isinstance(parsed, list):
                    raise ValueError("The 'regions' field must be a valid JSON array/list.")
                parsed_regions = parsed
            except Exception as e:
                logger.warning(f"Invalid regions JSON parameter: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid 'regions' JSON: {str(e)}",
                )

        # Step 4: Execute existing DocumentProcessingPipeline
        pipeline = get_pipeline()
        assigned_doc_id = document_id or f"doc_{int(time.time() * 1000)}"

        try:
            response = pipeline.process_document(
                image=image_obj,
                regions=parsed_regions,
                is_handwritten=is_handwritten,
                language=language or "kannada",
                page_number=page_number,
                document_id=assigned_doc_id,
                image_path=file.filename,
                apply_preprocessing=apply_preprocessing,
                apply_normalization=apply_normalization,
            )
            return response
        except Exception as e:
            logger.error(f"Pipeline processing failure for document '{assigned_doc_id}': {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Document processing failed in OCR pipeline: {str(e)}",
            )

    return app


# Module-level app instance for uvicorn (e.g. uvicorn src.api.main:app)
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
