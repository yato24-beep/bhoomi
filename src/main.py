"""
src/main.py
Root FastAPI application entry point, re-exporting the integrated platform application.
"""
from backend.app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
