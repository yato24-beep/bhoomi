"""Pytest configuration and compatibility fixtures."""

import sys
import os

# Ensure Windows PyTorch DLLs and openmp flags
import src

# Compatibility patch for Starlette TestClient with HTTPX 0.28+
try:
    import httpx
    _orig_client_init = httpx.Client.__init__

    def _patched_client_init(self, *args, **kwargs):
        if "app" in kwargs:
            app = kwargs.pop("app")
            if "transport" not in kwargs:
                kwargs["transport"] = httpx.ASGITransport(app=app)
        _orig_client_init(self, *args, **kwargs)

    httpx.Client.__init__ = _patched_client_init
except Exception:
    pass
