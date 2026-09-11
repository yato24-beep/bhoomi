"""Land Record Digitization package."""

import os
import sys

# Ensure OpenMP and Paddle flags are configured
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")

# Windows DLL Resolution: ensure torch/lib is in DLL directory search path before torch/albumentations/paddleocr import
if sys.platform == "win32":
    import site
    search_paths = []
    try:
        search_paths.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        search_paths.append(site.getusersitepackages())
    except Exception:
        pass
    for sp in search_paths:
        tlib = os.path.join(sp, "torch", "lib")
        if os.path.isdir(tlib):
            try:
                os.add_dll_directory(tlib)
            except Exception:
                pass
            os.environ["PATH"] = tlib + ";" + os.environ.get("PATH", "")
    try:
        import torch
    except Exception:
        pass
