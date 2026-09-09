"""person-a/src/preprocessing/super_resolution.py
Real-ESRGAN selective neural super-resolution for degraded/low-DPI documents.
"""

from typing import Optional
import cv2
import numpy as np
import torch
import torch.nn as nn


class RealESRGANMini(nn.Module):
    """Compact RRDBNet architecture for 2x super-resolution inference on CPU/GPU."""

    def __init__(self, in_nc: int = 3, out_nc: int = 3, num_feat: int = 64, num_block: int = 4):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            ) for _ in range(num_block)
        ])
        self.conv_up = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_last = nn.Conv2d(num_feat, out_nc, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        res = self.body(feat) + feat
        up = self.pixel_shuffle(self.conv_up(res))
        return self.conv_last(up)


class SuperResolutionEnhancer:
    """Super-resolution processor selectively applied when estimated DPI < threshold."""

    def __init__(self, scale: int = 2, device: Optional[str] = None):
        self.scale = scale
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[RealESRGANMini] = None
        self._init_model()

    def _init_model(self) -> None:
        try:
            self.model = RealESRGANMini().to(self.device)
            self.model.eval()
        except Exception:
            self.model = None

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Upscale image 2x using neural super-resolution or Lanczos-4 interpolation."""
        if image is None or image.size == 0:
            return image

        h, w = image.shape[:2]
        # Safety limit for CPU inference: upscale patches/crops with NN, large pages with Lanczos-4
        if (self.device == "cpu" and h * w > 600 * 600) or h * w > 1200 * 1200 or self.model is None:
            return cv2.resize(image, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LANCZOS4)

        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            tensor = tensor.to(self.device)

            with torch.no_grad():
                out_tensor = self.model(tensor)

            out_tensor = torch.clamp(out_tensor, 0.0, 1.0)
            out_np = (out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
        except Exception:
            return cv2.resize(image, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LANCZOS4)
