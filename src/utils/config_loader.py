"""
src/utils/config_loader.py
Loads pipeline and state-specific configuration files.
Supports automatic state detection from OCR text or upload-time state selection.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from loguru import logger


class ConfigLoader:
    """Manages loading of pipeline settings and state configurations."""

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        if base_dir is None:
            # Locate root configs directory
            self.base_dir = Path(__file__).resolve().parent.parent.parent / "configs"
        else:
            self.base_dir = Path(base_dir)

        self.states_dir = self.base_dir / "states"
        self._pipeline_config: Optional[Dict[str, Any]] = None
        self._state_cache: Dict[str, Dict[str, Any]] = {}

    def get_pipeline_config(self) -> Dict[str, Any]:
        """Load and cache main pipeline config."""
        if self._pipeline_config is None:
            config_file = self.base_dir / "pipeline.json"
            if config_file.is_file():
                with open(config_file, "r", encoding="utf-8") as f:
                    self._pipeline_config = json.load(f)
            else:
                logger.warning(f"Pipeline config not found at {config_file}, using defaults.")
                self._pipeline_config = {
                    "confidence_weights": {
                        "ocr_confidence": 0.25,
                        "pattern_strength": 0.20,
                        "rule_validation": 0.25,
                        "cross_record": 0.15,
                        "gis_consistency": 0.15,
                    },
                    "confidence_thresholds": {
                        "high_confidence_auto_approve": 0.85,
                        "medium_confidence_flag": 0.65,
                        "low_confidence_reject": 0.40,
                    },
                }
        return self._pipeline_config

    def get_state_config(self, state_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration for a specific state code (e.g. 'UP', 'MP', 'MH', 'BR').
        If missing, loads default.json.
        """
        code = (state_code or "default").lower().strip()

        if code in self._state_cache:
            return self._state_cache[code]

        STATE_NAME_MAP = {
            "ka": "karnataka",
            "karnataka": "karnataka",
            "tn": "tamilnadu",
            "tamilnadu": "tamilnadu",
            "tamil_nadu": "tamilnadu",
            "mh": "maharashtra",
            "maharashtra": "maharashtra",
            "br": "bihar",
            "bihar": "bihar",
            "up": "up",
            "uttar_pradesh": "up",
            "mp": "mp",
            "madhya_pradesh": "mp",
        }
        resolved_name = STATE_NAME_MAP.get(code, code)

        target_file = self.states_dir / f"{resolved_name}.json"
        if not target_file.is_file():
            # Try upper case or exact code
            target_file = self.states_dir / f"{code}.json"

        if not target_file.is_file():
            # Fall back to default
            logger.info(f"State config for '{state_code}' not found, falling back to default.json")
            target_file = self.states_dir / "default.json"

        if not target_file.is_file():
            raise FileNotFoundError(f"Neither {state_code}.json nor default.json found in {self.states_dir}")

        with open(target_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            self._state_cache[code] = cfg
            return cfg

    def detect_state_from_text(self, text: str) -> str:
        """
        Heuristic / keyword-based automatic state detection from OCR text.
        """
        text_lower = text.lower()
        
        # Check specific state markers
        if any(w in text_lower for w in [
            "ಕರ್ನಾಟಕ", "karnataka", "ಭೂಮಿ", "bhoomi", "ಪಹಣಿ", "pahani",
            "ಸರ್ವೆ ನಂ", "ಖಾತೆದಾರ", "ಬೆಂಗಳೂರು", "bangalore", "bengaluru",
            "bbmp", "mahanagara palike", "khatha", "ಪ್ರಮಾಣ ಪತ್ರ",
            "mutation", "mutation register", "rule 46", "ಮ್ಯುಟೇಶನ್",
        ]):
            return "KA"
        # Check Kannada script presence
        if any("\u0c80" <= ch <= "\u0cff" for ch in text):
            return "KA"
        # Check Tamil script presence
        if any("\u0b80" <= ch <= "\u0bff" for ch in text):
            return "TN"
        if any(w in text_lower for w in ["தமிழ்நாடு", "tamil nadu", "tamilnadu", "பட்டா", "patta", "சிட்டா", "chitta", "புல எண்"]):
            return "TN"
        if any(w in text_lower for w in ["उत्तर प्रदेश", "uttar pradesh", "खातौनी", "गाटा संख्या", "भौमिक अधिकार", "fasli"]):
            return "UP"
        if any(w in text_lower for w in ["मध्य प्रदेश", "madhya pradesh", "भू-अभिलेख", "भूमिस्वामी", "खसरा क्रमांक"]):
            return "MP"
        if any(w in text_lower for w in ["महाराष्ट्र", "maharashtra", "सातबारा", "satbara", "भोगवटादार", "गट क्रमांक", "हे.आर."]):
            return "MH"
        if any(w in text_lower for w in ["बिहार", "bihar", "जमाबंदी", "खतियान", "रैयत का नाम", "खेसरा संख्या"]):
            return "BR"

        return "DEFAULT"
