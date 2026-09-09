"""person-a/src/config/loader.py
Thread-safe cached loader for multi-state JSON configurations.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import ValidationError

from .models import StateConfiguration


class StateConfigLoader:
    """Loads and caches state configurations from configs/states/ directory."""

    def __init__(self, configs_dir: Optional[Path] = None):
        if configs_dir is None:
            # Default to person-a/configs/states
            self.configs_dir = Path(__file__).resolve().parent.parent.parent / "configs" / "states"
        else:
            self.configs_dir = Path(configs_dir)

        self._cache: Dict[str, StateConfiguration] = {}
        self._default_config: Optional[StateConfiguration] = None
        self._name_to_code: Dict[str, str] = {}
        self._load_all_configurations()

    def _load_all_configurations(self) -> None:
        if not self.configs_dir.is_dir():
            return

        for config_path in self.configs_dir.glob("*.json"):
            try:
                with config_path.open("r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                state_cfg = StateConfiguration.model_validate(raw_data)
                code = state_cfg.state_code.upper()
                self._cache[code] = state_cfg
                self._name_to_code[state_cfg.state_name.lower()] = code
                if code == "DEFAULT" or config_path.stem == "default":
                    self._default_config = state_cfg
            except (json.JSONDecodeError, ValidationError, OSError):
                continue

    def get_config(self, state_hint: Optional[str] = None, strict: bool = False) -> StateConfiguration:
        """Retrieve state configuration by state code or name with default fallback."""
        if state_hint:
            hint = state_hint.strip()
            code = hint.upper()
            if code in self._cache:
                return self._cache[code]
            lower_name = hint.lower()
            if lower_name in self._name_to_code:
                mapped_code = self._name_to_code[lower_name]
                return self._cache[mapped_code]

            if strict:
                raise KeyError(f"No configuration found for state '{state_hint}'")

        if self._default_config is not None:
            return self._default_config

        if "DEFAULT" in self._cache:
            return self._cache["DEFAULT"]

        return StateConfiguration(
            state_code="DEFAULT",
            state_name="All India Default",
            primary_language="en",
            fallback_languages=["hi"],
        )

    def list_available_states(self) -> List[str]:
        return sorted([k for k in self._cache.keys() if k != "DEFAULT"])


_loader_instance: Optional[StateConfigLoader] = None


def get_state_config(state_hint: Optional[str] = None, strict: bool = False) -> StateConfiguration:
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = StateConfigLoader()
    return _loader_instance.get_config(state_hint=state_hint, strict=strict)
