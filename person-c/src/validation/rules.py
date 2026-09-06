"""
src/validation/rules.py
Single-field rule validation against state-configured constraints and patterns.
"""

import re
from typing import Any, Dict, List, Tuple
from schemas import ExtractedField, RuleValidationItem, ValidationStatus


class RuleValidator:
    """
    Validates individual extracted fields against state configuration rules.
    """

    def __init__(self, state_config: Dict[str, Any]):
        self.config = state_config
        self.validation_rules = state_config.get("validation_rules", {})
        self.fields_def = state_config.get("fields", {})

    def validate_fields(
        self, fields: Dict[str, ExtractedField]
    ) -> Tuple[List[RuleValidationItem], ValidationStatus]:
        """
        Validates all extracted fields.
        Returns a list of validation items and an aggregate status.
        """
        validation_items: List[RuleValidationItem] = []
        has_error = False
        has_warning = False

        # 1. Check required fields
        for field_name, spec in self.fields_def.items():
            is_required = spec.get("required", False)
            if is_required and (field_name not in fields or not fields[field_name].raw_value):
                item = RuleValidationItem(
                    rule_name=f"required_field_{field_name}",
                    field_name=field_name,
                    passed=False,
                    severity="error",
                    message=f"Mandatory field '{field_name}' is missing from document extraction",
                    expected="Present",
                    actual="Missing",
                )
                validation_items.append(item)
                has_error = True

        # 2. Check field-specific validation rules
        for field_name, field_obj in fields.items():
            rule_spec = self.validation_rules.get(field_name)
            if not rule_spec:
                field_obj.validation_status = ValidationStatus.VALID
                continue

            field_errors = self._validate_single_field(field_obj, rule_spec)
            for err in field_errors:
                validation_items.append(err)
                field_obj.validation_messages.append(err.message)
                if not err.passed:
                    if err.severity == "error":
                        has_error = True
                        field_obj.validation_status = ValidationStatus.INVALID
                    elif err.severity == "warning" and field_obj.validation_status != ValidationStatus.INVALID:
                        has_warning = True
                        field_obj.validation_status = ValidationStatus.WARNING

            if not field_obj.validation_messages:
                field_obj.validation_status = ValidationStatus.VALID

        if has_error:
            aggregate_status = ValidationStatus.INVALID
        elif has_warning:
            aggregate_status = ValidationStatus.WARNING
        else:
            aggregate_status = ValidationStatus.VALID

        return validation_items, aggregate_status

    def _validate_single_field(
        self, field_obj: ExtractedField, rule_spec: Dict[str, Any]
    ) -> List[RuleValidationItem]:
        """Executes range, pattern, and length checks on a single field."""
        items: List[RuleValidationItem] = []
        val = field_obj.normalized_value

        # Numeric Range Check (e.g. land area)
        if "min" in rule_spec or "max" in rule_spec:
            try:
                num_val = float(val)
                min_val = rule_spec.get("min")
                max_val = rule_spec.get("max")

                if min_val is not None and num_val < min_val:
                    items.append(RuleValidationItem(
                        rule_name=f"min_range_{field_obj.field_name}",
                        field_name=field_obj.field_name,
                        passed=False,
                        severity="error",
                        message=f"Value {num_val} is below minimum allowed {min_val}",
                        expected=f">= {min_val}",
                        actual=str(num_val),
                    ))
                if max_val is not None and num_val > max_val:
                    items.append(RuleValidationItem(
                        rule_name=f"max_range_{field_obj.field_name}",
                        field_name=field_obj.field_name,
                        passed=False,
                        severity="error",
                        message=f"Value {num_val} exceeds maximum allowed {max_val}",
                        expected=f"<= {max_val}",
                        actual=str(num_val),
                    ))
            except (ValueError, TypeError):
                items.append(RuleValidationItem(
                    rule_name=f"numeric_type_{field_obj.field_name}",
                    field_name=field_obj.field_name,
                    passed=False,
                    severity="error",
                    message=f"Expected numeric value for {field_obj.field_name}, got '{val}'",
                    expected="Numeric",
                    actual=str(val),
                ))

        # Regex Pattern Check
        if "pattern" in rule_spec:
            pattern_str = rule_spec["pattern"]
            str_val = str(val).strip()
            if not re.match(pattern_str, str_val):
                items.append(RuleValidationItem(
                    rule_name=f"format_pattern_{field_obj.field_name}",
                    field_name=field_obj.field_name,
                    passed=False,
                    severity="error",
                    message=rule_spec.get("message", f"Format mismatch for {field_obj.field_name}"),
                    expected=pattern_str,
                    actual=str_val,
                ))

        # String Length Check
        if "min_length" in rule_spec or "max_length" in rule_spec:
            str_len = len(str(val).strip())
            min_len = rule_spec.get("min_length", 0)
            max_len = rule_spec.get("max_length", 10000)

            if str_len < min_len or str_len > max_len:
                items.append(RuleValidationItem(
                    rule_name=f"length_{field_obj.field_name}",
                    field_name=field_obj.field_name,
                    passed=False,
                    severity="warning",
                    message=rule_spec.get("message", f"String length {str_len} out of bounds [{min_len}, {max_len}]"),
                    expected=f"Length between {min_len} and {max_len}",
                    actual=str(str_len),
                ))

        return items
