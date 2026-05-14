"""
FHIR CRUD and clinical helper tools for the single REACT agent.

Wraps the unified fhir_utils module to provide REACT-compatible tool interface
for the five benchmark FHIR resource types:
- Patient
- Observation
- MedicationRequest
- Condition
- Procedure
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Import unified FHIR utilities
from .fhir_utils import (
    SUPPORTED_FHIR_RESOURCES,
    build_condition_resource,
    build_medication_request_resource,
    build_observation_resource,
    build_procedure_resource,
    create_fhir_resource,
    delete_fhir_resource,
    extract_answer_from_resource,
    get_supported_fhir_resources,
    load_fhir_resource_examples,
    query_lab_value,
    query_lab_values_range,
    query_patient_by_mrn,
    query_patient_by_name_dob,
    query_resources_by_patient,
    read_fhir_resource,
    request_fhir_url,
    search_fhir_resources,
    set_fhir_server_url,
    summarize_fhir_resources,
    update_fhir_resource,
)

def read_text_file(file_path_name: str) -> Dict[str, Any]:
    """
    Read a local text file used by the benchmark helper-tool tasks.

    Args:
        file_path_name: Absolute or workspace-relative file path.
    """
    path = Path(file_path_name)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / file_path_name
        if not path.exists():
            path = (
                Path(__file__).resolve().parents[1]
                / "bench_data"
                / "agentic_data_tool"
                / file_path_name
            )
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {"success": True, "file_path_name": str(path), "content": f.read()}
    except Exception as e:
        return {"success": False, "file_path_name": str(path), "error": str(e)}


# ============================================================================
# Clinical Assessment and Utility Functions (not in fhir_utils)
# ============================================================================


def assess_lab_value(value: float, lab_code: str) -> Dict[str, Any]:
    """Assess a lab value against simple built-in reference ranges."""
    lab_code = lab_code.upper()
    references = {
        "MG": {"normal_low": 1.7, "normal_high": 2.2, "units": "mg/dL"},
        "K": {"normal_low": 3.5, "normal_high": 5.0, "units": "mEq/L"},
        "GLU": {"normal_low": 70, "normal_high": 100, "units": "mg/dL"},
        "A1C": {"normal_high": 5.7, "units": "%"},
    }
    if lab_code not in references:
        return {"success": False, "error": f"Lab reference not found for {lab_code}"}

    ref = references[lab_code]
    status = "normal"
    severity = "low"
    recommendation = "No action needed"
    numeric_value = float(value)

    if lab_code == "MG":
        if numeric_value < 1.0:
            status, severity = "severe_deficiency", "critical"
            recommendation = "Order IV magnesium 4g over 4 hours if clinically appropriate"
        elif numeric_value < 1.5:
            status, severity = "moderate_deficiency", "high"
            recommendation = "Order IV magnesium 2g over 2 hours if clinically appropriate"
        elif numeric_value < ref["normal_low"]:
            status, severity = "mild_deficiency", "moderate"
            recommendation = "Consider IV magnesium 1g over 1 hour"
        elif numeric_value > ref["normal_high"]:
            status, severity = "high", "moderate"
            recommendation = "Monitor and evaluate clinical context"
    elif lab_code == "K":
        if numeric_value < 2.5:
            status, severity = "critical_low", "critical"
            recommendation = "Urgent potassium replacement and ECG monitoring"
        elif numeric_value < ref["normal_low"]:
            status, severity = "low", "high"
            recommendation = "Order potassium replacement if clinically appropriate"
        elif numeric_value > 6.5:
            status, severity = "critical_high", "critical"
            recommendation = "Urgent hyperkalemia management"
        elif numeric_value > ref["normal_high"]:
            status, severity = "high", "moderate"
            recommendation = "Monitor and evaluate clinical context"
    elif lab_code == "GLU":
        if numeric_value < ref["normal_low"]:
            status, severity = "hypoglycemia", "high"
            recommendation = "Treat hypoglycemia per protocol"
        elif numeric_value > ref["normal_high"]:
            status, severity = "elevated", "moderate"
            recommendation = "Monitor glucose trend"
    elif lab_code == "A1C":
        if numeric_value >= 6.5:
            status, severity = "diabetic_range", "moderate"
            recommendation = "Review diabetes management"
        elif numeric_value >= ref["normal_high"]:
            status, severity = "prediabetic_range", "moderate"
            recommendation = "Follow up for glycemic risk"

    return {
        "success": True,
        "lab_code": lab_code,
        "value": numeric_value,
        "units": ref.get("units", ""),
        "status": status,
        "severity": severity,
        "reference_range": {
            "low": ref.get("normal_low"),
            "high": ref.get("normal_high"),
        },
        "recommendation": recommendation,
    }


def calculate_average(values: List[float]) -> Dict[str, Any]:
    """Calculate average and simple statistics."""
    if not values:
        return {"success": False, "error": "No values provided"}
    clean = [float(v) for v in values]
    return {
        "success": True,
        "count": len(clean),
        "average": sum(clean) / len(clean),
        "minimum": min(clean),
        "maximum": max(clean),
    }


def assess_medication_need(
    lab_value: float,
    lab_code: str,
    patient_weight: float = 70,
) -> Dict[str, Any]:
    """Assess replacement-medication need for common electrolyte labs."""
    assessment = assess_lab_value(lab_value, lab_code)
    if not assessment.get("success"):
        return assessment
    lab_code = lab_code.upper()
    result = {
        **assessment,
        "patient_weight": patient_weight,
        "needs_medication": assessment["severity"] in ("moderate", "high", "critical"),
        "action_required": assessment["severity"] in ("high", "critical"),
    }
    if lab_code == "MG":
        result["medication_info"] = {
            "medication": "IV Magnesium", "ndc": "0338-1715-40"}
        if assessment["status"] == "severe_deficiency":
            result.update({"recommended_dose": "4g IV over 4 hours", "dose_value": 4,
                          "dose_unit": "g", "rate_value": 4, "rate_unit": "h"})
        elif assessment["status"] == "moderate_deficiency":
            result.update({"recommended_dose": "2g IV over 2 hours", "dose_value": 2,
                          "dose_unit": "g", "rate_value": 2, "rate_unit": "h"})
        elif assessment["status"] == "mild_deficiency":
            result.update({"recommended_dose": "1g IV over 1 hour", "dose_value": 1,
                          "dose_unit": "g", "rate_value": 1, "rate_unit": "h"})
    elif lab_code == "K" and float(lab_value) < 3.5:
        deficit = max(0.0, 3.5 - float(lab_value))
        meq_needed = round(deficit / 0.1 * 10)
        result["medication_info"] = {
            "medication": "Potassium replacement", "ndc": "40032-917-01"}
        result["recommended_dose"] = f"{meq_needed} mEq oral potassium"
        result["estimated_mEq_needed"] = meq_needed
    return result


# ============================================================================
# Authentication Wrapper Functions
# ============================================================================

# Lazy import to avoid circular dependency
_auth_client = None


def _get_auth_client():
    """Get or create the auth client instance."""
    global _auth_client
    if _auth_client is None:
        try:
            from .auth_client import get_react_auth_client
            _auth_client = get_react_auth_client()
        except ImportError:
            from auth_client import get_react_auth_client
            _auth_client = get_react_auth_client()
    return _auth_client


def auth_login(username: str, password: str) -> Dict[str, Any]:
    """
    Login to masb_auth service with username/password to get session token.

    Args:
        username: User's username from predefined_users.json
        password: User's password

    Returns:
        Dictionary with access_token, token_type, expires_in, role
    """
    try:
        client = _get_auth_client()
        return client.login(username, password)
    except Exception as e:
        return {"success": False, "error": str(e)}


def auth_get_user_info() -> Dict[str, Any]:
    """
    Get current authenticated user information including role.

    Returns:
        Dictionary with username, role, etc.
    """
    try:
        client = _get_auth_client()
        user_info = client.get_current_user_info()
        if user_info:
            return {"success": True, "user": user_info}
        return {"success": False, "error": "Not authenticated"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def auth_check_role(required_role: str) -> Dict[str, Any]:
    """
    Check if current user has required role (patient, physician, administrator).

    Args:
        required_role: Required role name

    Returns:
        Dictionary with authorized boolean
    """
    try:
        client = _get_auth_client()
        authorized = client.verify_role(required_role)
        return {"success": True, "authorized": authorized, "required_role": required_role}
    except Exception as e:
        return {"success": False, "error": str(e)}


def auth_logout() -> Dict[str, Any]:
    """
    Logout and clear authentication session.

    Returns:
        Dictionary with logout status
    """
    try:
        client = _get_auth_client()
        client.logout()
        return {"success": True, "message": "Logged out successfully"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Helper functions for convenience wrappers
def _format_name(name_dict: Dict[str, Any]) -> str:
    """Format patient name."""
    given = " ".join(name_dict.get("given", []))
    family = name_dict.get("family", "")
    return f"{given} {family}".strip()


def _calculate_age(dob: str, reference_date: Optional[str] = None) -> Optional[int]:
    """Calculate age from date of birth."""
    try:
        birth_date = datetime.fromisoformat(dob.replace("Z", "+00:00"))
        ref_date = _parse_reference_time(reference_date)
        return (ref_date.date() - birth_date.date()).days // 365
    except Exception:
        return None


def _parse_reference_time(reference_time: Optional[str]) -> datetime:
    """Parse reference time or return current time."""
    if not reference_time:
        return datetime.now()
    parsed = reference_time.replace("Z", "+00:00")
    return datetime.fromisoformat(parsed)


# ============================================================================
# Convenience FHIR Wrappers with Standardized Names
# ============================================================================


def create_observation(
    patient_mrn: str,
    lab_code: str,
    value: Any,
    unit: str,
    effective_datetime: Optional[str] = None,
    category_code: str = "laboratory",
    category_display: str = "Laboratory",
) -> Dict[str, Any]:
    """Create an Observation matching benchmark structure."""
    observation = build_observation_resource(
        patient_mrn=patient_mrn,
        lab_code=lab_code,
        value=value,
        unit=unit,
        effective_datetime=effective_datetime,
        category_code=category_code,
        category_display=category_display,
    )
    return create_fhir_resource("Observation", observation)


def create_medication_request(
    patient_mrn: str,
    medication_ndc: Optional[str] = None,
    dosage: str = "",
    reason: str = "",
    medication_text: Optional[str] = None,
    route: Optional[str] = None,
    dose_value: Optional[float] = None,
    dose_unit: Optional[str] = None,
    rate_value: Optional[float] = None,
    rate_unit: Optional[str] = None,
    authored_on: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a MedicationRequest matching benchmark structure."""
    med_request = build_medication_request_resource(
        patient_mrn=patient_mrn,
        medication_ndc=medication_ndc,
        dosage=dosage,
        reason=reason,
        medication_text=medication_text,
        route=route,
        dose_value=dose_value,
        dose_unit=dose_unit,
        rate_value=rate_value,
        rate_unit=rate_unit,
        authored_on=authored_on,
    )
    return create_fhir_resource("MedicationRequest", med_request)


def create_condition(
    patient_mrn: str,
    code: str,
    display: Optional[str] = None,
    onset_datetime: Optional[str] = None,
    recorded_date: Optional[str] = None,
    coding_system: str = "http://hl7.org/fhir/sid/icd-10",
) -> Dict[str, Any]:
    """Create a Condition matching benchmark structure."""
    condition = build_condition_resource(
        patient_mrn=patient_mrn,
        code=code,
        display=display,
        onset_datetime=onset_datetime,
        recorded_date=recorded_date,
        coding_system=coding_system,
    )
    return create_fhir_resource("Condition", condition)


def create_procedure(
    patient_mrn: str,
    code: str,
    display: Optional[str] = None,
    performed_datetime: Optional[str] = None,
    coding_system: str = "http://www.ama-assn.org/go/cpt",
) -> Dict[str, Any]:
    """Create a Procedure matching benchmark structure."""
    procedure = build_procedure_resource(
        patient_mrn=patient_mrn,
        code=code,
        display=display,
        performed_datetime=performed_datetime,
        coding_system=coding_system,
    )
    return create_fhir_resource("Procedure", procedure)


AVAILABLE_TOOLS: Dict[str, Dict[str, Any]] = {
    "get_supported_fhir_resources": {
        "func": get_supported_fhir_resources,
        "description": "Return the five benchmark FHIR resource types and example JSON shapes.",
        "parameters": {},
    },
    "read_text_file": {
        "func": read_text_file,
        "description": "Read a benchmark local text file such as medical_reference.txt.",
        "parameters": {"file_path_name": "str - Absolute or workspace-relative path."},
    },
    "read_fhir_resource": {
        "func": read_fhir_resource,
        "description": "Read one Patient, Observation, MedicationRequest, Condition, or Procedure by id.",
        "parameters": {"resource_type": "str", "resource_id": "str"},
    },
    "request_fhir_url": {
        "func": request_fhir_url,
        "description": "Execute a direct FHIR URL request for one of the five benchmark resource types.",
        "parameters": {"url": "str", "method": "GET/POST/PUT/DELETE optional", "payload": "dict optional"},
    },
    "search_fhir_resources": {
        "func": search_fhir_resources,
        "description": "Search Patient, Observation, MedicationRequest, Condition, or Procedure using FHIR query params.",
        "parameters": {"resource_type": "str", "search_params": "dict", "limit": "int optional"},
    },
    "create_fhir_resource": {
        "func": create_fhir_resource,
        "description": "Create a generic FHIR resource with a payload matching one of the five benchmark structures.",
        "parameters": {"resource_type": "str", "resource": "dict"},
    },
    "update_fhir_resource": {
        "func": update_fhir_resource,
        "description": "Update an existing FHIR resource by PUT with a full replacement payload.",
        "parameters": {"resource_type": "str", "resource_id": "str", "resource": "dict"},
    },
    "delete_fhir_resource": {
        "func": delete_fhir_resource,
        "description": "Delete one FHIR resource by type and id.",
        "parameters": {"resource_type": "str", "resource_id": "str"},
    },
    "query_patient_by_mrn": {
        "func": query_patient_by_mrn,
        "description": "Read Patient by MRN/id and return demographics.",
        "parameters": {"mrn": "str"},
    },
    "query_patient_by_name_dob": {
        "func": query_patient_by_name_dob,
        "description": "Search Patient by given name, family name, and optional birth date.",
        "parameters": {"given_name": "str", "family_name": "str", "dob": "str optional"},
    },
    "query_resources_by_patient": {
        "func": query_resources_by_patient,
        "description": "Search a clinical resource for a patient, optionally by code/date/status.",
        "parameters": {"resource_type": "str", "patient_mrn": "str", "code": "str optional", "date": "str optional", "status": "str optional"},
    },
    "query_lab_value": {
        "func": query_lab_value,
        "description": "Get most recent Observation value for a patient/lab code.",
        "parameters": {"patient_mrn": "str", "lab_code": "str", "hours_back": "int", "reference_time": "str optional"},
    },
    "query_lab_values_range": {
        "func": query_lab_values_range,
        "description": "Get multiple Observation values for a patient/lab code in a time window.",
        "parameters": {"patient_mrn": "str", "lab_code": "str", "hours_back": "int", "reference_time": "str optional"},
    },
    "create_observation": {
        "func": create_observation,
        "description": "Create an Observation using the benchmark Observation structure.",
        "parameters": {"patient_mrn": "str", "lab_code": "str", "value": "number/string", "unit": "str", "effective_datetime": "str optional"},
    },
    "create_medication_request": {
        "func": create_medication_request,
        "description": "Create a MedicationRequest using the benchmark MedicationRequest structure.",
        "parameters": {"patient_mrn": "str", "medication_ndc": "str optional", "dosage": "str", "route": "str optional", "dose_value": "float optional", "dose_unit": "str optional", "rate_value": "float optional", "rate_unit": "str optional"},
    },
    "create_condition": {
        "func": create_condition,
        "description": "Create a Condition using the benchmark Condition structure.",
        "parameters": {"patient_mrn": "str", "code": "str", "display": "str optional", "onset_datetime": "str optional"},
    },
    "create_procedure": {
        "func": create_procedure,
        "description": "Create a Procedure using the benchmark Procedure structure.",
        "parameters": {"patient_mrn": "str", "code": "str", "display": "str optional", "performed_datetime": "str optional"},
    },
    "extract_answer_from_resource": {
        "func": extract_answer_from_resource,
        "description": "Extract a value from a FHIR resource using a dotted path.",
        "parameters": {"resource": "dict", "field_path": "str", "default": "any optional"},
    },
    "summarize_fhir_resources": {
        "func": summarize_fhir_resources,
        "description": "Return compact summaries for a list of FHIR resources.",
        "parameters": {"resources": "list[dict]"},
    },
    "assess_lab_value": {
        "func": assess_lab_value,
        "description": "Assess a lab value against simple reference ranges.",
        "parameters": {"value": "float", "lab_code": "str"},
    },
    "calculate_average": {
        "func": calculate_average,
        "description": "Calculate average and simple statistics for numeric values.",
        "parameters": {"values": "list[float]"},
    },
    "assess_medication_need": {
        "func": assess_medication_need,
        "description": "Assess replacement medication need and dosing for MG/K values.",
        "parameters": {"lab_value": "float", "lab_code": "str", "patient_weight": "float optional"},
    },
    # Authentication tools (require masb_auth service running)
    "auth_login": {
        "func": auth_login,
        "description": "Login to masb_auth service with username/password to get session token.",
        "parameters": {"username": "str", "password": "str"},
    },
    "auth_get_user_info": {
        "func": auth_get_user_info,
        "description": "Get current authenticated user information including role.",
        "parameters": {},
    },
    "auth_check_role": {
        "func": auth_check_role,
        "description": "Check if current user has required role (patient, physician, administrator).",
        "parameters": {"required_role": "str"},
    },
    "auth_logout": {
        "func": auth_logout,
        "description": "Logout and clear authentication session.",
        "parameters": {},
    },
}


def get_tool_descriptions() -> str:
    """Format tool descriptions for the REACT prompt."""
    lines = ["Available tools:"]
    for tool_name, tool_info in AVAILABLE_TOOLS.items():
        lines.append(f"\n{tool_name}: {tool_info['description']}")
        params = tool_info.get("parameters", {})
        if params:
            lines.append("  Parameters:")
            for param_name, param_desc in params.items():
                lines.append(f"    - {param_name}: {param_desc}")
    return "\n".join(lines)


def get_tool_functions() -> Dict[str, Callable[..., Dict[str, Any]]]:
    """Return executable tool mapping."""
    return {name: spec["func"] for name, spec in AVAILABLE_TOOLS.items()}


# Backwards-compatible aliases used by earlier local experiments.
get_fhir_resource = read_fhir_resource
create_resource = create_fhir_resource
update_resource = update_fhir_resource
delete_resource = delete_fhir_resource
