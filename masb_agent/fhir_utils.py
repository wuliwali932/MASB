"""
Unified FHIR CRUD Utilities for Medical Agents.

This module provides a shared, centralized FHIR resource management layer
compatible with both REACT (masb_agent) and Orchestrator (masb_orchestra) agents.

The five benchmark FHIR resource types are:
- Patient
- Observation
- MedicationRequest
- Condition
- Procedure

All tools return standardized response dictionaries with 'success' key.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlparse

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)

# Configuration
FHIR_SERVER_URL = os.getenv("FHIR_SERVER_URL", "http://localhost:8080/fhir")
REQUEST_TIMEOUT = int(os.getenv("FHIR_REQUEST_TIMEOUT", "5"))

# Supported FHIR resource types across both agents
SUPPORTED_FHIR_RESOURCES = (
    "Patient",
    "Observation",
    "MedicationRequest",
    "Condition",
    "Procedure",
)

# Path to benchmark FHIR examples
FHIR_EXAMPLES_DIR = (
    Path(__file__).resolve().parents[1] /
    "bench_data" / "FHIR_resource_examples"
)


def set_fhir_server_url(url: str) -> None:
    """Set the base FHIR server URL used by all CRUD operations."""
    global FHIR_SERVER_URL
    FHIR_SERVER_URL = url.rstrip("/")


def get_fhir_server_url() -> str:
    """Get the current FHIR server URL."""
    return FHIR_SERVER_URL


def _headers() -> Dict[str, str]:
    """HTTP headers for FHIR requests."""
    headers = {
        "Accept": "application/fhir+json",
        "Content-Type": "application/fhir+json",
    }
    try:
        from .auth_client import get_react_auth_client
        token = get_react_auth_client().get_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
    except Exception:
        pass
    return headers


def _require_requests() -> Any:
    """Ensure requests library is available."""
    if requests is None:
        raise RuntimeError(
            "The 'requests' package is required for live FHIR HTTP calls.")
    return requests


def _auth_proxy_endpoint(method: str, url: str) -> Optional[tuple[str, Dict[str, str]]]:
    """Return masb_auth proxy endpoint if the REACT auth client has a session."""
    try:
        from .auth_client import get_react_auth_client
        auth_client = get_react_auth_client()
        token = auth_client.get_token()
        role = auth_client.get_role()
        if not token or not role:
            return None
        role_path = "admin" if role == "administrator" else role
        operation = {
            "GET": "read",
            "POST": "create",
            "PUT": "update",
            "DELETE": "delete",
        }.get(method.upper())
        if not operation:
            return None
        proxy_url = (
            f"{auth_client.auth_service_url}/{role_path}/{operation}"
            f"?fhir_api_request={quote(url, safe='')}"
        )
        return proxy_url, {"Authorization": f"Bearer {token}"}
    except Exception:
        return None


def _request_fhir(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    """Request FHIR directly, or via masb_auth when authenticated."""
    http = _require_requests()
    method = method.upper()
    request_url = url
    if params:
        request_url = http.Request("GET", url, params=params).prepare().url

    proxy = _auth_proxy_endpoint(method, request_url)
    if proxy:
        proxy_url, auth_headers = proxy
        if method == "GET":
            return http.get(proxy_url, headers=auth_headers, timeout=REQUEST_TIMEOUT)
        if method == "POST":
            return http.post(proxy_url, json=payload or {}, headers=auth_headers, timeout=REQUEST_TIMEOUT)
        if method == "PUT":
            return http.put(proxy_url, json=payload or {}, headers=auth_headers, timeout=REQUEST_TIMEOUT)
        if method == "DELETE":
            return http.delete(proxy_url, headers=auth_headers, timeout=REQUEST_TIMEOUT)

    if method == "GET":
        return http.get(url, params=params, headers=_headers(), timeout=REQUEST_TIMEOUT)
    if method == "POST":
        return http.post(url, json=payload or {}, headers=_headers(), timeout=REQUEST_TIMEOUT)
    if method == "PUT":
        return http.put(url, json=payload or {}, headers=_headers(), timeout=REQUEST_TIMEOUT)
    if method == "DELETE":
        return http.delete(url, headers=_headers(), timeout=REQUEST_TIMEOUT)
    raise ValueError("method must be one of GET, POST, PUT, DELETE")


def _validate_resource_type(resource_type: str) -> str:
    """Validate and normalize resource type."""
    if not resource_type:
        raise ValueError("resource_type is required")
    normalized = resource_type.strip()
    for supported in SUPPORTED_FHIR_RESOURCES:
        if supported.lower() == normalized.lower():
            return supported
    supported = ", ".join(SUPPORTED_FHIR_RESOURCES)
    raise ValueError(
        f"Unsupported FHIR resource type '{resource_type}'. Supported: {supported}")


def _resource_url(resource_type: str, resource_id: Optional[str] = None) -> str:
    """Construct FHIR resource URL."""
    resource_type = _validate_resource_type(resource_type)
    if resource_id:
        return f"{FHIR_SERVER_URL.rstrip('/')}/{resource_type}/{resource_id}"
    return f"{FHIR_SERVER_URL.rstrip('/')}/{resource_type}"


def _validate_fhir_url(url: str) -> str:
    """Validate that URL targets a FHIR endpoint with proper resource type."""
    parsed = urlparse(url)
    path_parts = [part for part in parsed.path.split("/") if part]
    if "fhir" not in path_parts:
        raise ValueError("URL must target a FHIR endpoint containing /fhir/")
    fhir_index = path_parts.index("fhir")
    if len(path_parts) <= fhir_index + 1:
        raise ValueError("URL must include a FHIR resource type after /fhir/")
    _validate_resource_type(path_parts[fhir_index + 1])
    return url


def _patient_subject(patient_mrn: str) -> Dict[str, Any]:
    """Build standard subject reference for patient-linked resources."""
    return {
        "reference": f"Patient/{patient_mrn}",
        "identifier": {
            "system": "http://terminology.hl7.org/CodeSystem/v2-0203",
            "value": patient_mrn,
        },
    }


def _coerce_json_object(value: Any) -> Dict[str, Any]:
    """Convert value to JSON dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Expected a JSON object/dict")


def _bundle_entries(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract resources from FHIR bundle."""
    return [entry.get("resource", entry) for entry in bundle.get("entry", [])]


def _extract_patient_id_from_subject(resource: Dict[str, Any]) -> Optional[str]:
    """Extract patient MRN from resource subject."""
    subject = resource.get("subject", {})
    identifier = subject.get("identifier", {})
    if identifier.get("value"):
        return identifier["value"]
    reference = subject.get("reference", "")
    if reference.startswith("Patient/"):
        return reference.split("/", 1)[1]
    return None


def _get_code(resource: Dict[str, Any]) -> Any:
    """Extract code from resource."""
    code_obj = resource.get("code") or resource.get(
        "medicationCodeableConcept") or {}
    coding = code_obj.get("coding", [])
    if coding:
        return coding[0].get("code") or coding[0].get("display")
    return code_obj.get("text")


def _get_event_time(resource: Dict[str, Any]) -> str:
    """Extract timestamp from resource."""
    return (
        resource.get("effectiveDateTime")
        or resource.get("issued")
        or resource.get("authoredOn")
        or resource.get("onsetDateTime")
        or resource.get("recordedDate")
        or resource.get("performedDateTime")
        or ""
    )


def _value_quantity(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Extract valueQuantity from resource."""
    value = resource.get("valueQuantity")
    if isinstance(value, dict):
        return value
    return {}


def _parse_reference_time(reference_time: Optional[str]) -> datetime:
    """Parse reference time or return current time."""
    if not reference_time:
        return datetime.now()
    parsed = reference_time.replace("Z", "+00:00")
    return datetime.fromisoformat(parsed)


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


# ============================================================================
# FHIR CRUD Operations
# ============================================================================


def read_fhir_resource(resource_type: str, resource_id: str) -> Dict[str, Any]:
    """Read one FHIR resource by type and id."""
    try:
        url = _resource_url(resource_type, resource_id)
        response = _request_fhir("GET", url)
        response.raise_for_status()
        return {
            "success": True,
            "operation": "read",
            "resource_type": _validate_resource_type(resource_type),
            "resource_id": resource_id,
            "resource": response.json(),
        }
    except Exception as e:
        logger.error("FHIR read failed: %s", e)
        return {
            "success": False,
            "operation": "read",
            "resource_type": resource_type,
            "resource_id": resource_id,
            "error": str(e),
        }


def search_fhir_resources(
    resource_type: str,
    search_params: Optional[Dict[str, Any]] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """Search FHIR resources with query parameters."""
    try:
        resource_type = _validate_resource_type(resource_type)
        params = dict(search_params or {})
        if limit and "_count" not in params:
            params["_count"] = limit
        response = _request_fhir(
            "GET",
            _resource_url(resource_type),
            params=params,
        )
        response.raise_for_status()
        bundle = response.json()
        resources = _bundle_entries(bundle)[:limit]
        return {
            "success": True,
            "operation": "search",
            "resource_type": resource_type,
            "search_params": params,
            "count": len(resources),
            "total": bundle.get("total", len(resources)),
            "resources": resources,
            "bundle": bundle,
        }
    except Exception as e:
        logger.error("FHIR search failed: %s", e)
        return {
            "success": False,
            "operation": "search",
            "resource_type": resource_type,
            "search_params": search_params or {},
            "error": str(e),
        }


def create_fhir_resource(resource_type: str, resource: Dict[str, Any]) -> Dict[str, Any]:
    """Create one FHIR resource."""
    try:
        resource_type = _validate_resource_type(resource_type)
        payload = _coerce_json_object(resource)
        payload["resourceType"] = payload.get("resourceType", resource_type)
        if payload["resourceType"] != resource_type:
            raise ValueError("resource.resourceType must match resource_type")
        response = _request_fhir(
            "POST",
            _resource_url(resource_type),
            payload=payload,
        )
        response.raise_for_status()
        data = response.json() if response.content else {}
        return {
            "success": True,
            "operation": "create",
            "created": True,
            "resource_type": resource_type,
            "resource_id": data.get("id", payload.get("id", "")),
            "resource": data or payload,
        }
    except Exception as e:
        logger.error("FHIR create failed: %s", e)
        return {
            "success": False,
            "operation": "create",
            "created": False,
            "resource_type": resource_type,
            "error": str(e),
        }


def update_fhir_resource(
    resource_type: str,
    resource_id: str,
    resource: Dict[str, Any],
) -> Dict[str, Any]:
    """Update one FHIR resource with PUT."""
    try:
        resource_type = _validate_resource_type(resource_type)
        payload = _coerce_json_object(resource)
        payload["resourceType"] = payload.get("resourceType", resource_type)
        payload["id"] = payload.get("id", resource_id)
        if payload["resourceType"] != resource_type:
            raise ValueError("resource.resourceType must match resource_type")
        if payload["id"] != resource_id:
            raise ValueError("resource.id must match resource_id")
        response = _request_fhir(
            "PUT",
            _resource_url(resource_type, resource_id),
            payload=payload,
        )
        response.raise_for_status()
        data = response.json() if response.content else {}
        return {
            "success": True,
            "operation": "update",
            "updated": True,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "resource": data or payload,
        }
    except Exception as e:
        logger.error("FHIR update failed: %s", e)
        return {
            "success": False,
            "operation": "update",
            "updated": False,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "error": str(e),
        }


def delete_fhir_resource(resource_type: str, resource_id: str) -> Dict[str, Any]:
    """Delete one FHIR resource by type and id."""
    try:
        resource_type = _validate_resource_type(resource_type)
        response = _request_fhir("DELETE", _resource_url(resource_type, resource_id))
        if response.status_code not in (200, 202, 204):
            response.raise_for_status()
        return {
            "success": True,
            "operation": "delete",
            "deleted": True,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "status_code": response.status_code,
        }
    except Exception as e:
        logger.error("FHIR delete failed: %s", e)
        return {
            "success": False,
            "operation": "delete",
            "deleted": False,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "error": str(e),
        }


def request_fhir_url(
    url: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a direct FHIR URL request for one of the five benchmark resources."""
    try:
        safe_url = _validate_fhir_url(url)
        method = method.upper()
        response = _request_fhir(method, safe_url, payload=payload)
        response.raise_for_status()
        data = response.json() if response.content else {}
        return {
            "success": True,
            "operation": method.lower(),
            "url": safe_url,
            "status_code": response.status_code,
            "response": data,
            "resources": _bundle_entries(data) if isinstance(data, dict) and "entry" in data else [],
        }
    except Exception as e:
        logger.error("FHIR URL request failed: %s", e)
        return {
            "success": False,
            "operation": method.lower(),
            "url": url,
            "error": str(e),
        }


# ============================================================================
# FHIR Resource Builders
# ============================================================================


def build_observation_resource(
    patient_mrn: str,
    lab_code: str,
    value: Any,
    unit: str,
    effective_datetime: Optional[str] = None,
    category_code: str = "laboratory",
    category_display: str = "Laboratory",
) -> Dict[str, Any]:
    """Build an Observation payload matching benchmark structure."""
    timestamp = effective_datetime or datetime.now().isoformat()
    return {
        "resourceType": "Observation",
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": category_code,
                        "display": category_display,
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": lab_code,
                    "display": lab_code,
                }
            ],
            "text": lab_code,
        },
        "subject": _patient_subject(patient_mrn),
        "effectiveDateTime": timestamp,
        "issued": timestamp,
        "valueQuantity": {
            "value": value,
            "unit": unit,
            "system": "http://unitsofmeasure.org",
            "code": unit,
        },
    }


def build_medication_request_resource(
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
    """Build a MedicationRequest payload matching benchmark structure."""
    medication: Dict[str, Any] = {}
    if medication_ndc:
        medication["coding"] = [
            {
                "system": "http://hl7.org/fhir/sid/ndc",
                "code": medication_ndc,
                "display": medication_text or medication_ndc,
            }
        ]
    medication["text"] = medication_text or medication_ndc or "Medication order"

    dosage_instruction: Dict[str, Any] = {}
    if dosage:
        dosage_instruction["text"] = dosage
    if route:
        dosage_instruction["route"] = {"text": route}
    dose_and_rate: Dict[str, Any] = {}
    if dose_value is not None and dose_unit:
        dose_and_rate["doseQuantity"] = {
            "value": dose_value, "unit": dose_unit}
    if rate_value is not None and rate_unit:
        dose_and_rate["rateQuantity"] = {
            "value": rate_value, "unit": rate_unit}
    if dose_and_rate:
        dosage_instruction["doseAndRate"] = [dose_and_rate]

    med_request: Dict[str, Any] = {
        "resourceType": "MedicationRequest",
        "status": "active",
        "intent": "order",
        "medicationCodeableConcept": medication,
        "subject": _patient_subject(patient_mrn),
        "authoredOn": authored_on or datetime.now().isoformat(),
        "dosageInstruction": [dosage_instruction or {"text": dosage}],
    }
    if reason:
        med_request["reasonCode"] = [{"text": reason}]
    return med_request


def build_condition_resource(
    patient_mrn: str,
    code: str,
    display: Optional[str] = None,
    onset_datetime: Optional[str] = None,
    recorded_date: Optional[str] = None,
    coding_system: str = "http://hl7.org/fhir/sid/icd-10",
) -> Dict[str, Any]:
    """Build a Condition payload matching benchmark structure."""
    timestamp = onset_datetime or recorded_date or datetime.now().isoformat()
    return {
        "resourceType": "Condition",
        "code": {
            "coding": [
                {
                    "system": coding_system,
                    "code": code,
                    "display": display or code,
                }
            ]
        },
        "subject": _patient_subject(patient_mrn),
        "onsetDateTime": timestamp,
        "recordedDate": recorded_date or timestamp,
    }


def build_procedure_resource(
    patient_mrn: str,
    code: str,
    display: Optional[str] = None,
    performed_datetime: Optional[str] = None,
    coding_system: str = "http://www.ama-assn.org/go/cpt",
) -> Dict[str, Any]:
    """Build a Procedure payload matching benchmark structure."""
    return {
        "resourceType": "Procedure",
        "code": {
            "coding": [
                {
                    "system": coding_system,
                    "code": code,
                    "display": display or code,
                }
            ],
            "text": display or code,
        },
        "subject": _patient_subject(patient_mrn),
        "performedDateTime": performed_datetime or datetime.now().isoformat(),
    }


# ============================================================================
# Convenience Query Functions
# ============================================================================


def query_patient_by_mrn(mrn: str) -> Dict[str, Any]:
    """Read Patient by MRN/id and return a compact patient summary."""
    result = read_fhir_resource("Patient", mrn)
    if not result.get("success"):
        return {**result, "found": False, "mrn": mrn}
    patient = result["resource"]
    dob = patient.get("birthDate", "")
    return {
        "success": True,
        "found": True,
        "mrn": patient.get("id", mrn),
        "name": _format_name(patient.get("name", [{}])[0]),
        "dob": dob,
        "age": _calculate_age(dob) if dob else None,
        "gender": patient.get("gender", ""),
        "telecom": patient.get("telecom", []),
        "resource": patient,
    }


def query_patient_by_name_dob(
    given_name: str,
    family_name: str,
    dob: Optional[str] = None,
) -> Dict[str, Any]:
    """Search Patient by given name, family name, and optional birth date."""
    params = {"given": given_name, "family": family_name}
    if dob:
        params["birthdate"] = dob
    result = search_fhir_resources("Patient", params, limit=10)
    if not result.get("success"):
        return {**result, "found": False}
    resources = result.get("resources", [])
    if not resources:
        return {"success": True, "found": False, "mrn": "Patient not found", "resources": []}
    patient = resources[0]
    return {
        "success": True,
        "found": True,
        "mrn": patient.get("id", ""),
        "name": _format_name(patient.get("name", [{}])[0]),
        "dob": patient.get("birthDate", dob or ""),
        "gender": patient.get("gender", ""),
        "contact": patient.get("telecom", []),
        "resources": resources,
    }


def query_resources_by_patient(
    resource_type: str,
    patient_mrn: str,
    code: Optional[str] = None,
    date: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """Search one of the clinical resources for a patient."""
    params: Dict[str, Any] = {"subject": patient_mrn}
    if code:
        params["code"] = code
    if date:
        params["date"] = date
    if status:
        params["status"] = status
    result = search_fhir_resources(resource_type, params, limit=limit)
    if result.get("success") and result.get("count", 0) > 0:
        return result

    patient_params = {**params}
    patient_params["patient"] = patient_params.pop("subject")
    fallback = search_fhir_resources(
        resource_type, patient_params, limit=limit)
    if fallback.get("success") and fallback.get("count", 0) > 0:
        fallback["search_params_used"] = "patient"
        return fallback

    result["patient_param_fallback"] = fallback
    return result


def query_lab_value(
    patient_mrn: str,
    lab_code: str,
    hours_back: int = 24,
    reference_time: Optional[str] = None,
) -> Dict[str, Any]:
    """Query the most recent Observation value for a patient/lab code."""
    params = {"subject": patient_mrn, "code": lab_code}
    if hours_back:
        cutoff = _parse_reference_time(
            reference_time) - timedelta(hours=hours_back)
        params["date"] = f"ge{cutoff.isoformat()}"
    result = search_fhir_resources("Observation", params, limit=5000)
    if not result.get("success"):
        return {**result, "found": False, "value": -1}
    observations = sorted(
        result.get("resources", []),
        key=lambda resource: resource.get("effectiveDateTime", ""),
        reverse=True,
    )
    if not observations:
        return {
            "success": True,
            "found": False,
            "value": -1,
            "lab_code": lab_code,
            "patient_mrn": patient_mrn,
        }
    obs = observations[0]
    value = _value_quantity(obs)
    return {
        "success": True,
        "found": True,
        "value": value.get("value", -1),
        "unit": value.get("unit", ""),
        "timestamp": obs.get("effectiveDateTime", ""),
        "lab_code": lab_code,
        "patient_mrn": patient_mrn,
        "resource": obs,
    }


def query_lab_values_range(
    patient_mrn: str,
    lab_code: str,
    hours_back: int = 24,
    reference_time: Optional[str] = None,
) -> Dict[str, Any]:
    """Query multiple Observation values for a patient/lab code in a time window."""
    params = {"subject": patient_mrn, "code": lab_code}
    if hours_back:
        cutoff = _parse_reference_time(
            reference_time) - timedelta(hours=hours_back)
        params["date"] = f"ge{cutoff.isoformat()}"
    result = search_fhir_resources("Observation", params, limit=5000)
    if not result.get("success"):
        return {**result, "found": False, "values": []}
    values = []
    for obs in result.get("resources", []):
        quantity = _value_quantity(obs)
        if "value" not in quantity:
            continue
        values.append(
            {
                "value": float(quantity["value"]),
                "unit": quantity.get("unit", ""),
                "timestamp": obs.get("effectiveDateTime", ""),
                "resource_id": obs.get("id", ""),
            }
        )
    return {
        "success": True,
        "found": bool(values),
        "count": len(values),
        "values": values,
        "lab_code": lab_code,
        "patient_mrn": patient_mrn,
    }


# ============================================================================
# Resource Summary and Analysis
# ============================================================================


def summarize_fhir_resources(resources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return compact summaries for searched FHIR resources."""
    summaries = []
    for resource in resources:
        summaries.append(
            {
                "resourceType": resource.get("resourceType"),
                "id": resource.get("id"),
                "patient_mrn": _extract_patient_id_from_subject(resource),
                "code": _get_code(resource),
                "time": _get_event_time(resource),
                "valueQuantity": resource.get("valueQuantity"),
                "status": resource.get("status"),
            }
        )
    return {"success": True, "count": len(summaries), "summaries": summaries}


def extract_answer_from_resource(
    resource: Dict[str, Any],
    field_path: str,
    default: Any = -1,
) -> Dict[str, Any]:
    """Extract a value from a FHIR resource using a dotted path."""
    try:
        current: Any = _coerce_json_object(resource)
        for part in field_path.split("."):
            if isinstance(current, list):
                current = current[int(part)]
            elif isinstance(current, dict):
                current = current[part]
            else:
                return {"success": True, "value": default, "found": False}
        return {"success": True, "value": current, "found": True}
    except Exception:
        return {"success": True, "value": default, "found": False}


def load_fhir_resource_examples() -> Dict[str, Dict[str, Any]]:
    """Load the five benchmark FHIR resource examples from bench_data."""
    examples: Dict[str, Dict[str, Any]] = {}
    for resource_type in SUPPORTED_FHIR_RESOURCES:
        example_path = FHIR_EXAMPLES_DIR / f"{resource_type}.json"
        try:
            with open(example_path, "r", encoding="utf-8") as f:
                examples[resource_type] = json.load(f)
        except FileNotFoundError:
            logger.warning("Missing FHIR example file: %s", example_path)
    return examples


def get_supported_fhir_resources() -> Dict[str, Any]:
    """Return supported resource types and their example shapes."""
    return {
        "success": True,
        "supported_resources": list(SUPPORTED_FHIR_RESOURCES),
        "examples": load_fhir_resource_examples(),
    }
