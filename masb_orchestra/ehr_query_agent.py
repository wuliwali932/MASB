"""
EHR Query Agent - CRUD Operations on Medical Data
Handles reading and writing operations to FHIR database.
Supports querying patient data, lab values, medications, etc.
Integrates with SMART on FHIR authentication from masb_auth.
Uses shared FHIR utilities from masb_agent.fhir_utils for consistency.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from langchain_core.language_models import BaseLanguageModel
from enum import Enum

try:
    import requests
except ImportError:
    requests = None

# Import shared FHIR utilities
try:
    from masb_agent.fhir_utils import (
        SUPPORTED_FHIR_RESOURCES,
        build_condition_resource,
        build_medication_request_resource,
        build_observation_resource,
        build_procedure_resource,
        create_fhir_resource,
        delete_fhir_resource,
        read_fhir_resource,
        search_fhir_resources,
        update_fhir_resource,
        set_fhir_server_url,
    )
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from masb_agent.fhir_utils import (
        SUPPORTED_FHIR_RESOURCES,
        build_condition_resource,
        build_medication_request_resource,
        build_observation_resource,
        build_procedure_resource,
        create_fhir_resource,
        delete_fhir_resource,
        read_fhir_resource,
        search_fhir_resources,
        update_fhir_resource,
        set_fhir_server_url,
    )

logger = logging.getLogger(__name__)


class FHIRResourceType(Enum):
    """FHIR resource types shared by REACT and orchestra agents."""
    PATIENT = "Patient"
    OBSERVATION = "Observation"
    MEDICATION_REQUEST = "MedicationRequest"
    CONDITION = "Condition"
    PROCEDURE = "Procedure"


@dataclass
class FHIRQuery:
    """FHIR query parameters."""
    resource_type: FHIRResourceType
    patient_id: Optional[str] = None
    code: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    status: Optional[str] = None
    limit: int = 100


class EHRQueryAgent:
    """
    EHR Query Agent for FHIR database operations.
    - Create Read Update Delete operations
    - Query patient demographics
    - Query lab values and observations
    - Query medications and procedures
    - Create and update medical records
    - SMART on FHIR authentication support

    Uses shared FHIR utilities from masb_agent.fhir_utils for consistency
    with REACT agent implementation.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        fhir_server_url: str = "http://localhost:8080/fhir",
        auth_manager: Optional[Any] = None
    ):
        """
        Initialize EHR Query Agent.

        Args:
            llm: Language model for query generation and interpretation
            fhir_server_url: Base URL of FHIR server
            auth_manager: SMART Auth Manager from auth_manager.py (optional)
        """
        self.llm = llm
        self.fhir_server_url = fhir_server_url
        self.auth_manager = auth_manager
        self.query_history: List[Dict[str, Any]] = []
        self.request_timeout = 5  # seconds

        # Set the FHIR server URL in the shared utilities
        set_fhir_server_url(fhir_server_url)

    def _get_headers(self) -> Dict[str, str]:
        """
        Get HTTP headers for FHIR requests.
        Includes authentication if auth_manager is provided.

        Returns:
            Dictionary of HTTP headers
        """
        headers = {
            "Content-Type": "application/fhir+json",
            "Accept": "application/fhir+json"
        }

        # Add authentication header if auth manager is available
        if self.auth_manager:
            try:
                auth_header = self.auth_manager.get_auth_header()
                headers.update(auth_header)
            except Exception as e:
                logger.warning(f"Failed to get auth header: {str(e)}")

        return headers

    def authenticate(self, username: str, password: str) -> bool:
        """
        Authenticate with SMART on FHIR service.

        Args:
            username: Username for authentication
            password: Password for authentication

        Returns:
            True if authentication successful, False otherwise
        """
        if not self.auth_manager:
            logger.warning("No auth_manager configured for authentication")
            return False

        try:
            result = self.auth_manager.login(username, password)
            user_info = self.auth_manager.get_current_user_info()
            logger.info(f"Authenticated user: {user_info}")
            return True
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False

    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """
        Get currently authenticated user information.

        Returns:
            User info dict or None if not authenticated
        """
        if not self.auth_manager:
            return None
        return self.auth_manager.get_current_user_info()

    def verify_role(self, required_role: str) -> bool:
        """
        Verify if current user has required role.

        Args:
            required_role: Required role name

        Returns:
            True if user has role, False otherwise
        """
        if not self.auth_manager:
            return False
        return self.auth_manager.verify_role(required_role)

    def _validate_resource_type(self, resource_type: str) -> str:
        """Validate a resource type against the shared five-resource contract."""
        for supported in SUPPORTED_FHIR_RESOURCES:
            if supported.lower() == resource_type.lower():
                return supported
        raise ValueError(
            f"Unsupported FHIR resource type '{resource_type}'. "
            f"Supported resources: {', '.join(SUPPORTED_FHIR_RESOURCES)}"
        )

    def read_resource(self, resource_type: str, resource_id: str) -> Dict[str, Any]:
        """Read a FHIR resource using shared utilities."""
        return read_fhir_resource(resource_type, resource_id)

    def search_resources(
        self,
        resource_type: str,
        search_params: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Search FHIR resources using shared utilities."""
        return search_fhir_resources(resource_type, search_params, limit)

    def create_resource(self, resource_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create a FHIR resource using shared utilities."""
        return create_fhir_resource(resource_type, payload)

    def update_resource(
        self, resource_type: str, resource_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update a FHIR resource using shared utilities."""
        return update_fhir_resource(resource_type, resource_id, payload)

    def delete_resource(self, resource_type: str, resource_id: str) -> Dict[str, Any]:
        """Delete a FHIR resource using shared utilities."""
        return delete_fhir_resource(resource_type, resource_id)

    def read_fhir_resource(self, resource_type: str, resource_id: str) -> Dict[str, Any]:
        """
        Read one Patient, Observation, MedicationRequest, Condition, or Procedure by id.
        """
        try:
            resource_type = self._validate_resource_type(resource_type)
            url = f"{self.fhir_server_url}/{resource_type}/{resource_id}"
            response = requests.get(
                url,
                headers=self._get_headers(),
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            result = {
                "success": True,
                "operation": "read",
                "resource_type": resource_type,
                "resource_id": resource_id,
                "resource": response.json(),
            }
            self._log_query("read_fhir_resource", result, result)
            return result
        except Exception as e:
            error_result = {
                "success": False,
                "operation": "read",
                "resource_type": resource_type,
                "resource_id": resource_id,
                "error": str(e),
            }
            self._log_query("read_fhir_resource", {
                            "resource_type": resource_type, "resource_id": resource_id}, error_result)
            return error_result

    def search_fhir_resources(
        self,
        resource_type: str,
        search_params: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Search one of the five supported benchmark FHIR resources.
        """
        try:
            resource_type = self._validate_resource_type(resource_type)
            params = dict(search_params or {})
            if limit and "_count" not in params:
                params["_count"] = limit
            url = f"{self.fhir_server_url}/{resource_type}"
            response = requests.get(
                url,
                params=params,
                headers=self._get_headers(),
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            bundle = response.json()
            resources = [
                entry.get("resource", entry)
                for entry in bundle.get("entry", [])
            ][:limit]
            result = {
                "success": True,
                "operation": "search",
                "resource_type": resource_type,
                "search_params": params,
                "count": len(resources),
                "total": bundle.get("total", len(resources)),
                "resources": resources,
                "bundle": bundle,
            }
            self._log_query("search_fhir_resources", params, result)
            return result
        except Exception as e:
            error_result = {
                "success": False,
                "operation": "search",
                "resource_type": resource_type,
                "search_params": search_params or {},
                "error": str(e),
            }
            self._log_query("search_fhir_resources",
                            search_params or {}, error_result)
            return error_result

    def create_fhir_resource(self, resource_type: str, resource: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a generic FHIR resource using the shared five-resource structure.
        """
        try:
            result = self._post_fhir_resource(resource_type, resource)
            self._log_query("create_fhir_resource", {
                            "resource_type": resource_type}, result)
            return result
        except Exception as e:
            error_result = {
                "success": False,
                "operation": "create",
                "created": False,
                "resource_type": resource_type,
                "error": str(e),
            }
            self._log_query("create_fhir_resource", {
                            "resource_type": resource_type}, error_result)
            return error_result

    def update_fhir_resource(
        self,
        resource_type: str,
        resource_id: str,
        resource: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update one FHIR resource by id with a full replacement payload.
        """
        try:
            resource_type = self._validate_resource_type(resource_type)
            payload = dict(resource)
            payload["resourceType"] = payload.get(
                "resourceType", resource_type)
            payload["id"] = payload.get("id", resource_id)
            if payload["resourceType"] != resource_type:
                raise ValueError(
                    "resource.resourceType must match resource_type")
            if payload["id"] != resource_id:
                raise ValueError("resource.id must match resource_id")
            url = f"{self.fhir_server_url}/{resource_type}/{resource_id}"
            response = requests.put(
                url,
                json=payload,
                headers=self._get_headers(),
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            data = response.json()
            result = {
                "success": True,
                "operation": "update",
                "updated": True,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "resource": data,
            }
            self._log_query("update_fhir_resource", {
                            "resource_type": resource_type, "resource_id": resource_id}, result)
            return result
        except Exception as e:
            error_result = {
                "success": False,
                "operation": "update",
                "updated": False,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "error": str(e),
            }
            self._log_query("update_fhir_resource", {
                            "resource_type": resource_type, "resource_id": resource_id}, error_result)
            return error_result

    def delete_fhir_resource(self, resource_type: str, resource_id: str) -> Dict[str, Any]:
        """
        Delete one FHIR resource by type and id.
        """
        try:
            resource_type = self._validate_resource_type(resource_type)
            url = f"{self.fhir_server_url}/{resource_type}/{resource_id}"
            response = requests.delete(
                url,
                headers=self._get_headers(),
                timeout=self.request_timeout,
            )
            if response.status_code not in (200, 202, 204):
                response.raise_for_status()
            result = {
                "success": True,
                "operation": "delete",
                "deleted": True,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "status_code": response.status_code,
            }
            self._log_query("delete_fhir_resource", {
                            "resource_type": resource_type, "resource_id": resource_id}, result)
            return result
        except Exception as e:
            error_result = {
                "success": False,
                "operation": "delete",
                "deleted": False,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "error": str(e),
            }
            self._log_query("delete_fhir_resource", {
                            "resource_type": resource_type, "resource_id": resource_id}, error_result)
            return error_result

    def query_patient_by_name_dob(self, given_name: str, family_name: str,
                                  dob: str) -> Dict[str, Any]:
        """
        Query patient by name and date of birth.

        Args:
            given_name: Patient's given name
            family_name: Patient's family name
            dob: Date of birth (YYYY-MM-DD)

        Returns:
            Patient information including MRN
        """
        try:
            url = f"{self.fhir_server_url}/Patient"
            params = {
                "given": given_name,
                "family": family_name,
                "birthdate": dob
            }

            headers = self._get_headers()
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=self.request_timeout
            )
            response.raise_for_status()

            data = response.json()
            if data.get("entry"):
                patient = data["entry"][0]["resource"]
                result = {
                    "found": True,
                    "mrn": patient.get("id", ""),
                    "name": {
                        "given": given_name,
                        "family": family_name
                    },
                    "dob": dob,
                    "gender": patient.get("gender", ""),
                    "contact": patient.get("telecom", [])
                }
            else:
                result = {"found": False, "mrn": "Patient not found"}

            self._log_query("query_patient_by_name_dob", params, result)
            return result

        except Exception as e:
            error_result = {"error": str(e), "found": False}
            self._log_query("query_patient_by_name_dob",
                            {"given_name": given_name,
                                "family_name": family_name, "dob": dob},
                            error_result)
            return error_result

    def query_patient_by_mrn(self, mrn: str) -> Dict[str, Any]:
        """
        Query patient by MRN (Medical Record Number).

        Args:
            mrn: Patient's MRN

        Returns:
            Patient demographics and information
        """
        try:
            url = f"{self.fhir_server_url}/Patient/{mrn}"
            headers = self._get_headers()
            response = requests.get(
                url,
                headers=headers,
                timeout=self.request_timeout
            )
            response.raise_for_status()

            patient = response.json()

            # Calculate age
            dob = patient.get("birthDate", "")
            age = self._calculate_age(dob) if dob else None

            result = {
                "mrn": mrn,
                "name": self._format_name(patient.get("name", [{}])[0]),
                "dob": dob,
                "age": age,
                "gender": patient.get("gender", ""),
                "telecom": patient.get("telecom", [])
            }

            self._log_query("query_patient_by_mrn", {"mrn": mrn}, result)
            return result

        except Exception as e:
            error_result = {"error": str(e), "mrn": mrn}
            self._log_query("query_patient_by_mrn", {"mrn": mrn}, error_result)
            return error_result

    def query_lab_value(self, patient_mrn: str, lab_code: str,
                        hours_back: int = 24) -> Dict[str, Any]:
        """
        Query lab values for a patient.

        Args:
            patient_mrn: Patient MRN
            lab_code: Lab code (e.g., "MG" for magnesium, "GLU" for glucose)
            hours_back: How many hours back to search (default 24)

        Returns:
            Most recent lab value and its timestamp
        """
        try:
            url = f"{self.fhir_server_url}/Observation"
            date_from = (datetime.now() -
                         timedelta(hours=hours_back)).isoformat()

            params = {
                "subject": patient_mrn,
                "code": lab_code,
                "date": f"ge{date_from}"
            }

            headers = self._get_headers()
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=self.request_timeout
            )
            response.raise_for_status()

            data = response.json()
            observations = [entry["resource"]
                            for entry in data.get("entry", [])]

            if observations:
                # Get most recent
                observations.sort(key=lambda x: x.get(
                    "effectiveDateTime", ""), reverse=True)
                obs = observations[0]

                value = obs.get("valueQuantity", {})
                result = {
                    "found": True,
                    "value": value.get("value", ""),
                    "unit": value.get("unit", ""),
                    "timestamp": obs.get("effectiveDateTime", ""),
                    "lab_code": lab_code,
                    "patient_mrn": patient_mrn
                }
            else:
                result = {
                    "found": False,
                    "value": -1,
                    "lab_code": lab_code,
                    "patient_mrn": patient_mrn
                }

            self._log_query("query_lab_value",
                            {"patient_mrn": patient_mrn, "lab_code": lab_code}, result)
            return result

        except Exception as e:
            error_result = {"error": str(e), "found": False, "value": -1}
            self._log_query("query_lab_value",
                            {"patient_mrn": patient_mrn, "lab_code": lab_code}, error_result)
            return error_result

    def query_lab_values_range(self, patient_mrn: str, lab_code: str,
                               hours_back: int = 24) -> Dict[str, Any]:
        """
        Query multiple lab values within a time range (for averaging).

        Args:
            patient_mrn: Patient MRN
            lab_code: Lab code
            hours_back: How many hours back to search

        Returns:
            List of lab values with timestamps
        """
        try:
            url = f"{self.fhir_server_url}/Observation"
            date_from = (datetime.now() -
                         timedelta(hours=hours_back)).isoformat()

            params = {
                "subject": patient_mrn,
                "code": lab_code,
                "date": f"ge{date_from}"
            }

            headers = self._get_headers()
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=self.request_timeout
            )
            response.raise_for_status()

            data = response.json()
            observations = [entry["resource"]
                            for entry in data.get("entry", [])]

            values = []
            for obs in observations:
                value = obs.get("valueQuantity", {})
                values.append({
                    "value": float(value.get("value", 0)),
                    "unit": value.get("unit", ""),
                    "timestamp": obs.get("effectiveDateTime", "")
                })

            result = {
                "found": len(values) > 0,
                "count": len(values),
                "values": values,
                "lab_code": lab_code,
                "patient_mrn": patient_mrn
            }

            self._log_query("query_lab_values_range",
                            {"patient_mrn": patient_mrn, "lab_code": lab_code}, result)
            return result

        except Exception as e:
            error_result = {"error": str(e), "found": False, "values": []}
            self._log_query("query_lab_values_range",
                            {"patient_mrn": patient_mrn, "lab_code": lab_code}, error_result)
            return error_result

    def create_observation(self, patient_mrn: str, lab_code: str,
                           value: float, unit: str,
                           effective_datetime: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new observation (record lab value).

        Args:
            patient_mrn: Patient MRN
            lab_code: Lab code
            value: Lab value
            unit: Unit of measurement

        Returns:
            Creation result with observation ID
        """
        try:
            url = f"{self.fhir_server_url}/Observation"

            observation = build_observation_resource(
                patient_mrn=patient_mrn,
                lab_code=lab_code,
                value=value,
                unit=unit,
                effective_datetime=effective_datetime,
            )

            headers = self._get_headers()
            response = requests.post(
                url,
                json=observation,
                headers=headers,
                timeout=self.request_timeout
            )
            response.raise_for_status()

            data = response.json()
            result = {
                "created": True,
                "observation_id": data.get("id", ""),
                "patient_mrn": patient_mrn,
                "lab_code": lab_code,
                "value": value,
                "unit": unit
            }

            self._log_query("create_observation",
                            {"patient_mrn": patient_mrn, "lab_code": lab_code},
                            result)
            return result

        except Exception as e:
            error_result = {"created": False, "error": str(e)}
            self._log_query("create_observation",
                            {"patient_mrn": patient_mrn, "lab_code": lab_code},
                            error_result)
            return error_result

    def create_medication_request(self, patient_mrn: str, medication_ndc: str,
                                  dosage: str, reason: str = "",
                                  medication_text: Optional[str] = None,
                                  route: Optional[str] = None,
                                  dose_value: Optional[float] = None,
                                  dose_unit: Optional[str] = None,
                                  rate_value: Optional[float] = None,
                                  rate_unit: Optional[str] = None,
                                  authored_on: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a medication request (order medication).

        Args:
            patient_mrn: Patient MRN
            medication_ndc: NDC code for medication
            dosage: Dosage instructions
            reason: Reason for medication

        Returns:
            Creation result
        """
        try:
            url = f"{self.fhir_server_url}/MedicationRequest"

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

            headers = self._get_headers()
            response = requests.post(
                url,
                json=med_request,
                headers=headers,
                timeout=self.request_timeout
            )
            response.raise_for_status()

            data = response.json()
            result = {
                "created": True,
                "medication_request_id": data.get("id", ""),
                "patient_mrn": patient_mrn,
                "medication_ndc": medication_ndc,
                "dosage": dosage
            }

            self._log_query("create_medication_request",
                            {"patient_mrn": patient_mrn,
                                "medication_ndc": medication_ndc},
                            result)
            return result

        except Exception as e:
            error_result = {"created": False, "error": str(e)}
            self._log_query("create_medication_request",
                            {"patient_mrn": patient_mrn,
                                "medication_ndc": medication_ndc},
                            error_result)
            return error_result

    def create_condition(self, patient_mrn: str, code: str,
                         display: Optional[str] = None,
                         onset_datetime: Optional[str] = None,
                         recorded_date: Optional[str] = None,
                         coding_system: str = "http://hl7.org/fhir/sid/icd-10") -> Dict[str, Any]:
        """
        Create a Condition resource using the shared benchmark FHIR shape.
        """
        try:
            condition = build_condition_resource(
                patient_mrn=patient_mrn,
                code=code,
                display=display,
                onset_datetime=onset_datetime,
                recorded_date=recorded_date,
                coding_system=coding_system,
            )
            result = self._post_fhir_resource("Condition", condition)
            self._log_query("create_condition",
                            {"patient_mrn": patient_mrn, "code": code},
                            result)
            return result
        except Exception as e:
            error_result = {"created": False, "error": str(e)}
            self._log_query("create_condition",
                            {"patient_mrn": patient_mrn, "code": code},
                            error_result)
            return error_result

    def create_procedure(self, patient_mrn: str, code: str,
                         display: Optional[str] = None,
                         performed_datetime: Optional[str] = None,
                         coding_system: str = "http://www.ama-assn.org/go/cpt") -> Dict[str, Any]:
        """
        Create a Procedure resource using the shared benchmark FHIR shape.
        """
        try:
            procedure = build_procedure_resource(
                patient_mrn=patient_mrn,
                code=code,
                display=display,
                performed_datetime=performed_datetime,
                coding_system=coding_system,
            )
            result = self._post_fhir_resource("Procedure", procedure)
            self._log_query("create_procedure",
                            {"patient_mrn": patient_mrn, "code": code},
                            result)
            return result
        except Exception as e:
            error_result = {"created": False, "error": str(e)}
            self._log_query("create_procedure",
                            {"patient_mrn": patient_mrn, "code": code},
                            error_result)
            return error_result

    def create_service_request(self, patient_mrn: str, service_code: str,
                               description: str) -> Dict[str, Any]:
        """
        Create a service request (order referral or procedure).

        Args:
            patient_mrn: Patient MRN
            service_code: SNOMED code for service
            description: Description of service/referral

        Returns:
            Creation result
        """
        try:
            url = f"{self.fhir_server_url}/ServiceRequest"

            service_request = {
                "resourceType": "ServiceRequest",
                "status": "draft",
                "intent": "order",
                "subject": {
                    "reference": f"Patient/{patient_mrn}"
                },
                "code": {
                    "coding": [{"code": service_code}]
                },
                "note": [
                    {
                        "text": description
                    }
                ],
                "authoredOn": datetime.now().isoformat()
            }

            headers = self._get_headers()
            response = requests.post(
                url,
                json=service_request,
                headers=headers,
                timeout=self.request_timeout
            )
            response.raise_for_status()

            data = response.json()
            result = {
                "created": True,
                "service_request_id": data.get("id", ""),
                "patient_mrn": patient_mrn,
                "service_code": service_code,
                "description": description
            }

            self._log_query("create_service_request",
                            {"patient_mrn": patient_mrn,
                                "service_code": service_code},
                            result)
            return result

        except Exception as e:
            error_result = {"created": False, "error": str(e)}
            self._log_query("create_service_request",
                            {"patient_mrn": patient_mrn,
                                "service_code": service_code},
                            error_result)
            return error_result

    def _calculate_age(self, dob: str, current_date: Optional[str] = None) -> int:
        """Calculate age from date of birth."""
        try:
            birth_date = datetime.fromisoformat(dob.replace("Z", "+00:00"))
            if current_date:
                current = datetime.fromisoformat(
                    current_date.replace("Z", "+00:00"))
            else:
                current = datetime.now()

            age = (current - birth_date).days // 365
            return age
        except Exception:
            return 0

    def _format_name(self, name_dict: Dict[str, Any]) -> str:
        """Format name from FHIR name object."""
        given = " ".join(name_dict.get("given", []))
        family = name_dict.get("family", "")
        return f"{given} {family}".strip()

    def _log_query(self, operation: str, params: Dict[str, Any],
                   result: Dict[str, Any]) -> None:
        """Log query for audit and debugging."""
        self.query_history.append({
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "params": params,
            "result": result
        })

    def get_query_history(self) -> List[Dict[str, Any]]:
        """Get query history."""
        return self.query_history
