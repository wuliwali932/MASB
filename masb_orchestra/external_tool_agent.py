"""
External Tool Agent - Call external APIs and tools
Manages integration with external medical tools, databases, and services.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import json
from langchain_core.language_models import BaseLanguageModel
from enum import Enum


class ToolCategory(Enum):
    """Categories of external tools."""
    LABORATORY = "laboratory"
    PHARMACY = "pharmacy"
    IMAGING = "imaging"
    REFERENCE = "reference"
    NOTIFICATION = "notification"
    CUSTOM = "custom"


@dataclass
class ExternalTool:
    """Definition of an external tool."""
    name: str
    category: ToolCategory
    description: str
    function: Callable
    parameters: List[Dict[str, Any]]
    requires_auth: bool = False


class ExternalToolAgent:
    """
    External Tool Agent for calling external APIs and services.
    - Integrate with external medical tools
    - Call reference APIs
    - Manage tool registrations
    - Execute tool calls with proper error handling
    """

    def __init__(self, llm: BaseLanguageModel):
        """
        Initialize External Tool Agent.

        Args:
            llm: Language model for tool selection and orchestration
        """
        self.llm = llm
        self.registered_tools: Dict[str, ExternalTool] = {}
        self.tool_history: List[Dict[str, Any]] = []
        self._initialize_default_tools()

    def _initialize_default_tools(self):
        """Initialize default external tools."""
        # Laboratory reference tool
        self.register_tool(
            "lab_reference",
            ToolCategory.REFERENCE,
            "Look up lab value reference ranges and normal values",
            self._lookup_lab_reference,
            [
                {"name": "lab_code", "type": "string",
                    "description": "Lab code (e.g., MG, GLU, K)"},
                {"name": "units", "type": "string",
                    "description": "Units of measurement"}
            ]
        )

        # Medical dosing reference tool
        self.register_tool(
            "dosing_calculator",
            ToolCategory.REFERENCE,
            "Calculate medication dosing based on patient parameters",
            self._calculate_dosing,
            [
                {"name": "medication", "type": "string",
                    "description": "Medication name"},
                {"name": "patient_weight", "type": "float",
                    "description": "Patient weight in kg"},
                {"name": "renal_function", "type": "string",
                    "description": "Renal function (normal/mild/moderate/severe)"}
            ]
        )

        # Drug interaction checker
        self.register_tool(
            "drug_interaction_checker",
            ToolCategory.PHARMACY,
            "Check for drug-drug interactions",
            self._check_drug_interactions,
            [
                {"name": "medications", "type": "list",
                    "description": "List of medication names"}
            ]
        )

        # Allergy checker
        self.register_tool(
            "allergy_checker",
            ToolCategory.REFERENCE,
            "Check for known allergies to medications",
            self._check_allergies,
            [
                {"name": "medication", "type": "string",
                    "description": "Medication name"},
                {"name": "patient_allergies", "type": "list",
                    "description": "Patient's known allergies"}
            ]
        )

        # Notification tool
        self.register_tool(
            "send_notification",
            ToolCategory.NOTIFICATION,
            "Send notifications to providers or patients",
            self._send_notification,
            [
                {"name": "recipient", "type": "string",
                    "description": "Recipient identifier"},
                {"name": "message", "type": "string",
                    "description": "Message content"},
                {"name": "priority", "type": "string",
                    "description": "Priority (low/normal/high/urgent)"}
            ]
        )

    def register_tool(self, name: str, category: ToolCategory,
                      description: str, function: Callable,
                      parameters: List[Dict[str, Any]],
                      requires_auth: bool = False) -> None:
        """
        Register an external tool.

        Args:
            name: Tool name
            category: Tool category
            description: Tool description
            function: Callable function
            parameters: List of parameter definitions
            requires_auth: Whether tool requires authentication
        """
        tool = ExternalTool(
            name=name,
            category=category,
            description=description,
            function=function,
            parameters=parameters,
            requires_auth=requires_auth
        )
        self.registered_tools[name] = tool

    def list_tools(self, category: Optional[ToolCategory] = None) -> List[Dict[str, Any]]:
        """
        List available tools.

        Args:
            category: Optional filter by category

        Returns:
            List of tool definitions
        """
        tools = []
        for name, tool in self.registered_tools.items():
            if category is None or tool.category == category:
                tools.append({
                    "name": name,
                    "category": tool.category.value,
                    "description": tool.description,
                    "parameters": tool.parameters
                })
        return tools

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute an external tool.

        Args:
            tool_name: Name of tool to execute
            **kwargs: Tool parameters

        Returns:
            Tool execution result
        """
        if tool_name not in self.registered_tools:
            result = {"error": f"Tool '{tool_name}' not found"}
            self._log_tool_call(tool_name, kwargs, result)
            return result

        try:
            tool = self.registered_tools[tool_name]
            result = tool.function(**kwargs)
            self._log_tool_call(tool_name, kwargs, result)
            return result
        except Exception as e:
            error_result = {"error": str(e), "tool": tool_name}
            self._log_tool_call(tool_name, kwargs, error_result)
            return error_result

    # Default tool implementations

    def _lookup_lab_reference(self, lab_code: str, units: str = "") -> Dict[str, Any]:
        """Look up lab reference ranges."""

        reference_ranges = {
            "MG": {  # Magnesium
                "normal_low": 1.7,
                "normal_high": 2.2,
                "units": "mg/dL",
                "mild_deficiency": {"low": 1.5, "high": 1.9},
                "moderate_deficiency": {"low": 1.0, "high": 1.4},
                "severe_deficiency": {"high": 1.0}
            },
            "K": {  # Potassium
                "normal_low": 3.5,
                "normal_high": 5.0,
                "units": "mEq/L",
                "critical_low": 2.5,
                "critical_high": 6.5
            },
            "GLU": {  # Glucose (fasting)
                "normal_low": 70,
                "normal_high": 100,
                "units": "mg/dL",
                "prediabetic_low": 100,
                "prediabetic_high": 125,
                "diabetic_high": 126
            },
            "A1C": {  # HbA1C
                "normal_high": 5.7,
                "units": "%",
                "prediabetic_low": 5.7,
                "prediabetic_high": 6.4,
                "diabetic_low": 6.5
            },
            "BP": {  # Blood Pressure
                "normal": "< 120/80",
                "elevated": "120-129 / < 80",
                "units": "mmHg"
            }
        }

        if lab_code in reference_ranges:
            return {
                "found": True,
                "lab_code": lab_code,
                "reference": reference_ranges[lab_code]
            }
        else:
            return {
                "found": False,
                "lab_code": lab_code,
                "error": f"Reference range not found for {lab_code}"
            }

    def _calculate_dosing(self, medication: str, patient_weight: float,
                          renal_function: str = "normal") -> Dict[str, Any]:
        """Calculate medication dosing."""

        dosing_guidelines = {
            "magnesium": {
                "base_dose": 1.0,  # grams
                "weight_factor": 0.01,  # per kg
                "renal_adjustments": {
                    "normal": 1.0,
                    "mild": 0.9,
                    "moderate": 0.7,
                    "severe": 0.5
                }
            },
            "potassium": {
                "base_dose": 10,  # mEq
                "weight_factor": 0.2,
                "renal_adjustments": {
                    "normal": 1.0,
                    "mild": 0.8,
                    "moderate": 0.5,
                    "severe": 0.3
                }
            }
        }

        medication_lower = medication.lower()
        if medication_lower in dosing_guidelines:
            guideline = dosing_guidelines[medication_lower]
            base = guideline["base_dose"]
            weight_adjustment = patient_weight * guideline["weight_factor"]
            renal_adjustment = guideline["renal_adjustments"].get(
                renal_function, 1.0)

            calculated_dose = (base + weight_adjustment) * renal_adjustment

            return {
                "medication": medication,
                "calculated_dose": round(calculated_dose, 2),
                "patient_weight": patient_weight,
                "renal_function": renal_function,
                "basis": "Weight-based calculation with renal adjustment"
            }
        else:
            return {
                "error": f"Dosing guidelines not found for {medication}",
                "medication": medication
            }

    def _check_drug_interactions(self, medications: List[str]) -> Dict[str, Any]:
        """Check for drug-drug interactions."""

        # Simplified interaction database
        known_interactions = {
            ("magnesium", "potassium"): {
                "severity": "moderate",
                "description": "Concurrent use may require monitoring of serum levels"
            },
            ("warfarin", "aspirin"): {
                "severity": "high",
                "description": "Increased bleeding risk"
            }
        }

        interactions = []
        medications_lower = [m.lower() for m in medications]

        for i, med1 in enumerate(medications_lower):
            for med2 in medications_lower[i+1:]:
                key = tuple(sorted([med1, med2]))
                if key in known_interactions:
                    interactions.append({
                        "drug1": med1,
                        "drug2": med2,
                        **known_interactions[key]
                    })

        return {
            "medications": medications,
            "interactions_found": len(interactions) > 0,
            "interactions": interactions
        }

    def _check_allergies(self, medication: str,
                         patient_allergies: List[str]) -> Dict[str, Any]:
        """Check for drug-allergy conflicts."""

        # Simplified allergy cross-reference
        drug_components = {
            "amoxicillin": ["penicillin", "beta-lactams"],
            "magnesium": ["magnesium"],
            "potassium": ["potassium"]
        }

        medication_lower = medication.lower()
        patient_allergies_lower = [a.lower() for a in patient_allergies]

        components = drug_components.get(medication_lower, [medication_lower])
        conflicting_allergies = [
            allergy for allergy in patient_allergies_lower
            if any(comp in allergy or allergy in comp for comp in components)
        ]

        return {
            "medication": medication,
            "patient_allergies": patient_allergies,
            "conflict_found": len(conflicting_allergies) > 0,
            "conflicting_allergies": conflicting_allergies,
            "safe": len(conflicting_allergies) == 0
        }

    def _send_notification(self, recipient: str, message: str,
                           priority: str = "normal") -> Dict[str, Any]:
        """Send notification."""

        # In production, this would integrate with notification service
        return {
            "status": "sent",
            "recipient": recipient,
            "message": message,
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
            "notification_id": f"notif_{datetime.now().timestamp()}"
        }

    def _log_tool_call(self, tool_name: str, params: Dict[str, Any],
                       result: Dict[str, Any]) -> None:
        """Log tool call for audit and debugging."""
        self.tool_history.append({
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "parameters": params,
            "result": result
        })

    def get_tool_history(self) -> List[Dict[str, Any]]:
        """Get tool call history."""
        return self.tool_history

    def execute_tool_from_instruction(self, instruction: str) -> Dict[str, Any]:
        """
        Use LLM to interpret instruction and execute appropriate tool.

        Args:
            instruction: Natural language instruction

        Returns:
            Tool execution result
        """
        tool_selection_prompt = f"""
        Based on the following instruction, identify which external tool should be called
        and what parameters it should receive. Respond as JSON with "tool_name" and "parameters".
        
        Available tools: {json.dumps(self.list_tools(), indent=2)}
        
        Instruction: {instruction}
        
        Respond only with valid JSON.
        """

        try:
            response = self.llm.invoke(tool_selection_prompt)
            raw_response = getattr(response, "content", str(response))
            self.tool_history.append({
                "timestamp": datetime.now().isoformat(),
                "tool": "llm_tool_selection",
                "parameters": {"instruction": instruction},
                "prompt": tool_selection_prompt,
                "llm_output": raw_response,
            })
            tool_request = json.loads(raw_response)

            tool_name = tool_request.get("tool_name")
            parameters = tool_request.get("parameters", {})

            return self.execute_tool(tool_name, **parameters)
        except Exception as e:
            return {"error": f"Failed to execute tool from instruction: {str(e)}"}
