"""
Medical Calculation Agent - Medical calculations and clinical decision support
Performs medical calculations, clinical reasoning, and decision support.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import math
from langchain_core.language_models import BaseLanguageModel
from enum import Enum


class RiskLevel(Enum):
    """Risk levels for clinical decisions."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LabValueAssessment:
    """Assessment of a lab value."""
    value: float
    reference_low: float
    reference_high: float
    status: str  # low, normal, high, critical
    severity: RiskLevel


class MedicalCalculationAgent:
    """
    Medical Calculation Agent for clinical calculations and decision support.
    - Perform medical calculations (age, BMI, eGFR, etc.)
    - Assess lab values
    - Clinical decision support
    - Dosing calculations
    - Risk stratification
    """

    def __init__(self, llm: BaseLanguageModel):
        """
        Initialize Medical Calculation Agent.

        Args:
            llm: Language model for clinical reasoning
        """
        self.llm = llm
        self.calculation_history: List[Dict[str, Any]] = []

    def calculate_age(self, dob: str, reference_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate age from date of birth.

        Args:
            dob: Date of birth (ISO format)
            reference_date: Reference date for age calculation (default: today)

        Returns:
            Age in years
        """
        try:
            birth_date = datetime.fromisoformat(dob.replace("Z", "+00:00"))
            if reference_date:
                ref_date = datetime.fromisoformat(
                    reference_date.replace("Z", "+00:00"))
            else:
                ref_date = datetime.now()

            age_days = (ref_date - birth_date).days
            age_years = age_days // 365

            result = {
                "dob": dob,
                "age_years": age_years,
                "age_days": age_days,
                "reference_date": reference_date or datetime.now().isoformat()
            }
            self._log_calculation("calculate_age", dob, result)
            return result

        except Exception as e:
            error_result = {"error": str(e)}
            self._log_calculation("calculate_age", dob, error_result)
            return error_result

    def calculate_bmi(self, weight_kg: float, height_m: float) -> Dict[str, Any]:
        """
        Calculate Body Mass Index.

        Args:
            weight_kg: Weight in kilograms
            height_m: Height in meters

        Returns:
            BMI and classification
        """
        try:
            bmi = weight_kg / (height_m ** 2)

            if bmi < 18.5:
                classification = "underweight"
            elif bmi < 25:
                classification = "normal_weight"
            elif bmi < 30:
                classification = "overweight"
            else:
                classification = "obese"

            result = {
                "weight_kg": weight_kg,
                "height_m": height_m,
                "bmi": round(bmi, 1),
                "classification": classification
            }
            self._log_calculation("calculate_bmi",
                                  {"weight_kg": weight_kg, "height_m": height_m}, result)
            return result

        except Exception as e:
            error_result = {"error": str(e)}
            self._log_calculation("calculate_bmi",
                                  {"weight_kg": weight_kg, "height_m": height_m}, error_result)
            return error_result

    def calculate_egfr(self, age: int, gender: str, creatinine: float) -> Dict[str, Any]:
        """
        Calculate estimated Glomerular Filtration Rate (eGFR) - MDRD formula.

        Args:
            age: Age in years
            gender: Gender (male/female)
            creatinine: Serum creatinine in mg/dL

        Returns:
            eGFR and renal function category
        """
        try:
            # MDRD formula
            gender_factor = 0.742 if gender.lower() == "female" else 1.0
            race_factor = 1.212 if "african" in gender.lower() else 1.0  # Simplified

            egfr = 175 * (creatinine ** -1.154) * \
                (age ** -0.203) * gender_factor * race_factor

            if egfr >= 90:
                category = "normal"
            elif egfr >= 60:
                category = "mild"
            elif egfr >= 45:
                category = "mild_moderate"
            elif egfr >= 30:
                category = "moderate_severe"
            elif egfr >= 15:
                category = "severe"
            else:
                category = "kidney_failure"

            result = {
                "age": age,
                "gender": gender,
                "creatinine": creatinine,
                "egfr": round(egfr, 1),
                "category": category
            }
            self._log_calculation("calculate_egfr",
                                  {"age": age, "gender": gender,
                                      "creatinine": creatinine},
                                  result)
            return result

        except Exception as e:
            error_result = {"error": str(e)}
            self._log_calculation("calculate_egfr",
                                  {"age": age, "gender": gender,
                                      "creatinine": creatinine},
                                  error_result)
            return error_result

    def assess_lab_value(self, value: float, lab_code: str) -> Dict[str, Any]:
        """
        Assess a lab value against reference ranges.

        Args:
            value: Lab value
            lab_code: Lab code (MG, K, GLU, A1C, etc.)

        Returns:
            Assessment including status and recommendations
        """

        references = {
            "MG": {
                "normal_low": 1.7,
                "normal_high": 2.2,
                "mild_def_low": 1.5,
                "moderate_def_low": 1.0,
                "units": "mg/dL"
            },
            "K": {
                "normal_low": 3.5,
                "normal_high": 5.0,
                "critical_low": 2.5,
                "critical_high": 6.5,
                "units": "mEq/L"
            },
            "GLU": {
                "normal_low": 70,
                "normal_high": 100,
                "prediabetic_high": 125,
                "diabetic_high": 126,
                "units": "mg/dL"
            },
            "A1C": {
                "normal_high": 5.7,
                "prediabetic_high": 6.4,
                "diabetic_low": 6.5,
                "units": "%"
            }
        }

        if lab_code not in references:
            return {
                "error": f"Lab reference not found for {lab_code}",
                "value": value
            }

        ref = references[lab_code]

        # Determine status and severity
        if lab_code == "MG":
            if value < 1.0:
                status = "severe_deficiency"
                severity = RiskLevel.CRITICAL
                recommendation = "Requires urgent IV magnesium replacement - 4g over 4 hours"
            elif value < 1.5:
                status = "moderate_deficiency"
                severity = RiskLevel.HIGH
                recommendation = "Order replacement IV magnesium - 2g over 2 hours"
            elif value < ref["normal_low"]:
                status = "mild_deficiency"
                severity = RiskLevel.MODERATE
                recommendation = "Consider IV magnesium replacement - 1g over 1 hour"
            elif value <= ref["normal_high"]:
                status = "normal"
                severity = RiskLevel.LOW
                recommendation = "No action needed"
            else:
                status = "high"
                severity = RiskLevel.MODERATE
                recommendation = "Monitor; consider diuretics if symptomatic"

        elif lab_code == "K":
            if value < 2.5:
                status = "critical_low"
                severity = RiskLevel.CRITICAL
                recommendation = "Urgent action required - high arrhythmia risk"
            elif value < ref["normal_low"]:
                status = "low"
                severity = RiskLevel.HIGH
                recommendation = "Order potassium replacement"
            elif value <= ref["normal_high"]:
                status = "normal"
                severity = RiskLevel.LOW
                recommendation = "No action needed"
            else:
                status = "high"
                severity = RiskLevel.MODERATE
                recommendation = "Monitor closely; may need dialysis support"

        elif lab_code == "GLU":
            if value < 70:
                status = "hypoglycemia"
                severity = RiskLevel.HIGH
                recommendation = "Treat with fast-acting carbohydrates immediately"
            elif value <= ref["normal_high"]:
                status = "normal"
                severity = RiskLevel.LOW
                recommendation = "No action needed"
            else:
                status = "elevated"
                severity = RiskLevel.MODERATE
                recommendation = "Monitor; may indicate diabetes or stress response"

        else:
            status = "unknown"
            severity = RiskLevel.LOW
            recommendation = "Unable to assess without reference data"

        result = {
            "value": value,
            "lab_code": lab_code,
            "units": ref.get("units", ""),
            "status": status,
            "severity": severity.value,
            "reference_range": {
                "low": ref.get("normal_low"),
                "high": ref.get("normal_high")
            },
            "recommendation": recommendation
        }

        self._log_calculation("assess_lab_value", {
                              "value": value, "lab_code": lab_code}, result)
        return result

    def calculate_average(self, values: List[float]) -> Dict[str, Any]:
        """
        Calculate average of lab values.

        Args:
            values: List of lab values

        Returns:
            Average and statistics
        """
        try:
            if not values:
                return {"error": "No values provided"}

            average = sum(values) / len(values)
            sorted_vals = sorted(values)

            if len(values) % 2 == 0:
                median = (sorted_vals[len(values)//2 - 1] +
                          sorted_vals[len(values)//2]) / 2
            else:
                median = sorted_vals[len(values)//2]

            variance = sum((x - average) ** 2 for x in values) / len(values)
            std_dev = math.sqrt(variance)

            result = {
                "count": len(values),
                "average": round(average, 2),
                "median": round(median, 2),
                "minimum": round(min(values), 2),
                "maximum": round(max(values), 2),
                "std_dev": round(std_dev, 2)
            }

            self._log_calculation("calculate_average", {
                                  "count": len(values)}, result)
            return result

        except Exception as e:
            error_result = {"error": str(e)}
            self._log_calculation("calculate_average", {}, error_result)
            return error_result

    def assess_medication_need(self, lab_value: float, lab_code: str,
                               patient_weight: float = 70) -> Dict[str, Any]:
        """
        Assess whether medication is needed and calculate dosing.

        Args:
            lab_value: Lab value
            lab_code: Lab code
            patient_weight: Patient weight in kg

        Returns:
            Medication recommendation and dosing
        """

        assessment = self.assess_lab_value(lab_value, lab_code)

        if "error" in assessment:
            return assessment

        medication_recommendations = {
            "MG": {
                "medication": "IV Magnesium",
                "ndc": "0338-1715-40",
                "severe_deficiency": f"4g IV over 4 hours",
                "moderate_deficiency": f"2g IV over 2 hours",
                "mild_deficiency": f"1g IV over 1 hour"
            },
            "K": {
                "medication": "Potassium replacement",
                "ndc": "40032-917-01",
                "calculation": "10 mEq per 0.1 mEq/L below 3.5"
            }
        }

        result = {
            **assessment,
            "needs_medication": assessment["severity"] != RiskLevel.LOW.value,
            "action_required": assessment["severity"] in [RiskLevel.HIGH.value, RiskLevel.CRITICAL.value]
        }

        if lab_code in medication_recommendations:
            rec = medication_recommendations[lab_code]
            result["medication_info"] = rec

            # Calculate specific dose for magnesium
            if lab_code == "MG":
                severity_key = None
                if assessment["status"] == "severe_deficiency":
                    severity_key = "severe_deficiency"
                elif assessment["status"] == "moderate_deficiency":
                    severity_key = "moderate_deficiency"
                elif assessment["status"] == "mild_deficiency":
                    severity_key = "mild_deficiency"

                if severity_key:
                    result["recommended_dose"] = rec[severity_key]

            # Calculate specific dose for potassium
            elif lab_code == "K":
                deficit = max(0, 3.5 - lab_value)
                mEq_needed = deficit / 0.1 * 10
                result["estimated_mEq_needed"] = round(mEq_needed, 0)
                result["recommended_dose"] = f"{int(round(mEq_needed, 0))} mEq oral potassium"

        self._log_calculation("assess_medication_need",
                              {"lab_value": lab_value, "lab_code": lab_code},
                              result)
        return result

    def format_clinical_summary(self, lab_code: str, value: float,
                                lab_timestamp: str = "") -> Dict[str, Any]:
        """
        Format clinical summary for display.

        Args:
            lab_code: Lab code
            value: Lab value
            lab_timestamp: Timestamp of lab

        Returns:
            Formatted summary
        """
        assessment = self.assess_lab_value(value, lab_code)

        summary = f"Laboratory Finding:\n"
        summary += f"Test: {lab_code}\n"
        summary += f"Value: {value} {assessment.get('units', '')}\n"
        summary += f"Status: {assessment.get('status', 'Unknown').upper()}\n"
        summary += f"Severity: {assessment.get('severity', 'Unknown').upper()}\n"
        summary += f"Recommendation: {assessment.get('recommendation', 'No recommendation')}\n"

        if lab_timestamp:
            summary += f"Timestamp: {lab_timestamp}\n"

        return {
            "formatted_summary": summary,
            "assessment": assessment
        }

    def _log_calculation(self, calculation_type: str, input_data: Any,
                         result: Dict[str, Any]) -> None:
        """Log calculation for audit."""
        self.calculation_history.append({
            "timestamp": datetime.now().isoformat(),
            "calculation_type": calculation_type,
            "input": input_data,
            "result": result
        })

    def get_calculation_history(self) -> List[Dict[str, Any]]:
        """Get calculation history."""
        return self.calculation_history
