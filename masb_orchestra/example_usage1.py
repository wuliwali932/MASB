"""
Example usage and test of the multi-agent medical orchestration system.
Demonstrates orchestrator, EHR query, external tool, and medical calculation agents.
"""

import json
from datetime import datetime
from .multi_agent_orchestrator import MultiAgentMedicalOrchestrator
from .orchestrator_agent import OrchestratorAgent
from .ehr_query_agent import EHRQueryAgent
from .external_tool_agent import ExternalToolAgent
from .medical_calculation_agent import MedicalCalculationAgent


def example_1_simple_query(orchestrator: MultiAgentMedicalOrchestrator):
    """Example 1: Simple MRN lookup query."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Patient Query")
    print("="*70)

    instruction = "What is the MRN of patient Peter Stafford with DOB 1932-12-29?"
    context = "You are the attending physician."

    result = orchestrator.execute(instruction, context)
    print(f"\nFinal Response: {result['final_response']}")
    print(f"Status: {result['status']}")

    return result


def example_2_compound_task(orchestrator: MultiAgentMedicalOrchestrator):
    """Example 2: Compound task with medical reasoning."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Compound Task - Check Lab and Order if Low")
    print("="*70)

    instruction = """Check patient S6315806's last serum magnesium level within last 24 hours.
    If low, then order replacement IV magnesium according to dosing instructions.
    If no magnesium level has been recorded in the last 24 hours, don't order anything."""

    context = """It's 2023-11-13T10:15:00+00:00 now. The code for magnesium is "MG".
    The NDC for replacement IV magnesium is 0338-1715-40.
    Dosing instructions:
    (1) Mild deficiency (eg, serum magnesium 1.5 to 1.9 mg/dL): IV: 1 g over 1 hour.
    (2) Moderate deficiency (eg, serum magnesium 1 to <1.5 mg/dL): IV: 2 g over 2 hours.
    (3) Severe deficiency (eg, serum magnesium <1 mg/dL): IV: 4 g over 4 hours."""

    result = orchestrator.execute(instruction, context)
    print(f"\nFinal Response: {result['final_response']}")
    print(f"Status: {result['status']}")

    return result


def example_3_calculation_task(orchestrator: MultiAgentMedicalOrchestrator):
    """Example 3: Medical calculation task."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Medical Calculation - Average Lab Values")
    print("="*70)

    instruction = "What is the average CBG of patient S6307599 over the last 24 hours?"
    context = """It's 2023-11-13T10:15:00+00:00 now.
    The code for CBG is "GLU".
    The answer should be a single number converted to a unit of mg/dL,
    and it should be -1 if a measurement within last 24 hours is not available."""

    result = orchestrator.execute(instruction, context)
    print(f"\nFinal Response: {result['final_response']}")
    print(f"Status: {result['status']}")

    return result


def demonstrate_agent_capabilities(orchestrator: MultiAgentMedicalOrchestrator):
    """Demonstrate capabilities of each subagent."""
    print("\n" + "="*70)
    print("AGENT CAPABILITIES DEMONSTRATION")
    print("="*70)

    capabilities = orchestrator.get_agent_capabilities()

    print("\n1. ORCHESTRATOR AGENT:")
    print("   Purpose: Task analysis, decomposition, and response merging")
    for cap in capabilities["orchestrator"]["capabilities"]:
        print(f"   ✓ {cap}")

    print("\n2. EHR QUERY AGENT:")
    print("   Purpose: CRUD operations on FHIR medical data")
    for cap in capabilities["ehr_query"]["capabilities"]:
        print(f"   ✓ {cap}")

    print("\n3. EXTERNAL TOOL AGENT:")
    print("   Purpose: Integration with external medical tools and APIs")
    print("   Available Tools:")
    for tool_info in capabilities["external_tool"]["tools"]:
        print(f"   ✓ {tool_info['name']}: {tool_info['description']}")

    print("\n4. MEDICAL CALCULATION AGENT:")
    print("   Purpose: Medical calculations and clinical decision support")
    for cap in capabilities["medical_calc"]["capabilities"]:
        print(f"   ✓ {cap}")


def demonstrate_task_decomposition(llm):
    """Demonstrate task decomposition capability."""
    print("\n" + "="*70)
    print("TASK DECOMPOSITION DEMONSTRATION")
    print("="*70)

    orchestrator = OrchestratorAgent(llm)

    # Complex compound task
    instruction = """Check patient S6315806's magnesium level.
    If low (below 1.7 mg/dL), order replacement IV magnesium.
    If moderate deficiency, use 2g over 2 hours."""

    decomposition = orchestrator.decompose_task(instruction)

    print(f"\nOriginal Task: {decomposition.original_task}")
    print(f"\nDecomposed into {len(decomposition.subtasks)} subtasks:")

    for i, subtask in enumerate(decomposition.subtasks, 1):
        print(f"\n{i}. {subtask.task_id}")
        print(f"   Type: {subtask.task_type.value}")
        print(f"   Assigned to: {subtask.assigned_agent}")
        print(f"   Instruction: {subtask.instruction}")

    print(f"\nExecution Order: {decomposition.execution_order}")
    print(f"Dependencies: {decomposition.dependencies}")


def demonstrate_ehr_queries(llm, fhir_url: str = "http://localhost:8080/fhir"):
    """Demonstrate EHR query operations."""
    print("\n" + "="*70)
    print("EHR QUERY AGENT DEMONSTRATION")
    print("="*70)

    ehr_agent = EHRQueryAgent(llm, fhir_url)

    # Example patient query
    print("\n1. Query Patient by Name and DOB:")
    print("   (Would query real FHIR server in production)")
    print("   Input: given_name='Peter', family_name='Stafford', dob='1932-12-29'")
    print("   Expected Result: MRN found")

    # Example lab value query
    print("\n2. Query Lab Value:")
    print("   (Would query real FHIR server in production)")
    print("   Input: patient_mrn='S6315806', lab_code='MG', hours_back=24")
    print("   Expected Result: Most recent magnesium level")

    # Example create observation
    print("\n3. Create Observation (Record Lab Value):")
    print("   (Would POST to real FHIR server in production)")
    print("   Input: patient_mrn='S6315806', lab_code='MG', value=1.8, unit='mg/dL'")
    print("   Expected Result: Observation created with ID")


def demonstrate_medical_calculations(llm):
    """Demonstrate medical calculation operations."""
    print("\n" + "="*70)
    print("MEDICAL CALCULATION AGENT DEMONSTRATION")
    print("="*70)

    calc_agent = MedicalCalculationAgent(llm)

    # Example age calculation
    print("\n1. Calculate Age:")
    result = calc_agent.calculate_age("1963-01-29")
    print(f"   Input: dob='1963-01-29'")
    print(f"   Output: age_years={result.get('age_years')}")

    # Example BMI calculation
    print("\n2. Calculate BMI:")
    result = calc_agent.calculate_bmi(70, 1.75)
    print(f"   Input: weight_kg=70, height_m=1.75")
    print(
        f"   Output: bmi={result.get('bmi')}, classification={result.get('classification')}")

    # Example lab value assessment
    print("\n3. Assess Lab Value:")
    result = calc_agent.assess_lab_value(1.2, "MG")
    print(f"   Input: value=1.2, lab_code='MG'")
    print(
        f"   Output: status={result.get('status')}, severity={result.get('severity')}")
    print(f"   Recommendation: {result.get('recommendation')}")

    # Example medication assessment
    print("\n4. Assess Medication Need:")
    result = calc_agent.assess_medication_need(1.2, "MG", patient_weight=75)
    print(f"   Input: lab_value=1.2, lab_code='MG', weight=75kg")
    print(f"   Output: needs_medication={result.get('needs_medication')}")
    if result.get('recommended_dose'):
        print(f"   Recommended Dose: {result.get('recommended_dose')}")


def demonstrate_external_tools(llm):
    """Demonstrate external tool operations."""
    print("\n" + "="*70)
    print("EXTERNAL TOOL AGENT DEMONSTRATION")
    print("="*70)

    tool_agent = ExternalToolAgent(llm)

    # Example lab reference lookup
    print("\n1. Lab Reference Lookup:")
    result = tool_agent.execute_tool(
        "lab_reference", lab_code="MG", units="mg/dL")
    print(f"   Input: lab_code='MG'")
    if result.get('found'):
        print(f"   Output: Reference range found - {result['reference']}")

    # Example dosing calculator
    print("\n2. Dosing Calculator:")
    result = tool_agent.execute_tool("dosing_calculator",
                                     medication="magnesium",
                                     patient_weight=75,
                                     renal_function="normal")
    print(f"   Input: medication='magnesium', weight=75kg, renal_function='normal'")
    if result.get('calculated_dose'):
        print(
            f"   Output: Calculated dose = {result['calculated_dose']} grams")

    # Example drug interaction check
    print("\n3. Drug Interaction Checker:")
    result = tool_agent.execute_tool("drug_interaction_checker",
                                     medications=["magnesium", "potassium"])
    print(f"   Input: medications=['magnesium', 'potassium']")
    print(
        f"   Output: Interactions found = {result.get('interactions_found')}")
    if result.get('interactions'):
        for interaction in result['interactions']:
            print(
                f"   - {interaction['drug1']} + {interaction['drug2']}: {interaction['severity']}")


def main():
    """Main demonstration of multi-agent medical orchestration."""

    # Note: In a real scenario, initialize with actual LLM
    # For now, using a mock/placeholder
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4", temperature=0)
    except:
        print(
            "Note: Using LLM placeholder. Install langchain-openai for full functionality.")
        llm = None

    if llm is None:
        print("\nDemonstrating with LLM placeholder (no actual LLM calls)")
        print("To use actual LLM, install: pip install langchain-openai")
        return

    # Initialize orchestrator
    orchestrator = MultiAgentMedicalOrchestrator(
        llm=llm,
        fhir_server_url="http://localhost:8080/fhir"
    )

    # Run demonstrations
    print("\n" + "="*70)
    print("MEDICAL AGENT ORCHESTRATION SYSTEM - COMPREHENSIVE DEMONSTRATION")
    print("="*70)

    # Show agent capabilities
    demonstrate_agent_capabilities(orchestrator)

    # Demonstrate individual agents
    demonstrate_task_decomposition(llm)
    demonstrate_ehr_queries(llm)
    demonstrate_medical_calculations(llm)
    demonstrate_external_tools(llm)

    # Run example tasks
    # example_1_simple_query(orchestrator)
    # example_2_compound_task(orchestrator)
    # example_3_calculation_task(orchestrator)

    print("\n" + "="*70)
    print("Demonstration complete!")
    print("="*70)


if __name__ == "__main__":
    main()
