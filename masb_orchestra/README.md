# Medical Agent Orchestration System

A multi-agent LLM-based system for medical task automation using LangChain and LangGraph. The system decomposes complex medical tasks into subtasks and coordinates four specialized subagents to execute them.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│          Multi-Agent Medical Orchestrator (LangGraph)            │
└─────────────────────────────────────────────────────────────────┘
                                  │
                  ┌───────────────┼───────────────┐
                  │               │               │
        ┌─────────▼──────┐ ┌──────▼──────┐ ┌────▼──────────┐
        │  Orchestrator   │ │ EHR Query   │ │ Medical Calc │
        │     Agent       │ │   Agent     │ │    Agent     │
        ├─────────────────┤ ├─────────────┤ ├──────────────┤
        │ • Task analysis │ │ • Read ops  │ │ • Age calc   │
        │ • Decompose     │ │ • Write ops │ │ • BMI calc   │
        │ • Assign tasks  │ │ • FHIR Spec │ │ • Lab assess │
        │ • Merge results │ │ • CRUD all  │ │ • Dosing     │
        └─────────────────┘ └─────────────┘ └──────────────┘
                  │               │               │
                  └───────────────┼───────────────┘
                                  │
                        ┌─────────▼──────────┐
                        │ External Tool      │
                        │   Agent            │
                        ├────────────────────┤
                        │ • Lab reference    │
                        │ • Drug interact    │
                        │ • Dosing calc      │
                        │ • Notifications    │
                        └────────────────────┘
```

## Four Subagents

### 1. **Orchestrator Agent** (`orchestrator_agent.py`)

Manages task decomposition and coordination.

**Capabilities:**

- Analyzes task instructions to understand intent
- Decomposes complex tasks into subtasks
- Assigns subtasks to appropriate specialized agents
- Manages execution flow and dependencies
- Merges responses from subagents into final answer

**Key Classes:**

- `OrchestratorAgent`: Main orchestration logic
- `TaskDecomposition`: Task structure with subtasks and dependencies
- `SubTask`: Individual task assignments
- `TaskType`: Enum of task types (QUERY, CREATE, UPDATE, CALCULATE, COMPOUND, etc.)

**Example:**

```python
orchestrator = OrchestratorAgent(llm)

# Decompose compound task
decomposition = orchestrator.decompose_task(
    "Check magnesium level. If low, order replacement."
)

# Results in:
# - Subtask 1: Query lab value (assigned to ehr_query)
# - Subtask 2: Assess result (assigned to medical_calc)
# - Subtask 3: Create medication order (assigned to ehr_query)
```

---

### 2. **EHR Query Agent** (`ehr_query_agent.py`)

Handles CRUD operations on FHIR medical data.

**Capabilities:**

- Query patients by name/DOB
- Query patients by MRN
- Query lab values and observations
- Query medication history
- Create observations (record lab values)
- Create medication requests (orders)
- Create service requests (referrals)
- Full audit logging

**Supported FHIR Resources:**

- Patient
- Observation (lab values, vital signs)
- MedicationRequest (medication orders)
- ServiceRequest (referrals, procedures)
- Condition, Medication, etc.

**Example:**

```python
ehr_agent = EHRQueryAgent(llm, fhir_server_url="http://localhost:8082/fhir")

# Query patient
patient = ehr_agent.query_patient_by_mrn("S6534835")

# Query lab value
lab = ehr_agent.query_lab_value("S6315806", "MG", hours_back=24)

# Record lab value
obs = ehr_agent.create_observation("S6315806", "MG", 1.8, "mg/dL")

# Order medication
med_req = ehr_agent.create_medication_request(
    "S6315806", 
    "0338-1715-40",  # NDC
    "2g IV over 2 hours",
    "Magnesium replacement for deficiency"
)
```

---

### 3. **Medical Calculation Agent** (`medical_calculation_agent.py`)

Performs medical calculations and clinical decision support.

**Capabilities:**

- Calculate age from DOB
- Calculate BMI
- Calculate eGFR (kidney function)
- Assess lab values against reference ranges
- Assess medication needs and dosing
- Calculate averages of lab values
- Clinical summaries and recommendations

**Reference Ranges (Built-in):**

- **Magnesium (MG)**: 1.7-2.2 mg/dL
  - Mild deficiency: 1.5-1.9
  - Moderate deficiency: 1.0-1.4
  - Severe deficiency: <1.0
- **Potassium (K)**: 3.5-5.0 mEq/L
- **Glucose (GLU)**: 70-100 mg/dL (fasting)
- **HbA1C (A1C)**: <5.7%

**Example:**

```python
calc_agent = MedicalCalculationAgent(llm)

# Calculate age
age_result = calc_agent.calculate_age("1963-01-29")  # age_years: 61

# Assess lab value
assessment = calc_agent.assess_lab_value(1.2, "MG")
# Returns: status='moderate_deficiency', severity='HIGH'
# Recommendation: "Order replacement IV magnesium - 2g over 2 hours"

# Get medication need
med_need = calc_agent.assess_medication_need(1.2, "MG", patient_weight=75)
# Returns: needs_medication=True, recommended_dose="2g IV over 2 hours"

# Calculate average
avg = calc_agent.calculate_average([1.8, 1.9, 2.0])  # average: 1.9
```

---

### 4. **External Tool Agent** (`external_tool_agent.py`)

Integrates with external medical tools and APIs.

**Built-in Tools:**

1. **Lab Reference Lookup**: Reference ranges for lab codes
2. **Dosing Calculator**: Weight-based dosing with renal adjustment
3. **Drug Interaction Checker**: Check for drug-drug interactions
4. **Allergy Checker**: Check drug-allergy conflicts
5. **Notification Sender**: Send notifications to providers

**Extensibility:**

- Register custom tools dynamically
- Tool catalog with parameter definitions
- Category-based organization

**Example:**

```python
tool_agent = ExternalToolAgent(llm)

# List available tools
tools = tool_agent.list_tools()

# Execute specific tool
result = tool_agent.execute_tool(
    "dosing_calculator",
    medication="magnesium",
    patient_weight=75,
    renal_function="normal"
)
# Returns: calculated_dose=2.25 grams

# Check drug interactions
interactions = tool_agent.execute_tool(
    "drug_interaction_checker",
    medications=["magnesium", "potassium"]
)

# Register custom tool
tool_agent.register_tool(
    name="custom_calculator",
    category=ToolCategory.REFERENCE,
    description="Custom medical calculation",
    function=my_custom_function,
    parameters=[...]
)
```

---

## Multi-Agent Workflow

The system uses LangGraph to coordinate agents:

```
1. DECOMPOSE (Orchestrator)
   - Analyze task
   - Identify subtasks
   - Assign to agents
   
2. EXECUTE (Specialized Agents)
   - EHR Query Agent: Database operations
   - Medical Calc Agent: Clinical reasoning
   - External Tool Agent: External APIs
   
3. MERGE (Orchestrator)
   - Collect results
   - Synthesize response
   - Return final answer
```

### Example Task Flow

**Input Task:**

```
"Check patient S6315806's magnesium level. 
If low, order replacement IV magnesium."
```

**Decomposition:**

```
Subtask 1: Query magnesium level
  - Agent: ehr_query
  - Type: QUERY
  - Action: query_lab_value(mrn, "MG")

Subtask 2: Assess magnesium level
  - Agent: medical_calc
  - Type: CALCULATE
  - Action: assess_lab_value(result.value, "MG")
  - Dependency: Requires Subtask 1 result

Subtask 3: Order medication if low
  - Agent: ehr_query
  - Type: CREATE
  - Action: create_medication_request(...)
  - Dependency: Requires Subtask 2 assessment
```

**Execution:**

- Subtask 1 executes → Returns: value=1.2 mg/dL
- Subtask 2 executes → Returns: status="moderate_deficiency", dose="2g over 2 hours"
- Subtask 3 executes → Returns: medication_request_id created

**Merge:**

```
"Patient S6315806 has a serum magnesium level of 1.2 mg/dL,
indicating moderate deficiency. IV magnesium replacement of 2g 
over 2 hours has been ordered (Medication Request ID: MR123456)."
```

---

## Usage Guide

### Installation

```bash
pip install langchain langchain-core langgraph
pip install langchain-openai  # For OpenAI LLM
pip install requests           # For FHIR API calls
```

### Basic Usage

```python
from langchain_openai import ChatOpenAI
from multi_agent_orchestrator import MultiAgentMedicalOrchestrator

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create orchestrator
orchestrator = MultiAgentMedicalOrchestrator(
    llm=llm,
    fhir_server_url="http://localhost:8082/fhir"
)

# Execute medical task
result = orchestrator.execute(
    instruction="What's the MRN of patient Peter Stafford with DOB 1932-12-29?",
    context="You are the attending physician."
)

print(result['final_response'])
print(f"Status: {result['status']}")
print(f"Execution Time: {result['completed_at']}")
```

### Advanced Usage

```python
# Get agent capabilities
capabilities = orchestrator.get_agent_capabilities()

# Access individual agents
ehr_agent = orchestrator.ehr_query_agent
calc_agent = orchestrator.medical_calc_agent
tool_agent = orchestrator.external_tool_agent

# Direct agent calls
lab_value = ehr_agent.query_lab_value("S6315806", "MG")
assessment = calc_agent.assess_lab_value(lab_value['value'], "MG")

# Access execution history
history = orchestrator.ehr_query_agent.get_query_history()
calc_history = calc_agent.get_calculation_history()
```

---

## File Structure

```
masb_orchestra/
├── orchestrator_agent.py          # Task orchestration
├── ehr_query_agent.py             # FHIR CRUD operations
├── external_tool_agent.py         # External tools integration
├── medical_calculation_agent.py   # Medical calculations
├── multi_agent_orchestrator.py    # Main workflow engine (LangGraph)
├── example_usage.py               # Comprehensive examples
├── README.md                      # This file
└── __init__.py
```

---

## Supported Medical Tasks

### 1. Patient Queries

```
- Find MRN by patient name and DOB
- Get patient demographics and age
- Query patient medical history
```

### 2. Lab Value Queries

```
- Get most recent lab value
- Get lab value history (time range)
- Calculate averages
- Assess against reference ranges
```

### 3. Medication Management

```
- Query medication history
- Order medications with dosing
- Check drug-drug interactions
- Check drug-allergy conflicts
```

### 4. Clinical Decision Support

```
- Assess deficiency levels
- Recommend dosing
- Calculate kidney function (eGFR)
- Calculate BMI
```

### 5. Referral Management

```
- Create service requests
- Order procedures
- Create referrals with detailed notes
```

---

## Error Handling

The system implements comprehensive error handling:

1. **Task Decomposition Errors**: Falls back to single-task execution
2. **EHR Query Errors**: Logged with context, returns error message
3. **Calculation Errors**: Returns calculation failure, does not block workflow
4. **Tool Execution Errors**: Graceful failure with attempt logging
5. **Workflow Errors**: Execution status marked as "failed" or "completed_with_errors"

All errors are tracked in the execution log for audit and debugging.

---

## Security Considerations

1. **Authentication**: FHIR server authentication can be configured per deployment
2. **Audit Trail**: All operations logged with timestamp, agent, and parameters
3. **Data Validation**: Input validation on all patient IDs and medical codes
4. **Medication Safety**: Built-in reference checks for dosing and interactions
5. **HIPAA Compliance**: Ready for HIPAA-compliant deployments

---

## Extension Points

### Add New Tool to External Tool Agent

```python
def my_custom_tool(param1: str, param2: float) -> Dict:
    """Custom medical tool."""
    # Implementation
    return {"result": "..."}

tool_agent.register_tool(
    name="my_custom_tool",
    category=ToolCategory.REFERENCE,
    description="My custom tool description",
    function=my_custom_tool,
    parameters=[
        {"name": "param1", "type": "string"},
        {"name": "param2", "type": "float"}
    ]
)
```

### Add New Medical Calculation

```python
@tool
def calculate_medication_clearance(self, age: int, weight: float) -> Dict:
    """Calculate medication clearance."""
    clearance = weight * 0.8 - (age * 0.1)
    return {"clearance": max(0, clearance)}

# Add to MedicalCalculationAgent
```

### Add New FHIR Operation

```python
@tool
def create_diagnosis(self, patient_mrn: str, diagnosis_code: str) -> Dict:
    """Create diagnosis condition."""
    # Implementation
    pass

# Add to EHRQueryAgent
```

---

## Performance Notes

- **Task Decomposition**: ~500ms with LLM
- **EHR Queries**: 100-500ms depending on FHIR server
- **Medical Calculations**: <10ms (local)
- **External Tools**: 100-2000ms depending on API
- **Full Workflow**: 1-5 seconds for typical compound task

---

## Testing

Run examples to test the system:

```bash
python example_usage.py
```

This will demonstrate:

1. Agent capabilities
2. Task decomposition
3. Individual agent operations
4. Example medical tasks
5. Multi-agent workflow execution

---

## References

- FHIR Standard: <https://www.hl7.org/fhir/>
- LangChain: <https://python.langchain.com/>
- LangGraph: <https://langchain-ai.github.io/langgraph/>
- Medical Coding: <https://www.snomed.org/>

---
