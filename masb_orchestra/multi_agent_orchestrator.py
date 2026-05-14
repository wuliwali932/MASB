"""
Multi-Agent Orchestration Workflow using LangGraph
Coordinates orchestrator, EHR query, external tool, and medical calculation agents.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import os
from pathlib import Path
import re

from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import StateGraph, END

from .orchestrator_agent import OrchestratorAgent, SubTask, TaskDecomposition
from .ehr_query_agent import EHRQueryAgent
from .external_tool_agent import ExternalToolAgent
from .medical_calculation_agent import MedicalCalculationAgent
from .auth_manager import get_auth_manager
from masb_agent.fhir_utils import summarize_fhir_resources
from run_agent_task.agent_options import normalize_memory_type, option_prompt, summarize_memory, thinking_enabled


@dataclass
class AgentState:
    """State object for multi-agent workflow."""

    # Input
    task_instruction: str = ""
    task_context: str = ""
    task_id: str = ""

    # Task decomposition
    task_decomposition: Optional[TaskDecomposition] = None
    subtasks: Dict[str, SubTask] = field(default_factory=dict)

    # Execution tracking
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    subtask_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    # Output
    final_response: str = ""
    execution_status: str = "pending"  # pending, in_progress, completed, failed
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class AgentType(Enum):
    """Types of agents in the workflow."""
    ORCHESTRATOR = "orchestrator"
    EHR_QUERY = "ehr_query"
    EXTERNAL_TOOL = "external_tool"
    MEDICAL_CALC = "medical_calc"


class MultiAgentMedicalOrchestrator:
    """
    Main orchestration engine coordinating all medical subagents.
    Uses LangGraph for workflow management and routing.
    """

    DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"

    def __init__(
        self,
        llm: BaseLanguageModel,
        fhir_server_url: str = "http://localhost:8080/fhir",
        output_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        thinking_mode: str = "disabled",
        memory_type: str = "complete",
    ):
        """
        Initialize multi-agent orchestrator.

        Args:
            llm: Language model
            fhir_server_url: FHIR server URL
        """
        self.llm = llm
        self.fhir_server_url = fhir_server_url
        self.model_name = model_name or getattr(llm, "model_name", None) or getattr(llm, "model", "unknown")
        self.output_dir = Path(output_dir) if output_dir else self.DEFAULT_OUTPUT_DIR
        self.thinking_mode = "enabled" if thinking_enabled(thinking_mode) else "disabled"
        self.memory_type = normalize_memory_type(memory_type)
        self.option_prompt = option_prompt(self.thinking_mode, self.memory_type)

        # Initialize subagents
        self.auth_manager = get_auth_manager(
            auth_service_url=os.getenv("MASB_AUTH_URL", "http://localhost:8000")
        )
        self.orchestrator = OrchestratorAgent(llm, system_guidance=self.option_prompt)
        self.ehr_query_agent = EHRQueryAgent(llm, fhir_server_url, self.auth_manager)
        self.external_tool_agent = ExternalToolAgent(llm)
        self.medical_calc_agent = MedicalCalculationAgent(llm)

        # Build workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """
        Build LangGraph workflow.

        Returns:
            Compiled StateGraph
        """
        workflow = StateGraph(AgentState)

        # Add nodes for each agent
        workflow.add_node("decompose_task", self._decompose_task_node)
        workflow.add_node("ehr_query_worker", self._ehr_query_worker_node)
        workflow.add_node("external_tool_worker",
                          self._external_tool_worker_node)
        workflow.add_node("medical_calc_worker",
                          self._medical_calc_worker_node)
        workflow.add_node("merge_responses", self._merge_responses_node)

        # Set entry point
        workflow.set_entry_point("decompose_task")

        # Add edges and routing logic
        workflow.add_edge("decompose_task", "ehr_query_worker")
        workflow.add_edge("ehr_query_worker", "merge_responses")
        workflow.add_edge("external_tool_worker", "merge_responses")
        workflow.add_edge("medical_calc_worker", "merge_responses")
        workflow.add_edge("merge_responses", END)

        return workflow

    def _decompose_task_node(self, state: AgentState) -> AgentState:
        """
        Orchestrator: Analyze and decompose task into subtasks.

        Args:
            state: Current agent state

        Returns:
            Updated state with decomposition
        """
        print(f"\n=== ORCHESTRATOR: Decomposing Task ===")
        print(f"Instruction: {state.task_instruction}")

        try:
            # Decompose task
            decomposition = self.orchestrator.decompose_task(
                state.task_instruction,
                state.task_context
            )

            state.task_decomposition = decomposition
            state.subtasks = {st.task_id: st for st in decomposition.subtasks}

            # Log execution
            state.execution_log.append({
                "timestamp": datetime.now().isoformat(),
                "agent": "orchestrator",
                "action": "decompose_task",
                "num_subtasks": len(decomposition.subtasks),
                "execution_order": decomposition.execution_order
            })

            print(
                f"✓ Task decomposed into {len(decomposition.subtasks)} subtasks")
            for st in decomposition.subtasks:
                print(f"  - {st.task_id}: {st.assigned_agent}")

            state.execution_status = "in_progress"

        except Exception as e:
            state.errors.append({
                "timestamp": datetime.now().isoformat(),
                "agent": "orchestrator",
                "error": str(e)
            })
            state.execution_status = "failed"
            print(f"✗ Orchestrator Error: {str(e)}")

        return state

    def _ehr_query_worker_node(self, state: AgentState) -> AgentState:
        """
        EHR Query Agent: Execute EHR query subtasks.

        Args:
            state: Current agent state

        Returns:
            Updated state with results
        """
        print(f"\n=== EHR QUERY AGENT: Executing Queries ===")

        if not state.task_decomposition:
            return state

        try:
            for subtask in state.task_decomposition.subtasks:
                if subtask.assigned_agent != "ehr_query":
                    continue

                print(f"\nExecuting subtask: {subtask.task_id}")
                print(f"Instruction: {subtask.instruction}")

                # Execute based on task type
                result = self._execute_ehr_query_subtask(subtask, state)
                state.subtask_results[subtask.task_id] = result

                state.execution_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "agent": "ehr_query",
                    "action": f"execute_{subtask.task_type.value}",
                    "subtask_id": subtask.task_id,
                    "result": result
                })

                print(f"✓ Result: {json.dumps(result, indent=2)}")

        except Exception as e:
            state.errors.append({
                "timestamp": datetime.now().isoformat(),
                "agent": "ehr_query",
                "error": str(e)
            })
            print(f"✗ EHR Query Error: {str(e)}")

        return state

    def _external_tool_worker_node(self, state: AgentState) -> AgentState:
        """
        External Tool Agent: Execute external tool subtasks.

        Args:
            state: Current agent state

        Returns:
            Updated state with results
        """
        print(f"\n=== EXTERNAL TOOL AGENT: Executing Tools ===")

        if not state.task_decomposition:
            return state

        try:
            for subtask in state.task_decomposition.subtasks:
                if subtask.assigned_agent != "external_tool":
                    continue

                print(f"\nExecuting subtask: {subtask.task_id}")
                print(f"Instruction: {subtask.instruction}")

                result = self.external_tool_agent.execute_tool_from_instruction(
                    subtask.instruction
                )

                state.subtask_results[subtask.task_id] = result

                state.execution_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "agent": "external_tool",
                    "action": "execute_tool",
                    "subtask_id": subtask.task_id,
                    "result": result
                })

                print(f"✓ Result: {json.dumps(result, indent=2)}")

        except Exception as e:
            state.errors.append({
                "timestamp": datetime.now().isoformat(),
                "agent": "external_tool",
                "error": str(e)
            })
            print(f"✗ External Tool Error: {str(e)}")

        return state

    def _medical_calc_worker_node(self, state: AgentState) -> AgentState:
        """
        Medical Calculation Agent: Execute medical calculation subtasks.

        Args:
            state: Current agent state

        Returns:
            Updated state with results
        """
        print(f"\n=== MEDICAL CALCULATION AGENT: Executing Calculations ===")

        if not state.task_decomposition:
            return state

        try:
            for subtask in state.task_decomposition.subtasks:
                if subtask.assigned_agent != "medical_calc":
                    continue

                print(f"\nExecuting subtask: {subtask.task_id}")
                print(f"Instruction: {subtask.instruction}")

                # Execute medical calculation
                result = self._execute_medical_calc_subtask(
                    subtask, self._memory_payload(state.subtask_results))

                state.subtask_results[subtask.task_id] = result

                state.execution_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "agent": "medical_calc",
                    "action": "execute_calculation",
                    "subtask_id": subtask.task_id,
                    "result": result
                })

                print(f"✓ Result: {json.dumps(result, indent=2)}")

        except Exception as e:
            state.errors.append({
                "timestamp": datetime.now().isoformat(),
                "agent": "medical_calc",
                "error": str(e)
            })
            print(f"✗ Medical Calculation Error: {str(e)}")

        return state

    def _merge_responses_node(self, state: AgentState) -> AgentState:
        """
        Merge responses from all subagents into final answer.

        Args:
            state: Current agent state

        Returns:
            Updated state with final response
        """
        print(f"\n=== ORCHESTRATOR: Merging Responses ===")

        try:
            final_response = self.orchestrator.merge_responses(
                self._memory_payload(state.subtask_results),
                original_task=state.task_instruction,
                context=self._task_context(state.task_context),
            )
            state.final_response = final_response

            state.execution_log.append({
                "timestamp": datetime.now().isoformat(),
                "agent": "orchestrator",
                "action": "merge_responses",
                "final_response": final_response
            })

            if state.errors:
                state.execution_status = "completed_with_errors"
            else:
                state.execution_status = "completed"

            print(f"✓ Final Response:\n{final_response}")

        except Exception as e:
            state.errors.append({
                "timestamp": datetime.now().isoformat(),
                "agent": "orchestrator",
                "error": f"Failed to merge responses: {str(e)}"
            })
            state.execution_status = "failed"
            print(f"✗ Merge Error: {str(e)}")

        state.updated_at = datetime.now().isoformat()
        return state

    def _execute_ehr_query_subtask(self, subtask: SubTask, state: AgentState) -> Dict[str, Any]:
        """Execute EHR query subtask using entities available in instruction/context."""
        text = f"{subtask.instruction}\n{state.task_context}"
        operations: List[Dict[str, Any]] = []
        resources: List[Dict[str, Any]] = []

        for resource_type, resource_id in self._extract_resource_refs(text):
            result = self.ehr_query_agent.read_resource(resource_type, resource_id)
            operations.append({
                "operation": "read",
                "resource_type": resource_type,
                "resource_id": resource_id,
                "result": result,
            })
            if result.get("success") and isinstance(result.get("resource"), dict):
                resources.append(result["resource"])

        for given_name, family_name in self._extract_patient_searches(text):
            result = self.ehr_query_agent.search_resources(
                "Patient",
                {"given": given_name, "family": family_name},
                limit=10,
            )
            operations.append({
                "operation": "search",
                "resource_type": "Patient",
                "search_params": {"given": given_name, "family": family_name},
                "result": result,
            })
            resources.extend(result.get("resources", []) if result.get("success") else [])

        if not operations:
            return {
                "task_id": subtask.task_id,
                "status": "no_supported_query_detected",
                "instruction": subtask.instruction,
                "message": "No direct FHIR resource id or patient name search could be parsed from instruction/context.",
            }

        return {
            "task_id": subtask.task_id,
            "status": "executed",
            "operations": operations,
            "summary": summarize_fhir_resources(resources),
            "resources": resources,
        }

    def _extract_resource_refs(self, text: str) -> List[tuple[str, str]]:
        """Find explicit FHIR resource references such as 'Procedure 639059'."""
        pattern = r"\b(Patient|Observation|MedicationRequest|Condition|Procedure)\s+([A-Za-z0-9_.-]+)\b"
        refs = []
        seen = set()
        for resource_type, resource_id in re.findall(pattern, text):
            key = (resource_type, resource_id)
            if key not in seen:
                seen.add(key)
                refs.append(key)
        return refs

    def _extract_patient_searches(self, text: str) -> List[tuple[str, str]]:
        """Find patient-name search phrases in benchmark instructions."""
        patterns = [
            r"patient search for\s+([A-Z][A-Za-z'-]+)\s+([A-Z][A-Za-z'-]+)",
            r"patient named\s+([A-Z][A-Za-z'-]+)\s+([A-Z][A-Za-z'-]+)",
            r"patient\s+([A-Z][A-Za-z'-]+)\s+([A-Z][A-Za-z'-]+)\s+with DOB",
        ]
        names = []
        seen = set()
        for pattern in patterns:
            for given, family in re.findall(pattern, text):
                key = (given, family)
                if key not in seen:
                    seen.add(key)
                    names.append(key)
        return names

    def _execute_medical_calc_subtask(self, subtask: SubTask,
                                      previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute medical calculation subtask."""
        # This would route to appropriate calculation methods
        # For now, return a placeholder
        return {
            "task_id": subtask.task_id,
            "status": "executed",
            "calculation": "Medical calculation placeholder"
        }

    def execute(
        self,
        instruction: str,
        context: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute medical task through orchestrated workflow.

        Args:
            instruction: Medical task instruction
            context: Additional context

        Returns:
            Execution result with final response
        """
        print(f"\n{'='*60}")
        print(f"MEDICAL TASK ORCHESTRATION")
        print(f"{'='*60}")
        self._reset_agent_histories()

        # Create initial state
        state = AgentState(
            task_instruction=instruction,
            task_context=context,
            task_id=f"task_{datetime.now().timestamp()}",
            metadata=metadata or {},
        )
        self._authenticate_for_task(instruction, context, state)

        # Execute workflow
        try:
            # For now, execute nodes sequentially
            # In production, would use compiled LangGraph
            state = self._decompose_task_node(state)
            state = self._ehr_query_worker_node(state)
            state = self._external_tool_worker_node(state)
            state = self._medical_calc_worker_node(state)
            state = self._merge_responses_node(state)

        except Exception as e:
            print(f"\n✗ Workflow Error: {str(e)}")
            state.execution_status = "failed"
            state.errors.append({
                "timestamp": datetime.now().isoformat(),
                "level": "workflow",
                "error": str(e)
            })

        # Return results
        result = {
            "agent_type": "orchestra",
            "task_id": state.task_id,
            "status": state.execution_status,
            "metadata": metadata or {},
            "model": self.model_name,
            "thinking_mode": self.thinking_mode,
            "memory_type": self.memory_type,
            "instruction": state.task_instruction,
            "context": state.task_context,
            "task_decomposition": state.task_decomposition,
            "final_response": state.final_response,
            "subtask_results": state.subtask_results,
            "execution_log": state.execution_log,
            "agent_histories": self._collect_agent_histories(),
            "errors": state.errors,
            "created_at": state.created_at,
            "completed_at": state.updated_at
        }
        result["log_path"] = self._write_run_log(result)
        return result

    def _task_context(self, context: str) -> str:
        return f"{context}\n{self.option_prompt}"

    def _memory_payload(self, value: Any) -> Any:
        return value if self.memory_type == "complete" else summarize_memory(value, 2400)

    def _collect_agent_histories(self) -> Dict[str, Any]:
        """Collect internal subagent histories for full run logging."""
        return {
            "orchestrator": self.orchestrator.execution_history,
            "ehr_query": self.ehr_query_agent.get_query_history(),
            "external_tool": self.external_tool_agent.get_tool_history(),
            "medical_calc": self.medical_calc_agent.get_calculation_history(),
        }

    def _reset_agent_histories(self) -> None:
        """Keep each output log scoped to one task run."""
        self.orchestrator.execution_history.clear()
        self.ehr_query_agent.query_history.clear()
        self.external_tool_agent.tool_history.clear()
        self.medical_calc_agent.calculation_history.clear()

    def _authenticate_for_task(self, instruction: str, context: str, state: AgentState) -> None:
        """Authenticate the EHR subagent with a role suggested by instruction/context."""
        text = f"{instruction}\n{context}".lower()
        if "administrator" in text or "admin" in text:
            username, password, role = "admin1", "a1secret", "administrator"
        elif "patient" in text and "physician" not in text and "doctor" not in text:
            username, password, role = "patient1", "p1secret", "patient"
        else:
            username, password, role = "doctor1", "d1secret", "physician"

        auth_record: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "agent": "ehr_query",
            "action": "auth_login",
            "username": username,
            "role": role,
        }
        try:
            ok = self.ehr_query_agent.authenticate(username, password)
            auth_record["success"] = ok
            user = self.ehr_query_agent.get_current_user()
            if user:
                auth_record["user"] = {k: user.get(k) for k in ("sub", "role", "username")}
        except Exception as e:
            auth_record["success"] = False
            auth_record["error"] = str(e)
        state.execution_log.append(auth_record)

    def _write_run_log(self, result: Dict[str, Any]) -> str:
        """Write a full per-run orchestration log to the shared output folder."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(self.model_name))
        task_id = result.get("metadata", {}).get("task_id") or result.get("task_id", "task")
        safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.output_dir / f"orchestra_{safe_model}_{safe_task}_{timestamp}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, default=str, indent=2, ensure_ascii=False)
        return str(path)

    def get_agent_capabilities(self) -> Dict[str, Any]:
        """Get capabilities of all subagents."""
        return {
            "orchestrator": {
                "capabilities": [
                    "Task analysis",
                    "Task decomposition",
                    "Subtask assignment",
                    "Response merging"
                ]
            },
            "ehr_query": {
                "capabilities": [
                    "Query patients by name/DOB",
                    "Query patients by MRN",
                    "Query lab values",
                    "Create observations",
                    "Create medication requests",
                    "Create service requests"
                ]
            },
            "external_tool": {
                "tools": self.external_tool_agent.list_tools()
            },
            "medical_calc": {
                "capabilities": [
                    "Calculate age",
                    "Calculate BMI",
                    "Calculate eGFR",
                    "Assess lab values",
                    "Calculate averages",
                    "Assess medication need"
                ]
            }
        }
