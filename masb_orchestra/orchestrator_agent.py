"""
Orchestrator Agent - Task Decomposition and Coordination
Analyzes task instructions, decomposes tasks into subtasks, 
assigns subtasks to appropriate subagents, and merges responses.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import re
from datetime import datetime
from langchain_core.language_models import BaseLanguageModel


class TaskType(Enum):
    """Types of medical tasks."""
    QUERY = "query"  # Read operations
    CREATE = "create"  # Create operations
    UPDATE = "update"  # Update operations
    DELETE = "delete"  # Delete operations
    CALCULATE = "calculate"  # Medical calculations
    EXTERNAL_TOOL = "external_tool"  # External tool usage
    COMPOUND = "compound"  # Multiple operations


@dataclass
class SubTask:
    """Represents a subtask for a subagent."""
    task_id: str
    task_type: TaskType
    instruction: str
    context: Dict[str, Any]
    assigned_agent: str  # 'ehr_query', 'external_tool', 'medical_calc'
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[str] = None
    error: Optional[str] = None


@dataclass
class TaskDecomposition:
    """Task decomposition result."""
    original_task: str
    subtasks: List[SubTask]
    dependencies: Dict[str, List[str]]  # task_id -> list of dependent task_ids
    execution_order: List[str]  # topologically sorted task IDs


class OrchestratorAgent:
    """
    Orchestrator Agent that manages other subagents.
    - Analyzes task instructions
    - Decomposes tasks into subtasks
    - Assigns subtasks to appropriate subagents
    - Manages execution flow
    - Merges responses from subagents
    """

    def __init__(self, llm: BaseLanguageModel, system_guidance: str = ""):
        """
        Initialize orchestrator agent.

        Args:
            llm: Language model for task decomposition and reasoning
        """
        self.llm = llm
        self.system_guidance = system_guidance
        self.execution_history: List[Dict[str, Any]] = []
        self.subtasks: Dict[str, SubTask] = {}

    def analyze_task(self, instruction: str, context: str = "") -> Dict[str, Any]:
        """
        Analyze task instruction to understand its nature.

        Args:
            instruction: The medical task instruction
            context: Additional context about the task

        Returns:
            Dictionary with task analysis
        """
        analysis_prompt = f"""
        {self.system_guidance}

        Analyze the following medical task instruction and provide:
        1. Task type (QUERY, CREATE, UPDATE, DELETE, CALCULATE, EXTERNAL_TOOL, COMPOUND)
        2. Key medical entities involved (patient MRN, lab codes, medications, etc.)
        3. Required operations
        4. Whether it requires medical reasoning/calculations
        5. Any security or compliance considerations
        
        Instruction: {instruction}
        Context: {context}
        
        Provide response as JSON with keys: task_type, entities, operations, requires_calculation, security_notes
        """

        # Use LLM to analyze task
        response = self.llm.invoke(analysis_prompt)
        raw_response = getattr(response, "content", str(response))

        try:
            analysis = json.loads(raw_response)
        except (TypeError, json.JSONDecodeError):
            # Fallback analysis if JSON parsing fails
            analysis = {
                "task_type": self._infer_task_type(instruction),
                "entities": self._extract_entities(instruction),
                "operations": self._extract_operations(instruction),
                "requires_calculation": any(keyword in instruction.lower()
                                            for keyword in ["calculate", "average", "dosing", "check if"]),
                "security_notes": "Review for sensitive data access"
            }

        self.execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "action": "analyze_task",
            "prompt": analysis_prompt,
            "llm_output": raw_response,
            "analysis": analysis,
        })
        return analysis

    def decompose_task(self, instruction: str, context: str = "") -> TaskDecomposition:
        """
        Decompose a complex task into subtasks.

        Args:
            instruction: The medical task instruction
            context: Additional context about the task

        Returns:
            TaskDecomposition with subtasks and execution order
        """
        # Analyze task first
        analysis = self.analyze_task(instruction, context)
        task_type = analysis.get("task_type", "query")
        entities = analysis.get("entities", {})
        if not isinstance(entities, dict):
            entities = {"entities": entities}

        subtasks = []
        dependencies = {}
        lower_instruction = instruction.lower()
        action_compound = bool(
            re.search(r"\b(order|calculate|average|dose|dosing|create|update)\b", lower_instruction)
            or re.search(r"\brecord\b(?!\s+(is|was|missing))", lower_instruction)
        )

        if (task_type == "COMPOUND" and action_compound) or (
            "check" in lower_instruction and re.search(r"\border\b", lower_instruction)
        ):
            # Multi-step task: typically read, calculate, then write
            # Example: "Check magnesium level. If low, order replacement"

            # Step 1: Read lab value
            read_task = SubTask(
                task_id="subtask_1_read",
                task_type=TaskType.QUERY,
                instruction=f"Query patient lab value: {instruction}",
                context={"original_instruction": instruction, **entities},
                assigned_agent="ehr_query"
            )
            subtasks.append(read_task)

            # Step 2: Medical calculation/decision
            calc_task = SubTask(
                task_id="subtask_2_calculate",
                task_type=TaskType.CALCULATE,
                instruction=f"Analyze lab result and determine if action needed: {instruction}",
                context={"requires_result_from": "subtask_1_read", **entities},
                assigned_agent="medical_calc"
            )
            subtasks.append(calc_task)
            dependencies["subtask_2_calculate"] = ["subtask_1_read"]

            # Step 3: If needed, write action (order medication, create order, etc.)
            if re.search(r"\b(order|create|update)\b", lower_instruction) or re.search(
                r"\brecord\b(?!\s+(is|was|missing))", lower_instruction
            ):
                write_task = SubTask(
                    task_id="subtask_3_write",
                    task_type=TaskType.CREATE,
                    instruction=f"Execute action based on decision: {instruction}",
                    context={
                        "requires_result_from": "subtask_2_calculate", **entities},
                    assigned_agent="ehr_query"
                )
                subtasks.append(write_task)
                dependencies["subtask_3_write"] = ["subtask_2_calculate"]

        elif task_type in ("QUERY", "query", "COMPOUND"):
            # Simple read task
            subtask = SubTask(
                task_id="subtask_1_query",
                task_type=TaskType.QUERY,
                instruction=instruction,
                context={"original_instruction": instruction, **entities},
                assigned_agent="ehr_query"
            )
            subtasks.append(subtask)

        elif task_type == "CALCULATE" or task_type == "calculate":
            # Medical calculation task
            subtask = SubTask(
                task_id="subtask_1_calculate",
                task_type=TaskType.CALCULATE,
                instruction=instruction,
                context={"original_instruction": instruction, **entities},
                assigned_agent="medical_calc"
            )
            subtasks.append(subtask)

        elif task_type in ["CREATE", "UPDATE", "create", "update"]:
            # Write operation
            subtask = SubTask(
                task_id="subtask_1_write",
                task_type=TaskType.CREATE if task_type in [
                    "CREATE", "create"] else TaskType.UPDATE,
                instruction=instruction,
                context={"original_instruction": instruction, **entities},
                assigned_agent="ehr_query"
            )
            subtasks.append(subtask)

        else:
            # Default: single task
            subtask = SubTask(
                task_id="subtask_1_default",
                task_type=TaskType.QUERY,
                instruction=instruction,
                context={"original_instruction": instruction, **entities},
                assigned_agent="ehr_query"
            )
            subtasks.append(subtask)

        # Store subtasks
        for subtask in subtasks:
            self.subtasks[subtask.task_id] = subtask

        # Determine execution order (topological sort)
        execution_order = self._topological_sort(
            [t.task_id for t in subtasks],
            dependencies
        )

        return TaskDecomposition(
            original_task=instruction,
            subtasks=subtasks,
            dependencies=dependencies,
            execution_order=execution_order
        )

    def assign_subtasks(self, decomposition: TaskDecomposition) -> Dict[str, List[SubTask]]:
        """
        Assign subtasks to appropriate subagents based on task type.

        Args:
            decomposition: Task decomposition result

        Returns:
            Dictionary mapping agent names to their assigned subtasks
        """
        assignments = {
            "ehr_query": [],
            "external_tool": [],
            "medical_calc": []
        }

        for subtask in decomposition.subtasks:
            agent = subtask.assigned_agent
            if agent in assignments:
                assignments[agent].append(subtask)

        return assignments

    def merge_responses(
        self,
        subtask_results: Dict[str, Any],
        original_task: str = "",
        context: str = "",
    ) -> str:
        """
        Merge responses from all subagents into a final answer.

        Args:
            subtask_results: Dictionary mapping task IDs to their results

        Returns:
            Merged final response
        """
        merge_prompt = f"""
        {self.system_guidance}

        Merge the following subtask results into the final response for the medical task.

        Original task:
        {original_task}

        Context:
        {context}
        
        Subtask Results:
        {json.dumps(subtask_results, indent=2)}
        
        Return exactly the format requested by the original task.
        If the task asks for a list, return only a JSON-loadable list.
        If it asks for -1 on missing/mismatched records, return only -1.
        Do not add summaries, markdown, warnings, or recommendations unless explicitly requested.
        """

        response = self.llm.invoke(merge_prompt)
        merged_response = getattr(response, "content", str(response))

        # Log execution
        self.execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "action": "merge_responses",
            "prompt": merge_prompt,
            "llm_output": merged_response,
            "subtask_results": subtask_results,
            "final_response": merged_response
        })

        return merged_response

    def execute_task(self, instruction: str, context: str = "") -> Dict[str, Any]:
        """
        Execute a complete medical task through orchestration.

        Args:
            instruction: Medical task instruction
            context: Additional context

        Returns:
            Execution result with final answer and details
        """
        # Step 1: Decompose task
        decomposition = self.decompose_task(instruction, context)

        # Step 2: Assign subtasks
        assignments = self.assign_subtasks(decomposition)

        # Step 3: Return execution plan (actual execution happens in coordinated workflow)
        return {
            "status": "decomposed",
            "decomposition": {
                "original_task": decomposition.original_task,
                "num_subtasks": len(decomposition.subtasks),
                "subtasks": [
                    {
                        "id": st.task_id,
                        "type": st.task_type.value,
                        "instruction": st.instruction,
                        "assigned_agent": st.assigned_agent
                    }
                    for st in decomposition.subtasks
                ],
                "execution_order": decomposition.execution_order,
                "dependencies": decomposition.dependencies
            },
            "assignments": {
                agent: len(tasks) for agent, tasks in assignments.items()
            }
        }

    def _infer_task_type(self, instruction: str) -> str:
        """Infer task type from instruction text."""
        instruction_lower = instruction.lower()

        if "check if" in instruction_lower or (
            "check" in instruction_lower and "order" in instruction_lower
        ):
            return "COMPOUND"
        elif any(word in instruction_lower for word in ["order", "record", "create", "enter"]):
            return "CREATE"
        elif any(word in instruction_lower for word in ["update", "change", "modify"]):
            return "UPDATE"
        elif any(word in instruction_lower for word in ["calculate", "average", "recent", "last"]):
            return "CALCULATE"
        elif any(word in instruction_lower for word in ["what", "find", "get", "retrieve", "which"]):
            return "QUERY"
        else:
            return "QUERY"

    def _extract_entities(self, instruction: str) -> Dict[str, List[str]]:
        """Extract medical entities from instruction."""
        entities = {
            "patient_mrn": [],
            "lab_codes": [],
            "medications": [],
            "time_references": []
        }

        # Simple extraction - in production, use NER
        import re

        # Extract MRN (pattern: S followed by 7 digits)
        mrns = re.findall(r'S\d{7}', instruction)
        entities["patient_mrn"] = mrns

        # Extract common lab codes
        lab_codes = re.findall(r'\b[A-Z]{1,3}\b', instruction)
        entities["lab_codes"] = [code for code in lab_codes
                                 if code in ["MG", "K", "GLU", "A1C", "BP", "QT"]]
        lab_name_map = {
            "magnesium": "MG",
            "potassium": "K",
            "glucose": "GLU",
            "hemoglobin a1c": "A1C",
            "a1c": "A1C",
            "blood pressure": "BP",
        }
        lower_instruction = instruction.lower()
        for name, code in lab_name_map.items():
            if name in lower_instruction and code not in entities["lab_codes"]:
                entities["lab_codes"].append(code)

        # Extract time references
        time_refs = re.findall(
            r'\d+\s*(hour|day|week|month|year)s?', instruction)
        entities["time_references"] = time_refs

        return entities

    def _extract_operations(self, instruction: str) -> List[str]:
        """Extract operations from instruction."""
        operations = []
        instruction_lower = instruction.lower()

        if any(word in instruction_lower for word in ["query", "what", "find", "get", "retrieve"]):
            operations.append("read")
        if any(word in instruction_lower for word in ["order", "record", "create", "enter"]):
            operations.append("write")
        if any(word in instruction_lower for word in ["check if", "calculate", "average"]):
            operations.append("calculate")

        return operations

    def _topological_sort(self, task_ids: List[str],
                          dependencies: Dict[str, List[str]]) -> List[str]:
        """
        Topologically sort tasks based on dependencies.

        Args:
            task_ids: List of task IDs
            dependencies: Dictionary mapping task_id to list of dependent task_ids

        Returns:
            Sorted list of task IDs
        """
        # Build reverse dependency graph
        in_degree = {task_id: 0 for task_id in task_ids}
        graph = {task_id: [] for task_id in task_ids}

        for task_id, deps in dependencies.items():
            for dep in deps:
                if dep in graph:
                    graph[dep].append(task_id)
                    in_degree[task_id] += 1

        # Kahn's algorithm
        queue = [task_id for task_id in task_ids if in_degree[task_id] == 0]
        sorted_tasks = []

        while queue:
            current = queue.pop(0)
            sorted_tasks.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return sorted_tasks if len(sorted_tasks) == len(task_ids) else task_ids

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self.execution_history
