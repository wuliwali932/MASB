"""Agent construction and execution helpers for sample task runs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

from run_agent_task.common import AGENTIC_TOOL_DIR, LLM_CONFIG_PATH, safe_name, read_text, write_json
from run_agent_task.agent_options import option_metadata, option_prompt
from run_agent_task.llm_clients import create_chat_llm
from run_agent_task.task_sampling import Task, task_metadata, task_prompt


def build_agent_system_prompt(thinking_mode: str = "disabled", memory_type: str = "complete") -> str:
    parts = [
        read_text(AGENTIC_TOOL_DIR / "agent_helper_tool_fhir_v1.txt"),
        "\nSMART on FHIR API specification:\n",
        read_text(AGENTIC_TOOL_DIR / "funcs_smartonfhir_server_v1.json"),
        "\nFHIR resource API specification:\n",
        read_text(AGENTIC_TOOL_DIR / "funcs_fhir_resources_server_v1.json"),
        "\nLocal tool specification:\n",
        read_text(AGENTIC_TOOL_DIR / "funcs_local_v1.json"),
        "\nPredefined users and login credentials:\n",
        read_text(AGENTIC_TOOL_DIR / "predefined_users.json"),
        "\nFHIR server: http://localhost:8080/fhir\n",
        "SMART on FHIR auth server: http://localhost:8000\n",
        "Use only the task instruction and task context as task-specific input.\n",
        option_prompt(thinking_mode, memory_type),
    ]
    return "\n".join(parts)


def create_react_agent(model_alias: str, thinking_mode: str = "disabled", memory_type: str = "complete"):
    from masb_agent.agent import AgentsManager, MASBAgent

    MASBAgent.load_config(str(LLM_CONFIG_PATH))
    return AgentsManager.create_agent(
        model_alias,
        build_agent_system_prompt(thinking_mode, memory_type),
        thinking_mode=thinking_mode,
        memory_type=memory_type,
    )


def create_orchestra_agent(
    model_alias: str,
    config: Dict[str, Any],
    output_dir: Path,
    fhir_url: str,
    thinking_mode: str = "disabled",
    memory_type: str = "complete",
):
    from masb_orchestra.multi_agent_orchestrator import MultiAgentMedicalOrchestrator

    llm, model_name = create_chat_llm(config, model_alias, thinking_mode=thinking_mode)
    return MultiAgentMedicalOrchestrator(
        llm=llm,
        fhir_server_url=fhir_url,
        output_dir=str(output_dir),
        model_name=model_name,
        thinking_mode=thinking_mode,
        memory_type=memory_type,
    )


def write_failure_log(output_dir: Path, agent_type: str, model_alias: str, task: Task, error: Exception) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = output_dir / f"{agent_type}_{safe_name(model_alias)}_{safe_name(task['id'])}_{timestamp}_error.json"
    write_json(
        path,
        {
            "agent_type": agent_type,
            "model_alias": model_alias,
            "metadata": task_metadata(task, model_alias, agent_type),
            "task_key": task.get("task_key"),
            "instruction": task["instruction"],
            "context": task["context"],
            "error": repr(error),
            "timestamp": datetime.now().isoformat(),
        },
    )
    return str(path)


def result_record(
    agent_type: str,
    model_alias: str,
    task: Task,
    status: str,
    thinking_mode: str = "disabled",
    memory_type: str = "complete",
    **extra: Any,
) -> Dict[str, Any]:
    return {
        "agent_type": agent_type,
        "model": model_alias,
        "task_id": task["id"],
        "task_key": task.get("task_key"),
        "category": task.get("category"),
        "attack_type": task.get("attack_type"),
        "source_file": task.get("source_file"),
        "task_index": task.get("task_index"),
        "status": status,
        **option_metadata(thinking_mode, memory_type),
        **extra,
    }


def run_react_tasks(
    model_alias: str,
    tasks: Sequence[Task],
    output_dir: Path,
    thinking_mode: str = "disabled",
    memory_type: str = "complete",
) -> List[Dict[str, Any]]:
    agent = create_react_agent(model_alias, thinking_mode, memory_type)
    results: List[Dict[str, Any]] = []
    for task in tasks:
        try:
            metadata = {**task_metadata(task, model_alias, "react"), **option_metadata(thinking_mode, memory_type)}
            final_answer = agent.run(task_prompt(task), metadata=metadata)
            log_path = agent.history[-1].get("log_path", "") if agent.history else ""
            results.append(
                result_record(
                    "react",
                    model_alias,
                    task,
                    "completed",
                    thinking_mode,
                    memory_type,
                    final_response=final_answer,
                    log_path=log_path,
                )
            )
        except Exception as exc:
            results.append(
                result_record(
                    "react",
                    model_alias,
                    task,
                    "failed",
                    thinking_mode,
                    memory_type,
                    error=repr(exc),
                    log_path=write_failure_log(output_dir, "react", model_alias, task, exc),
                )
            )
    return results


def run_orchestra_tasks(
    model_alias: str,
    tasks: Sequence[Task],
    config: Dict[str, Any],
    output_dir: Path,
    fhir_url: str,
    thinking_mode: str = "disabled",
    memory_type: str = "complete",
) -> List[Dict[str, Any]]:
    orchestrator = create_orchestra_agent(model_alias, config, output_dir, fhir_url, thinking_mode, memory_type)
    results: List[Dict[str, Any]] = []
    for task in tasks:
        try:
            metadata = {**task_metadata(task, model_alias, "orchestra"), **option_metadata(thinking_mode, memory_type)}
            result = orchestrator.execute(task["instruction"], task["context"], metadata=metadata)
            results.append(
                result_record(
                    "orchestra",
                    model_alias,
                    task,
                    result.get("status", "completed"),
                    thinking_mode,
                    memory_type,
                    final_response=result.get("final_response"),
                    log_path=result.get("log_path"),
                )
            )
        except Exception as exc:
            results.append(
                result_record(
                    "orchestra",
                    model_alias,
                    task,
                    "failed",
                    thinking_mode,
                    memory_type,
                    error=repr(exc),
                    log_path=write_failure_log(output_dir, "orchestra", model_alias, task, exc),
                )
            )
    return results
