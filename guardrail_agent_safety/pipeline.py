"""Shared guardrail benchmark pipeline."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Sequence

from guardrail_agent_safety.guardrail import GuardrailConfig, GuardrailEvaluator
from run_agent_task.agent_runners import run_orchestra_tasks, run_react_tasks
from run_agent_task.common import OUTPUT_DIR, configure_environment, load_yaml_config
from run_agent_task.safety_scoring import score_results
from run_agent_task.task_sampling import Task, sample_attacked_tasks


GUARD_OUTPUT_DIR = OUTPUT_DIR / "guardrail_agent_safety"


def run_agents(
    tasks: Sequence[Task],
    models: Sequence[str],
    agents: Sequence[str],
    config: Dict[str, Any],
    output_dir: Path,
    fhir_url: str,
    thinking_mode: str = "disabled",
    memory_type: str = "complete",
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for model_alias in models:
        if "react" in agents:
            results.extend(run_react_tasks(model_alias, tasks, output_dir, thinking_mode, memory_type))
        if "orchestra" in agents:
            results.extend(run_orchestra_tasks(model_alias, tasks, config, output_dir, fhir_url, thinking_mode, memory_type))
    return results


def evaluate_guardrail(
    tasks: Sequence[Task],
    results: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    output_dir: Path,
    safety_model: str,
    embedding_model: str,
    threshold: float,
    contamination: float,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], GuardrailEvaluator]:
    scored = score_results(tasks, results, config, output_dir, safety_model)
    guard_config = GuardrailConfig(
        judge_model=safety_model,
        embedding_model=embedding_model,
        threshold=threshold,
        contamination=contamination,
    )
    evaluator = GuardrailEvaluator(config, output_dir, guard_config)
    evaluated = evaluator.evaluate(tasks, scored)
    return evaluated, evaluator.ablation_records(evaluated), evaluator


def run_guardrail_pipeline(args: Namespace) -> Path:
    config = load_yaml_config()
    configure_environment(config, args.output_dir, args.fhir_url)
    tasks = sample_attacked_tasks(args.tasks_dir, args.seed, args.task_count)
    results = run_agents(
        tasks,
        args.models,
        args.agents,
        config,
        args.output_dir,
        args.fhir_url,
        getattr(args, "thinking_mode", "disabled"),
        getattr(args, "memory_type", "complete"),
    )
    evaluated, records, evaluator = evaluate_guardrail(
        tasks,
        results,
        config,
        args.output_dir,
        args.safety_model,
        args.embedding_model,
        args.threshold,
        args.contamination,
    )
    return evaluator.write_manifest(tasks, evaluated, records)


def print_summary(manifest_path: Path, conditions: Sequence[str] | None = None) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = set(conditions or [])
    print("condition\tASR\tRR\tTSARR\tmore_safe")
    for row in manifest["summary"]["overall"]:
        if selected and row["condition"] not in selected:
            continue
        print(
            f"{row['condition']}\t{row['ASR']:.4f}\t{row['RR']:.4f}\t"
            f"{row['TSARR']:.4f}\t{row['more_safe_than_no_defense']}"
        )
