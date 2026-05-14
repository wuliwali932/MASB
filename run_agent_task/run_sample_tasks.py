#!/usr/bin/env python3
"""Run one adjacent attack/clean MASB task pair and optionally score safety."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_agent_task.agent_runners import run_orchestra_tasks, run_react_tasks
from run_agent_task.common import (
    DEFAULT_AGENTS,
    DEFAULT_MODELS,
    DEFAULT_SAFETY_MODEL,
    FHIR_SERVER_URL,
    OUTPUT_DIR,
    TASKS_DIR,
    configure_environment,
    load_yaml_config,
    write_json,
)
from run_agent_task.safety_scoring import score_results
from run_agent_task.task_sampling import adjacent_attack_pairs, manifest_task, sample_task_pair


def write_manifest(
    output_dir: Path,
    selected_tasks: Sequence[Dict[str, Any]],
    results: Sequence[Dict[str, Any]],
    safety_scoring: bool,
    safety_model: str,
) -> Path:
    path = output_dir / f"sample_pair_run_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    write_json(
        path,
        {
            "created_at": datetime.now().isoformat(),
            "selection": "one adjacent pair: attack:true followed immediately by attack:false",
            "safety_scoring": {"enabled": safety_scoring, "judge_model": safety_model if safety_scoring else None},
            "selected_tasks": [manifest_task(task) for task in selected_tasks],
            "results": list(results),
        },
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one adjacent MASB attack/clean task pair.")
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--tasks-dir", type=Path, default=TASKS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--fhir-url", default=FHIR_SERVER_URL)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--agents", nargs="+", choices=DEFAULT_AGENTS, default=DEFAULT_AGENTS)
    parser.add_argument("--safety-model", default=DEFAULT_SAFETY_MODEL)
    parser.add_argument("--no-safety-scoring", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml_config()
    configure_environment(config, args.output_dir, args.fhir_url)
    selected_tasks = sample_task_pair(args.tasks_dir, args.seed)

    results: List[Dict[str, Any]] = []
    for model_alias in args.models:
        if "react" in args.agents:
            results.extend(run_react_tasks(model_alias, selected_tasks, args.output_dir))
        if "orchestra" in args.agents:
            results.extend(run_orchestra_tasks(model_alias, selected_tasks, config, args.output_dir, args.fhir_url))

    safety_enabled = not args.no_safety_scoring
    if safety_enabled:
        results = score_results(selected_tasks, results, config, args.output_dir, args.safety_model)

    manifest_path = write_manifest(args.output_dir, selected_tasks, results, safety_enabled, args.safety_model)
    print(f"Selected pair: {selected_tasks[0]['id']} -> {selected_tasks[1]['id']}")
    print(f"Wrote run manifest: {manifest_path}")
    print(f"Per-run full logs are in: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
