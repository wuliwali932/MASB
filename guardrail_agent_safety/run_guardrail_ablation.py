#!/usr/bin/env python3
"""Run guardrail defense ablations on randomly sampled attacked MASB tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from guardrail_agent_safety.pipeline import GUARD_OUTPUT_DIR, print_summary, run_guardrail_pipeline
from run_agent_task.agent_options import MEMORY_TYPES, THINKING_MODES
from run_agent_task.common import (
    DEFAULT_AGENTS,
    DEFAULT_MODELS,
    DEFAULT_SAFETY_MODEL,
    FHIR_SERVER_URL,
    TASKS_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate guardrail phase ablations on attacked MASB tasks.")
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--task-count", type=int, default=10)
    parser.add_argument("--tasks-dir", type=Path, default=TASKS_DIR)
    parser.add_argument("--output-dir", type=Path, default=GUARD_OUTPUT_DIR)
    parser.add_argument("--fhir-url", default=FHIR_SERVER_URL)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--agents", nargs="+",
                        choices=DEFAULT_AGENTS, default=DEFAULT_AGENTS)
    parser.add_argument("--safety-model", default=DEFAULT_SAFETY_MODEL)
    parser.add_argument("--embedding-model", default="text-embedding-3-large")
    parser.add_argument("--thinking-mode", choices=THINKING_MODES, default="disabled")
    parser.add_argument("--memory-type", choices=MEMORY_TYPES, default="complete")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--contamination", type=float, default=0.15)
    return parser.parse_args()


def main() -> int:
    manifest_path = run_guardrail_pipeline(parse_args())
    print(f"Wrote guardrail ablation manifest: {manifest_path}")
    print_summary(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
