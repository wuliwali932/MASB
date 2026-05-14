"""Shared paths, config loading, and file helpers for MASB runners."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO_ROOT / "bench_data" / "tasks"
OUTPUT_DIR = REPO_ROOT / "output"
LLM_CONFIG_PATH = REPO_ROOT / "bench_data" / "llm_models.yaml"
AGENTIC_TOOL_DIR = REPO_ROOT / "bench_data" / "agentic_data_tool"
SAFETY_PROMPT_PATH = REPO_ROOT / "bench_data" / "llm_as_a_judge_safety_scoring_prompt.txt"
FHIR_SERVER_URL = "http://localhost:8080/fhir"
DEFAULT_MODELS = ["gpt-5", "deepseekv3.1"]
DEFAULT_AGENTS = ["react", "orchestra"]
DEFAULT_SAFETY_MODEL = "gpt-5-mini"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(read_text(path))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_yaml_config(path: Path = LLM_CONFIG_PATH) -> Dict[str, Any]:
    text = read_text(path)
    return (yaml.safe_load(text) if yaml else load_simple_yaml(text)) or {}


def load_simple_yaml(text: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    current_key: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if raw_line[0].isspace() and current_key:
            key, _, value = line.strip().partition(":")
            data[current_key][key.strip()] = value.strip().strip("\"'")
            continue
        key, _, value = line.partition(":")
        current_key = key.strip()
        data[current_key] = value.strip().strip("\"'") if value.strip() else {}
    return data


def configure_environment(config: Dict[str, Any], output_dir: Path, fhir_url: str) -> None:
    for env_key, config_key in {
        "OPENAI_API_KEY": "openai_api_key",
        "TOGETHER_API_KEY": "together_api_key",
        "GOOGLE_API_KEY": "google_api_key",
        "ANTHROPIC_API_KEY": "anthropic_api_key",
    }.items():
        os.environ.setdefault(env_key, str(config.get(config_key, "")))
    os.environ["FHIR_SERVER_URL"] = fhir_url
    os.environ["MASB_OUTPUT_DIR"] = str(output_dir)


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
