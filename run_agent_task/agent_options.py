"""Shared agent option helpers for architecture comparisons."""

from __future__ import annotations

import json
from typing import Any, Dict


THINKING_MODES = ("disabled", "enabled")
MEMORY_TYPES = ("complete", "summarized")
DEFAULT_COMPARISON_MODELS = ["gpt-5", "gemini3-pro", "claude-opus-4.5", "qwen3-next", "kimi-k2"]
DEFAULT_GUARDRAIL_COMPARISON_MODELS = ["gpt-5", "qwen3-next", "llama3.1-8b", "llama-guard-3-8b"]
DEFAULT_THINKING_PAIR_MODELS = ["gpt-5", "gemini3-pro", "claude-opus-4.5", "qwen3-next", "kimi-k2"]
DEFAULT_GUARD_BACKBONE_MODELS = ["Llama-Guard-4-12B", "llama-3.1-70B", "llama3.1-8b", "llama-guard-3-8b"]
MODEL_DISPLAY_NAMES = {
    "Llama-Guard-4-12B": "Llama-Guard-4 (12B)",
    "llama-3.1-70B": "Llama-3.1 (70.6B)",
    "Llama-3.1-70B-Instruct-Turbo": "Llama-3.1 (70.6B)",
    "llama3.1-8b": "Meta Llama 3.1 (8B)",
    "llama-guard-3-8b": "Meta Llama Guard 3 (8B)",
}


def thinking_enabled(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value not in THINKING_MODES:
        raise ValueError(f"thinking_mode must be one of {THINKING_MODES}, got {value!r}")
    return value == "enabled"


def normalize_memory_type(value: str) -> str:
    if value not in MEMORY_TYPES:
        raise ValueError(f"memory_type must be one of {MEMORY_TYPES}, got {value!r}")
    return value


def option_prompt(thinking_mode: str, memory_type: str) -> str:
    thinking = (
        "Thinking mode is enabled. Deliberate internally before each tool or final answer, "
        "but do not expose hidden chain-of-thought; output only concise task reasoning."
        if thinking_enabled(thinking_mode)
        else "Thinking mode is disabled. Use direct, concise reasoning and avoid step-by-step deliberation."
    )
    memory = (
        "Use complete prior observations when deciding the next action."
        if memory_type == "complete"
        else "Use summarized prior observations: keep only task-relevant facts, tool names, errors, and final values."
    )
    return f"\nAgent option settings:\n- {thinking}\n- {memory}\n"


def option_metadata(thinking_mode: str, memory_type: str) -> Dict[str, str]:
    return {"thinking_mode": thinking_mode, "memory_type": memory_type}


def model_alias_for_options(model_alias: str, thinking_mode: str) -> str:
    enabled = thinking_enabled(thinking_mode)
    normalized = model_alias.strip()
    key = normalized.lower()
    aliases = {
        "gemini3-pro": "gemini-3-pro-preview",
        "gemini-3-pro": "gemini-3-pro-preview",
        "claude-opus-4.5": "claude-opus-4.1",
        "qwen3-next": "qwen3-next-80b-a3b-thinking" if enabled else "qwen3-next-80b-a3b-instruct",
        "qwen3-next-81b": "qwen3-next-80b-a3b-thinking" if enabled else "qwen3-next-80b-a3b-instruct",
        "qwen3-next-80b": "qwen3-next-80b-a3b-thinking" if enabled else "qwen3-next-80b-a3b-instruct",
        "kimi-k2": "kimi-k2-thinking" if enabled else "kimi-k2-instruct",
        "llama3.1(8b)": "llama3.1-8b",
        "llama3.1-8b": "llama3.1-8b",
        "llama-3.1-8b": "llama3.1-8b",
        "meta llama 3.1(8b)": "llama3.1-8b",
        "meta llama 3.1 (8b)": "llama3.1-8b",
        "meta-llama-3.1-8b": "llama3.1-8b",
        "llama3.1-8b-instruct": "llama3.1-8b",
        "llama-3.1(70.6b)": "llama-3.1-70B",
        "llama-3.1 (70.6b)": "llama-3.1-70B",
        "llama3.1-70.6b": "llama-3.1-70B",
        "llama-3.1-70.6b": "llama-3.1-70B",
        "llama3.1-70b": "llama-3.1-70B",
        "llama-3.1-70b": "llama-3.1-70B",
        "llama-guard-4(12b)": "Llama-Guard-4-12B",
        "llama-guard-4 (12b)": "Llama-Guard-4-12B",
        "llama guard 4(12b)": "Llama-Guard-4-12B",
        "llama guard 4 (12b)": "Llama-Guard-4-12B",
        "llama-guard-4-12b": "Llama-Guard-4-12B",
        "llama guard 3(8b)": "llama-guard-3-8b",
        "llama guard 3 (8b)": "llama-guard-3-8b",
        "meta llama guard 3(8b)": "llama-guard-3-8b",
        "meta llama guard 3 (8b)": "llama-guard-3-8b",
        "meta-llama-guard-3-8b": "llama-guard-3-8b",
        "llama-guard-3-8b": "llama-guard-3-8b",
        "llama-guard3-8b": "llama-guard-3-8b",
    }
    return aliases.get(key, normalized)


def model_display_name(model_alias: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_alias_for_options(model_alias, "disabled"), model_alias)


def summarize_memory(value: Any, limit: int = 1400) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + "...[summarized]"
