"""LLM model resolution helpers shared by runners and safety scoring."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from run_agent_task.agent_options import model_alias_for_options


MODEL_CONFIG_KEYS = {
    "deepseekv3": "deepseekv3.1",
    "gemini3-pro": "gemini-3-pro-preview",
    "gemini-3-pro": "gemini-3-pro-preview",
    "claude-opus-4.5": "claude-opus-4.1",
    "qwen3-next": "qwen3-next-80b-a3b-instruct",
    "qwen3-next-81b": "qwen3-next-80b-a3b-instruct",
    "qwen3-next-80b": "qwen3-next-80b-a3b-instruct",
    "kimi-k2": "kimi-k2-instruct",
    "llama3.1-8b": "llama3.1-8b",
    "llama-3.1-8b": "llama3.1-8b",
    "llama-3.1-70b": "llama-3.1-70B",
    "llama-3.1-70.6b": "llama-3.1-70B",
    "llama3.1-70b": "llama-3.1-70B",
    "llama3.1-70.6b": "llama-3.1-70B",
    "llama-guard-4-12b": "Llama-Guard-4-12B",
    "llama-guard-3-8b": "llama-guard-3-8b",
    "llama-3.3": "Llama-3.3-70B-Instruct-Turbo",
    "qwen3": "qwen3-next-80b-a3b-instruct",
    "gemma3n": "gemma3n-e4b-instruct",
    "mistral": "Mixtral-8x22B-Instruct",
    "claude-sonnet-4.5": "claude-sonnet4.5",
}


def resolve_model(config: Dict[str, Any], model_alias: str, thinking_mode: str = "disabled") -> Tuple[str, str, str]:
    """Return provider, concrete model name, and API key for a model alias."""
    resolved_alias = model_alias_for_options(model_alias, thinking_mode)
    config_key = MODEL_CONFIG_KEYS.get(resolved_alias, resolved_alias)
    builtin_models = {
        "llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "llama-guard-3-8b": "meta-llama/Llama-Guard-3-8B",
    }
    model_name = config.get(config_key, {}).get("model", builtin_models.get(config_key, resolved_alias))
    provider = infer_provider(model_alias, model_name)
    api_key = {
        "openai": "openai_api_key",
        "google": "google_api_key",
        "anthropic": "anthropic_api_key",
        "together": "together_api_key",
    }[provider]
    return provider, model_name, str(config.get(api_key, ""))


def infer_provider(model_alias: str, model_name: str) -> str:
    text = f"{model_alias} {model_name}".lower()
    if "gemini" in text:
        return "google"
    if "claude" in text:
        return "anthropic"
    if str(model_name).startswith(("deepseek-ai/", "meta-llama/", "Qwen/", "google/", "mistralai/", "moonshotai/")):
        return "together"
    return "openai"


def chat_openai_kwargs(provider: str, model_name: str, api_key: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"model": model_name, "api_key": api_key}
    if provider == "together":
        kwargs["base_url"] = "https://api.together.xyz/v1"
    if not str(model_name).startswith(("gpt-5", "o1", "o3", "o4")):
        kwargs["temperature"] = 0
    return kwargs


def create_chat_llm(config: Dict[str, Any], model_alias: str, thinking_mode: str = "disabled"):
    provider, model_name, api_key = resolve_model(config, model_alias, thinking_mode)
    if provider in {"openai", "together"}:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(**chat_openai_kwargs(provider, model_name, api_key)), model_name
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0), model_name
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name, api_key=api_key, temperature=0, max_tokens=1200), model_name
    raise ValueError(f"Unsupported provider for {model_alias}: {provider}")
