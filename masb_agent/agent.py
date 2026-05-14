"""
Single-agent REACT implementation for MASB.

This module intentionally uses one agent architecture for EHR tasks:
Think -> Act(tool call) -> Observe(tool response) -> repeat -> Final Answer.

The tool layer is compatible with the five benchmark FHIR resource structures
used by masb_orchestra and bench_data/FHIR_resource_examples.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None

try:
    from .agent_tools import get_tool_descriptions, get_tool_functions, set_fhir_server_url
except ImportError:
    from agent_tools import get_tool_descriptions, get_tool_functions, set_fhir_server_url

try:
    from run_agent_task.agent_options import model_alias_for_options, normalize_memory_type, option_prompt, summarize_memory, thinking_enabled
except ImportError:
    def model_alias_for_options(model_alias: str, thinking_mode: str) -> str:
        return model_alias

    def normalize_memory_type(value: str) -> str:
        return value

    def option_prompt(thinking_mode: str, memory_type: str) -> str:
        return ""

    def summarize_memory(value: Any, limit: int = 1400) -> str:
        text = json.dumps(value, ensure_ascii=False, default=str)
        return text if len(text) <= limit else text[:limit] + "...[summarized]"

    def thinking_enabled(value: str | bool) -> bool:
        return bool(value)


class MASBAgent:
    """Base class for a single REACT medical agent."""

    _llm_config: Dict[str, Any] = {}
    DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"

    def __init__(
        self,
        model_name: str,
        sys_prompt: Optional[str] = None,
        config_file: str = "llm_models.yaml",
        max_steps: int = 8,
        output_dir: Optional[str] = None,
        thinking_mode: str = "disabled",
        memory_type: str = "complete",
        **_: Any,
    ) -> None:
        if not MASBAgent._llm_config:
            MASBAgent.load_config(config_file)
        self.model_name = model_name
        self.sys_prompt = sys_prompt or "You are a helpful medical EHR agent."
        self.max_steps = max_steps
        self.thinking_mode = "enabled" if thinking_enabled(thinking_mode) else "disabled"
        self.memory_type = normalize_memory_type(memory_type)
        self.option_prompt = option_prompt(self.thinking_mode, self.memory_type)
        self.output_dir = Path(output_dir) if output_dir else Path(
            os.getenv("MASB_OUTPUT_DIR", str(self.DEFAULT_OUTPUT_DIR))
        )
        self.tools: Dict[str, Callable[..., Dict[str, Any]]] = get_tool_functions()
        self.tool_descriptions = get_tool_descriptions()
        self.history: List[Dict[str, Any]] = []

        fhir_url = os.getenv("FHIR_SERVER_URL")
        if fhir_url:
            set_fhir_server_url(fhir_url)

    @classmethod
    def load_config(cls, config_file: str = "llm_models.yaml") -> Dict[str, Any]:
        config_path = Path(__file__).resolve().parent / config_file
        if not config_path.exists():
            config_path = Path(__file__).resolve().parents[1] / "bench_data" / config_file
        if not config_path.exists():
            cls._llm_config = {}
            return cls._llm_config
        with open(config_path, "r", encoding="utf-8") as f:
            if yaml is not None:
                cls._llm_config = yaml.safe_load(f) or {}
            else:
                cls._llm_config = _load_simple_yaml(f.read())
        return cls._llm_config

    def inference(self, request: str) -> str:
        """Benchmark-compatible entrypoint."""
        return self.run(request)

    def run(
        self,
        user_request: str,
        max_steps: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Run the Think -> Act -> Observe REACT loop."""
        step_limit = max_steps or self.max_steps
        scratchpad: List[Dict[str, Any]] = []
        final_answer = ""
        started_at = datetime.now().isoformat()

        for step_index in range(1, step_limit + 1):
            prompt = self._build_react_prompt(user_request, scratchpad)
            llm_output = self._do_inference(prompt).strip()
            parsed = self._parse_react_output(llm_output)

            step_record: Dict[str, Any] = {
                "step": step_index,
                "prompt": prompt,
                "llm_output": llm_output,
                "thought": parsed.get("thought", ""),
            }

            if parsed.get("finish"):
                final_answer = parsed.get("final_answer", "")
                step_record["final_answer"] = final_answer
                scratchpad.append(step_record)
                self._finish_run(user_request, final_answer, scratchpad, started_at, metadata)
                return final_answer

            action = parsed.get("action")
            if not action:
                final_answer = parsed.get("final_answer") or llm_output
                step_record["final_answer"] = final_answer
                scratchpad.append(step_record)
                self._finish_run(user_request, final_answer, scratchpad, started_at, metadata)
                return final_answer

            tool_name, tool_args = action
            observation = self._execute_tool(tool_name, tool_args)
            step_record.update(
                {
                    "action": tool_name,
                    "action_input": tool_args,
                    "observation": observation,
                }
            )
            scratchpad.append(step_record)

        final_answer = self._max_steps_answer(scratchpad)
        self._finish_run(user_request, final_answer, scratchpad, started_at, metadata)
        return final_answer

    def _build_react_prompt(
        self,
        user_request: str,
        scratchpad: List[Dict[str, Any]],
    ) -> str:
        transcript = []
        for step in scratchpad:
            transcript.append(f"Thought: {step.get('thought', '')}")
            if "action" in step:
                transcript.append(f"Action: {step['action']}")
                transcript.append(f"Action Input: {json.dumps(step.get('action_input', {}), ensure_ascii=False)}")
                transcript.append(f"Observation: {json.dumps(step.get('observation', {}), ensure_ascii=False)}")
            if "final_answer" in step:
                transcript.append(f"Final Answer: {step['final_answer']}")

        scratchpad_text = "\n".join(transcript) if transcript else "(no prior steps)"
        if self.memory_type == "summarized":
            scratchpad_text = self._summarize_scratchpad(scratchpad)

        return f"""{self.sys_prompt}
{self.option_prompt}

You are a single REACT medical EHR agent. Complete the user task by looping over:
1. Think: decide the next safe, task-relevant step.
2. Act: call exactly one available tool when EHR data or CRUD is needed.
3. Observe: use the tool response before deciding the next step.

Output exactly one of these formats each turn:

Thought: brief task-state reasoning
Action: tool_name
Action Input: {{"arg": "value"}}

or:

Thought: brief task-state reasoning
Final Answer: answer

{self.tool_descriptions}

User task:
{user_request}

Previous REACT steps:
{scratchpad_text}
"""

    def _summarize_scratchpad(self, scratchpad: List[Dict[str, Any]]) -> str:
        if not scratchpad:
            return "(no prior steps)"
        summary = []
        for step in scratchpad:
            summary.append(
                {
                    "step": step.get("step"),
                    "thought": step.get("thought", ""),
                    "action": step.get("action"),
                    "action_input": step.get("action_input"),
                    "observation_summary": summarize_memory(step.get("observation"), 800)
                    if "observation" in step
                    else None,
                    "final_answer": step.get("final_answer"),
                }
            )
        return summarize_memory(summary, 2200)

    def _parse_react_output(self, output: str) -> Dict[str, Any]:
        thought = self._extract_line_block(output, "Thought")
        final_answer = self._extract_final_answer(output)
        if final_answer is not None:
            return {"thought": thought, "finish": True, "final_answer": final_answer}

        action = self._extract_action(output)
        if action:
            return {"thought": thought, "finish": False, "action": action}

        return {"thought": thought, "finish": False, "final_answer": output}

    def _extract_final_answer(self, output: str) -> Optional[str]:
        patterns = [
            r"Final Answer\s*:\s*(.*)\Z",
            r"FINISH\s*:?\s*(.*)\Z",
        ]
        for pattern in patterns:
            match = re.search(pattern, output, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return None

    def _extract_action(self, output: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        call_match = re.search(
            r"Action\s*:\s*([A-Za-z_][\w]*)\s*\((\{.*\})\)",
            output,
            flags=re.DOTALL,
        )
        if call_match:
            return call_match.group(1).strip(), self._parse_json_args(call_match.group(2))

        action_match = re.search(r"Action\s*:\s*([A-Za-z_][\w]*)", output)
        if action_match:
            tool_name = action_match.group(1).strip()
            args_text = self._extract_action_input(output)
            return tool_name, self._parse_json_args(args_text)
        return None

    def _extract_action_input(self, output: str) -> str:
        match = re.search(
            r"Action Input\s*:\s*(.*?)(?:\n\s*(?:Observation|Thought|Final Answer)\s*:|\Z)",
            output,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return match.group(1).strip() if match else "{}"

    def _parse_json_args(self, args_text: str) -> Dict[str, Any]:
        if not args_text:
            return {}
        cleaned = args_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return self._parse_relaxed_args(cleaned)

    def _parse_relaxed_args(self, args_text: str) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        for line in args_text.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().strip('"').strip("'")
            value = value.strip().strip(",")
            try:
                args[key] = json.loads(value)
            except json.JSONDecodeError:
                args[key] = value.strip('"').strip("'")
        return args

    def _extract_line_block(self, output: str, label: str) -> str:
        match = re.search(
            rf"{label}\s*:\s*(.*?)(?:\n\s*(?:Action|Action Input|Observation|Final Answer)\s*:|\Z)",
            output,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return match.group(1).strip() if match else ""

    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Unknown tool '{tool_name}'",
                "available_tools": sorted(self.tools.keys()),
            }
        try:
            result = self.tools[tool_name](**tool_args)
            if isinstance(result, dict):
                return result
            return {"success": True, "result": result}
        except TypeError as e:
            return {
                "success": False,
                "error": f"Invalid arguments for {tool_name}: {str(e)}",
                "action_input": tool_args,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "action_input": tool_args}

    def _max_steps_answer(self, scratchpad: List[Dict[str, Any]]) -> str:
        if not scratchpad:
            return "No answer."
        last_observation = scratchpad[-1].get("observation")
        if last_observation is not None:
            return json.dumps(last_observation, ensure_ascii=False)
        return scratchpad[-1].get("llm_output", "No answer.")

    def _record_history(
        self,
        user_request: str,
        final_answer: str,
        scratchpad: List[Dict[str, Any]],
        started_at: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.history.append(
            {
                "agent_type": "react",
                "timestamp": datetime.now().isoformat(),
                "started_at": started_at,
                "completed_at": datetime.now().isoformat(),
                "model": self.model_name,
                "thinking_mode": self.thinking_mode,
                "memory_type": self.memory_type,
                "metadata": metadata or {},
                "user": user_request,
                "agent": final_answer,
                "steps": scratchpad,
            }
        )

    def _finish_run(
        self,
        user_request: str,
        final_answer: str,
        scratchpad: List[Dict[str, Any]],
        started_at: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._record_history(user_request, final_answer, scratchpad, started_at, metadata)
        log_path = self._write_run_log(self.history[-1])
        self.history[-1]["log_path"] = log_path

    def _write_run_log(self, record: Dict[str, Any]) -> str:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.model_name)
        task_id = record.get("metadata", {}).get("task_id", "task")
        safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.output_dir / f"react_{safe_model}_{safe_task}_{timestamp}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, default=str, indent=2, ensure_ascii=False)
        return str(path)

    def dump_history(self, output_dir: Optional[str] = None) -> str:
        directory = Path(output_dir) if output_dir else Path.cwd()
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.model_name)
        path = directory / f"agent_history_{safe_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, default=str, indent=2, ensure_ascii=False)
        return str(path)

    def _do_inference(self, request: str) -> str:
        raise NotImplementedError


class REACTAgent(MASBAgent):
    """OpenAI-backed REACT agent by default."""

    def __init__(self, model_name: str, sys_prompt: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(model_name, sys_prompt=sys_prompt, **kwargs)
        import openai

        api_key = (
            os.getenv("OPENAI_API_KEY")
            or MASBAgent._llm_config.get("openai_api_key")
            or MASBAgent._llm_config.get("OPENAI_API_KEY")
        )
        self.client = openai.OpenAI(api_key=api_key)

    def _do_inference(self, request: str) -> str:
        request_args: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": request},
            ],
        }
        if self.model_name.startswith(("gpt-5", "o1", "o3", "o4")):
            request_args["max_completion_tokens"] = 1200
        else:
            request_args["temperature"] = 0
            request_args["max_tokens"] = 1200
        response = self.client.chat.completions.create(**request_args)
        return response.choices[0].message.content or ""


class OpenAIAgent(REACTAgent):
    """Compatibility alias for existing benchmark code."""


class GoogleAgent(MASBAgent):
    """Gemini-backed REACT agent with the same loop and tools."""

    def __init__(self, model_name: str, sys_prompt: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(model_name, sys_prompt=sys_prompt, **kwargs)
        from google import genai
        from google.genai import types

        self._genai_types = types
        api_key = os.getenv("GOOGLE_API_KEY") or MASBAgent._llm_config.get("google_api_key")
        self.client = genai.Client(api_key=api_key)

    def _do_inference(self, request: str) -> str:
        config = self._genai_types.GenerateContentConfig(system_instruction=self.sys_prompt)
        response = self.client.models.generate_content(
            model=self.model_name,
            config=config,
            contents=request,
        )
        return response.text or ""


class AnthropicAgent(MASBAgent):
    """Anthropic-backed REACT agent with the same loop and tools."""

    def __init__(self, model_name: str, sys_prompt: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(model_name, sys_prompt=sys_prompt, **kwargs)
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY") or MASBAgent._llm_config.get("anthropic_api_key")
        self.client = anthropic.Anthropic(api_key=api_key)

    def _do_inference(self, request: str) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1200,
            temperature=0,
            system=self.sys_prompt,
            messages=[{"role": "user", "content": request}],
        )
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")


class TogetherAgent(MASBAgent):
    """Together-backed REACT agent with the same loop and tools."""

    def __init__(self, model_name: str, sys_prompt: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(model_name, sys_prompt=sys_prompt, **kwargs)
        import openai

        api_key = os.getenv("TOGETHER_API_KEY") or MASBAgent._llm_config.get("together_api_key")
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1"),
        )

    def _do_inference(self, request: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=1200,
            temperature=0,
            messages=[
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": request},
            ],
            stream=False,
        )
        return response.choices[0].message.content or ""


class AgentsManager:
    """Factory for benchmark runs. Every created agent uses REACT."""

    agents: List[MASBAgent] = []

    MODEL_ALIASES: Dict[str, Tuple[str, str]] = {
        "gpt-5": ("openai", "gpt-5"),
        "gpt-5-mini": ("openai", "gpt-5-mini"),
        "gpt-4o-mini": ("openai", "gpt-4o-mini"),
        "gpt-4o": ("openai", "gpt-4o"),
        "gemini3-pro": ("google", "gemini-3-pro-preview"),
        "gemini-3-pro": ("google", "gemini-3-pro-preview"),
        "gemini-2.5-flash": ("google", "gemini-2.5-flash"),
        "gemini-2.5-pro": ("google", "gemini-2.5-pro"),
        "gemini-3-pro-preview": ("google", "gemini-3-pro-preview"),
        "claude-sonnet-4.5": ("anthropic", "claude-sonnet4.5"),
        "claude-sonnet4.5": ("anthropic", "claude-sonnet4.5"),
        "claude-haiku-4.5": ("anthropic", "claude-haiku-4.5"),
        "claude-opus-4.1": ("anthropic", "claude-opus-4.1"),
        "claude-opus-4.5": ("anthropic", "claude-opus-4.1"),
        "Llama-4-Scout-17B-16E": ("together", "Llama-4-Scout-17B-16E"),
        "Llama-4-Scout-17B-16E-Instruct": ("together", "Llama-4-Scout-17B-16E-Instruct"),
        "Llama-Guard-4-12B": ("together", "Llama-Guard-4-12B"),
        "llama-3.3": ("together", "Llama-3.3-70B-Instruct-Turbo"),
        "Llama-3.3-70B-Instruct-Turbo": ("together", "Llama-3.3-70B-Instruct-Turbo"),
        "Llama-3.1-70B-Instruct-Turbo": ("together", "Llama-3.1-70B-Instruct-Turbo"),
        "llama-3.1-70B": ("together", "llama-3.1-70B"),
        "llama3.1-8b": ("together", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
        "llama-3.1-8b": ("together", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
        "llama3.1(8b)": ("together", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
        "llama-guard-3-8b": ("together", "meta-llama/Llama-Guard-3-8B"),
        "llama-guard3-8b": ("together", "meta-llama/Llama-Guard-3-8B"),
        "llama guard 3(8b)": ("together", "meta-llama/Llama-Guard-3-8B"),
        "deepseekv3.1": ("together", "deepseekv3.1"),
        "deepseekv3": ("together", "deepseekv3.1"),
        "qwen3": ("together", "qwen3-next-80b-a3b-instruct"),
        "qwen3-next": ("together", "qwen3-next-80b-a3b-instruct"),
        "qwen3-next-81b": ("together", "qwen3-next-80b-a3b-instruct"),
        "qwen3-next-80b": ("together", "qwen3-next-80b-a3b-instruct"),
        "qwen3-next-80b-a3b-instruct": ("together", "qwen3-next-80b-a3b-instruct"),
        "qwen3-next-80b-a3b-thinking": ("together", "qwen3-next-80b-a3b-thinking"),
        "gemma3n": ("together", "gemma3n-e4b-instruct"),
        "gemma3n-e4b-instruct": ("together", "gemma3n-e4b-instruct"),
        "Mixtral-8x7B": ("together", "Mixtral-8x7B"),
        "Mixtral-8x22B-Instruct": ("together", "Mixtral-8x22B-Instruct"),
        "mistral": ("together", "Mixtral-8x22B-Instruct"),
        "kimi-k2": ("together", "kimi-k2-instruct"),
        "kimi-k2-instruct": ("together", "kimi-k2-instruct"),
        "kimi-k2-thinking": ("together", "kimi-k2-thinking"),
    }

    @staticmethod
    def create_agent(
        name: str,
        sys_prompt: str,
        thinking_mode: str = "disabled",
        memory_type: str = "complete",
    ) -> MASBAgent:
        if not MASBAgent._llm_config:
            MASBAgent.load_config()

        name = model_alias_for_options(name, thinking_mode)
        provider, config_key = AgentsManager.MODEL_ALIASES.get(name, ("openai", name))
        model_config = MASBAgent._llm_config.get(config_key, {})
        model_name = model_config.get("model", config_key) if isinstance(model_config, dict) else config_key

        if provider == "openai":
            agent: MASBAgent = OpenAIAgent(model_name, sys_prompt=sys_prompt, thinking_mode=thinking_mode, memory_type=memory_type)
        elif provider == "google":
            agent = GoogleAgent(model_name, sys_prompt=sys_prompt, thinking_mode=thinking_mode, memory_type=memory_type)
        elif provider == "anthropic":
            agent = AnthropicAgent(model_name, sys_prompt=sys_prompt, thinking_mode=thinking_mode, memory_type=memory_type)
        elif provider == "together":
            agent = TogetherAgent(model_name, sys_prompt=sys_prompt, thinking_mode=thinking_mode, memory_type=memory_type)
        else:
            raise ValueError(f"Unknown LLM provider for agent '{name}': {provider}")

        AgentsManager.agents.append(agent)
        return agent

    @staticmethod
    def dump_history(output_dir: Optional[str] = None) -> List[str]:
        return [agent.dump_history(output_dir=output_dir) for agent in AgentsManager.agents]

def _load_simple_yaml(text: str) -> Dict[str, Any]:
    """
    Small fallback for this repo's llm_models.yaml when PyYAML is unavailable.

    It supports top-level scalar keys and one-level nested mappings such as:
    gpt-4o:
        model: "gpt-4o"
    """
    data: Dict[str, Any] = {}
    current_key: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not raw_line.startswith((" ", "\t")):
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if value:
                data[key] = value.strip("\"'")
                current_key = None
            else:
                data[key] = {}
                current_key = key
        elif current_key:
            key, _, value = line.strip().partition(":")
            if key:
                data[current_key][key.strip()] = value.strip().strip("\"'")
    return data


MASBAgent.load_config()
