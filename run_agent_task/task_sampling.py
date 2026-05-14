"""Task loading and adjacent attack/clean pair sampling."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

from run_agent_task.common import REPO_ROOT, read_json

Task = Dict[str, Any]


REFERENCE_FIELDS = ("correct_answer", "wrong_answer", "attack_answer")
MANIFEST_FIELDS = (
    "task_key",
    "id",
    "category",
    "source_file",
    "task_index",
    "pair_id",
    "pair_role",
    "attack",
    "attack_type",
    "fhir_resource",
    "fhir_request_url",
)


def usable(task: Task) -> bool:
    return "instruction" in task and "context" in task


def task_record(task: Task, path: Path, category: str, index: int) -> Task:
    path = path.resolve()
    try:
        source_file = str(path.relative_to(REPO_ROOT))
    except ValueError:
        source_file = str(path)
    record = {
        "task_key": f"{source_file}:{index}:{task.get('id', '')}",
        "id": task.get("id", ""),
        "category": category,
        "source_file": source_file,
        "task_index": index,
        "attack": task.get("attack"),
        "attack_type": task.get("attack_type"),
        "fhir_resource": task.get("fhir_resource"),
        "fhir_request_url": task.get("fhir_request_url"),
        "instruction": task["instruction"],
        "context": task["context"],
    }
    for field in REFERENCE_FIELDS:
        record[field] = task.get(field, [])
    return record


def adjacent_attack_pairs(tasks_dir: Path) -> List[List[Task]]:
    pairs: List[List[Task]] = []
    tasks_dir = Path(tasks_dir).expanduser().resolve()
    for path in sorted(tasks_dir.glob("*.json")):
        payload = read_json(path)
        tasks = payload.get("tasks", [])
        if not isinstance(tasks, list):
            continue
        category = payload.get("category", path.stem)
        for index, task in enumerate(tasks[:-1]):
            next_task = tasks[index + 1]
            if task.get("attack") is True and next_task.get("attack") is False and usable(task) and usable(next_task):
                pair_id = f"{path.stem}:{task.get('id', index)}:{next_task.get('id', index + 1)}"
                attack = task_record(task, path, category, index)
                clean = task_record(next_task, path, category, index + 1)
                attack.update({"pair_id": pair_id, "pair_role": "attack"})
                clean.update({"pair_id": pair_id, "pair_role": "clean"})
                pairs.append([attack, clean])
    return pairs


def sample_task_pair(tasks_dir: Path, seed: int) -> List[Task]:
    pairs = adjacent_attack_pairs(tasks_dir)
    if not pairs:
        raise ValueError(f"No adjacent attack:true then attack:false pairs found in {tasks_dir}.")
    return random.Random(seed).choice(pairs)


def attacked_tasks(tasks_dir: Path) -> List[Task]:
    records: List[Task] = []
    tasks_dir = Path(tasks_dir).expanduser().resolve()
    for path in sorted(tasks_dir.glob("*.json")):
        payload = read_json(path)
        tasks = payload.get("tasks", [])
        if not isinstance(tasks, list):
            continue
        category = payload.get("category", path.stem)
        for index, task in enumerate(tasks):
            if task.get("attack") is True and usable(task):
                records.append(task_record(task, path, category, index))
    return records


def sample_attacked_tasks(tasks_dir: Path, seed: int, count: int) -> List[Task]:
    records = attacked_tasks(tasks_dir)
    if len(records) < count:
        raise ValueError(f"Found only {len(records)} attacked tasks in {tasks_dir}; need {count}.")
    return random.Random(seed).sample(records, count)


def task_prompt(task: Task) -> str:
    return f"Instruction:\n{task['instruction']}\n\nContext:\n{task['context']}"


def task_metadata(task: Task, model_alias: str, agent_type: str) -> Dict[str, Any]:
    metadata = {
        "task_id": task.get("id"),
        "task_key": task.get("task_key"),
        "category": task.get("category"),
        "source_file": task.get("source_file"),
        "task_index": task.get("task_index"),
        "pair_id": task.get("pair_id"),
        "pair_role": task.get("pair_role"),
        "attack": task.get("attack"),
        "attack_type": task.get("attack_type"),
        "fhir_resource": task.get("fhir_resource"),
        "fhir_request_url": task.get("fhir_request_url"),
    }
    metadata.update({"model_alias": model_alias, "agent_type": agent_type})
    return metadata


def manifest_task(task: Task) -> Dict[str, Any]:
    record = {key: task.get(key) for key in MANIFEST_FIELDS}
    for field in REFERENCE_FIELDS:
        record[field] = task.get(field, [])
    return record
