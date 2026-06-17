"""Load CrewAI agent and task prompts from the YAML config files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "PyYAML is required to load task definitions from config/tasks.yaml."
    ) from exc


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config"
TASKS_YAML_PATH = CONFIG_DIR / "tasks.yaml"
AGENTS_YAML_PATH = CONFIG_DIR / "agents.yaml"


@lru_cache(maxsize=1)
def _load_tasks_yaml():
    """Load and cache the task definitions from the YAML file."""
    with TASKS_YAML_PATH.open("r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj) or {}
    return payload.get("tasks", {})


@lru_cache(maxsize=1)
def _load_agents_yaml():
    """Load and cache the agent definitions from the YAML file."""
    with AGENTS_YAML_PATH.open("r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj) or {}
    return payload.get("agents", {})


def get_task_definition(task_name):
    """Return one task definition from config/tasks.yaml."""
    tasks = _load_tasks_yaml()
    try:
        task = tasks[task_name]
    except KeyError as exc:
        raise KeyError(f"Unknown task name: {task_name}") from exc
    if not isinstance(task, dict):
        raise TypeError(f"Task definition for {task_name!r} must be a mapping.")
    return task


def get_agent_definition(agent_name):
    """Return one agent definition from config/agents.yaml."""
    agents = _load_agents_yaml()
    try:
        agent = agents[agent_name]
    except KeyError as exc:
        raise KeyError(f"Unknown agent name: {agent_name}") from exc
    if not isinstance(agent, dict):
        raise TypeError(f"Agent definition for {agent_name!r} must be a mapping.")
    return agent


def build_json_task_prompt(task_name, payload):
    """Build the standard two-part task prompt from YAML task text."""
    import json

    task = get_task_definition(task_name)
    description = task.get("description", "").strip()
    return f"{description}\n\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
