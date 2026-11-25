"""One plugin discovery path for built-ins and separately installed adapters."""

from importlib import import_module
from importlib.metadata import entry_points
import json
from pathlib import Path

BUILTINS = {
    "benchmarks": {"robodojo": "bole.benchmarks.robodojo.adapter:RoboDojoAdapter"},
    "harnesses": {"codex": "bole.harnesses.codex:CodexHarness"},
    "policies": {},
}


def discover(kind):
    if kind not in BUILTINS:
        raise ValueError(f"unknown adapter kind: {kind}")
    frozen = Path(__file__).with_name("_frozen_plugins.json")
    if frozen.is_file():
        return json.loads(frozen.read_text())[kind]
    found = dict(BUILTINS[kind])
    for entry in entry_points().select(group=f"bole.{kind}"):
        if entry.name in found:
            raise ValueError(f"duplicate {kind} adapter: {entry.name}")
        found[entry.name] = entry.value
    return found


def load(kind, name):
    found = discover(kind)
    if name not in found:
        raise ValueError(f"{kind} adapter {name!r} is not installed; available: {sorted(found)}")
    module, attribute = found[name].split(":", 1)
    factory = getattr(import_module(module), attribute)
    return factory()
