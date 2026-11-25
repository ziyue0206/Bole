"""Read RoboDojo's native task and layout declarations without importing Isaac.

The public registry resolves a module and a class with the same task name.  We
inspect that declaration with AST, and read the selected checkout's task quotas
and official dimension mapping; simulator code is never executed by planning.
"""

from __future__ import annotations

import ast
import copy
import re
from pathlib import Path

import yaml


class CatalogError(ValueError):
    """The selected native source or assets cannot supply the requested plan."""


def _literal_assignments(tree: ast.AST) -> dict:
    values = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        try:
            literal = ast.literal_eval(value)
        except (ValueError, TypeError, SyntaxError):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                values[target.id] = literal
    return values


def _static_instruction(tree: ast.Module, task: str) -> list[str] | None:
    """Extract only unconditional literal templates, never scene-dependent data.

    A nonliteral or conditional native instruction is supplied by the worker
    after reset.  Metadata must not invent an instruction for such a task.
    """
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}

    def find_method(name: str, visited: set[str]):
        if name in visited or name not in classes:
            return None
        visited.add(name)
        cls = classes[name]
        for node in cls.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "gen_instruction":
                    return node
        for base in cls.bases:
            if isinstance(base, ast.Name):
                method = find_method(base.id, visited)
                if method is not None:
                    return method
        return None

    method = find_method(task, set())
    if method is None:
        return None
    values = {}
    for node in method.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # A docstring has no effect on the returned templates.
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            found = _literal_assignments(ast.Module(body=[node], type_ignores=[]))
            if not found:
                return None
            values.update(found)
        elif isinstance(node, ast.Return):
            if isinstance(node.value, ast.Name):
                result = values.get(node.value.id)
            else:
                try:
                    result = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError):
                    return None
            if isinstance(result, str):
                result = [result]
            if isinstance(result, (list, tuple)) and result and all(
                isinstance(value, str) and value.strip() for value in result
            ):
                return list(result)
            return None
        else:
            return None
    return None


class Catalog:
    def __init__(self, source: Path, assets: Path):
        self.source = Path(source).expanduser().resolve()
        self.assets = Path(assets).expanduser().resolve()
        self._tasks: list[dict] | None = None
        self.dimensions: dict[str, list[str]] = {}
        self.expected_seeds: list[int] = []

    def tasks(self) -> list[dict]:
        if self._tasks is not None:
            return copy.deepcopy(self._tasks)
        base = self.source / "task/RoboDojo"
        index_path = base / "config/_task.yml"
        summary_path = self.source / "scripts/internal/summarize_result.py"
        registry_path = base / "task_registry.py"
        for path in (index_path, summary_path, registry_path):
            if not path.is_file():
                raise CatalogError(f"Missing native catalog source: {path}")
        try:
            index = yaml.safe_load(index_path.read_text(encoding="utf-8"))
            metadata = _literal_assignments(
                ast.parse(summary_path.read_text(encoding="utf-8"), filename=str(summary_path))
            )
        except (OSError, SyntaxError, yaml.YAMLError) as error:
            raise CatalogError(f"Cannot read native catalog: {error}") from error
        if not isinstance(index, dict):
            raise CatalogError(f"Native task index is not a mapping: {index_path}")
        common, overrides = index.get("common", {}), index.get("tasks", {})
        if not isinstance(common, dict) or not isinstance(overrides, dict):
            raise CatalogError("Native task common/overrides must be mappings")
        dimensions, seeds = metadata.get("DIMENSIONS"), metadata.get("EXPECTED_SEEDS")
        if not isinstance(dimensions, dict) or not dimensions:
            raise CatalogError("Native DIMENSIONS must be a nonempty literal mapping")
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise CatalogError("Native EXPECTED_SEEDS must list distinct nonnegative integers")
        dimension_for = {}
        for dimension, names in dimensions.items():
            if not isinstance(dimension, str) or not isinstance(names, list) or not names:
                raise CatalogError("Invalid native dimension declaration")
            for name in names:
                if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
                    raise CatalogError(f"Invalid task name in native dimensions: {name!r}")
                if name in dimension_for:
                    raise CatalogError(f"Native task belongs to multiple dimensions: {name}")
                dimension_for[name] = dimension
        paired = set(dimensions.get("Generalization", []))
        expected = set(dimension_for) | {name + "_random" for name in paired}
        records = []
        for path in sorted((base / "tasks").glob("*.py")):
            if path.name.startswith("_"):
                continue
            task = path.stem
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError) as error:
                raise CatalogError(f"Cannot inspect native task {task}: {error}") from error
            if task not in {n.name for n in tree.body if isinstance(n, ast.ClassDef)}:
                continue  # Helper modules are not loadable task registry entries.
            if task not in expected:
                raise CatalogError(f"Native task has no official dimension/variant mapping: {task}")
            config_path = base / "config" / (task + ".yml")
            if not config_path.is_file():
                raise CatalogError(f"Missing native task configuration: {config_path}")
            settings = overrides.get(task, {})
            if settings is None:
                settings = {}
            if not isinstance(settings, dict):
                raise CatalogError(f"Invalid native task overrides: {task}")
            count = settings.get("eval_nums", common.get("eval_nums", 50))
            if type(count) is not int or count < 1:
                raise CatalogError(f"Invalid native episode quota for {task}: {count!r}")
            logical = task.removesuffix("_random") if task.endswith("_random") else task
            templates = _static_instruction(tree, task)
            records.append({
                "id": task,
                "logical_task_id": logical,
                "variant": "random" if task.endswith("_random") else "standard",
                "dimension": dimension_for[logical],
                "native_episodes_per_seed": count,
                "description": templates[0] if templates and len(templates) == 1 else None,
                "description_templates": templates,
                "description_status": "static_templates" if templates else "runtime_required",
                "description_source": f"task/RoboDojo/tasks/{task}.py:gen_instruction",
                "runtime_instruction_required": True,
                "source_path": str(path),
                "config_path": str(config_path),
            })
        missing = expected - {row["id"] for row in records}
        if missing:
            raise CatalogError(f"Missing native task implementations: {', '.join(sorted(missing))}")
        self.dimensions = copy.deepcopy(dimensions)
        self.expected_seeds = list(seeds)
        self._tasks = records
        return copy.deepcopy(records)

    def instances(self, task: str, seed: int) -> list[dict]:
        if task not in {row["id"] for row in self.tasks()}:
            raise CatalogError(f"Unknown RoboDojo task: {task!r}")
        if type(seed) is not int or seed < 0:
            raise CatalogError("Evaluation seed must be a nonnegative integer")
        directory = self.assets / "Eval_Layout/RoboDojo/arx_x5" / str(seed)
        if not directory.is_dir():
            raise CatalogError(f"Missing RoboDojo layout bank for seed {seed}: {directory}")
        pattern = re.compile(re.escape(task) + r"_(\d+)\.json\Z")
        paths = []
        suffixes = set()
        for path in directory.iterdir():
            match = pattern.fullmatch(path.name)
            if match is None:
                continue
            if not path.is_file() or path.stat().st_size == 0:
                raise CatalogError(f"Missing or empty RoboDojo layout: {path}")
            suffix = int(match.group(1))
            if suffix in suffixes:
                raise CatalogError(f"Ambiguous numeric layout suffix for {task}, seed {seed}: {suffix}")
            suffixes.add(suffix)
            paths.append((suffix, path))
        if not paths:
            raise CatalogError(f"No RoboDojo layouts for task {task}, seed {seed}: {directory}")
        # Native SeedManager uses the index in this numerical ordering, not the
        # filename suffix, as layout_id.  Noncontiguous suffixes are valid.
        return [
            {"seed": seed, "layout_id": index, "path": str(path.resolve()),
             "filename_index": suffix}
            for index, (suffix, path) in enumerate(sorted(paths, key=lambda item: item[0]))
        ]
