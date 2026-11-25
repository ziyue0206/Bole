"""Render the real protocol and the registered definitions into a user-editable template."""

import json
import re
from pathlib import Path

RESOURCE_DISCOVERY = "The workspace may contain reusable resources. You may inspect and use them at your discretion."


def render(config, episode, info, definitions):
    mode = config["mode"]
    rules = (
        "The benchmark's native evaluator grades this episode. You cannot reset it. "
        "Use finish to end or abandon immediately; finish does not assert success. "
        "A normal final response also ends the episode. The platform may close the episode "
        "at a budget boundary, cancellation, unrecoverable error, or after its first native "
        f"success followed by {config['termination']['success_grace_steps']} actually executed "
        "control steps. The platform does not execute extra actions for you. "
        "Native success, scores and success-triggered countdowns are hidden during execution. "
    )
    rules += (
        "This is continuous learning within this task: your native session and writable "
        "workspace persist. After the episode stops, one short native-result feedback is "
        "delivered. The runner separately initializes the next configured instance. "
        "No reflection, written summary or forced compaction is required."
        if mode == "learn" else
        "This is an independent evaluation in a new native session. Any initial learned "
        "workspace and native memory are private writable copies. Other evaluation episodes "
        "cannot see your changes. No native grading feedback is sent after the episode."
    )
    values = {
        "task_name": episode["task_id"],
        "task_description": info.get("instruction") or "Use the native task instruction in the initial observation.",
        "robot_interface": json.dumps({"capabilities": info["capabilities"],
                                       "registered_tools": definitions}, indent=2),
        "observation_delivery": "The initial observation is delivered once with this message. "
            "Subsequent observations are obtained when you request observe. Camera names and "
            "image order are labeled in each observation. A read does not advance physics.",
        "episode_rules": rules,
        "budgets": json.dumps({"limits": config["limits"], "native_control_limit": info.get("step_limit"),
                               "clock_origin": "initial observation accepted by native harness"}, indent=2),
        "allowed_material": "Use the declared tools, your isolated workspace and the configured "
            "native Codex capabilities. Only explicitly selected learned assets are prior "
            "material. Install Python dependencies only in /workspace/.venv; use "
            "/workspace/.venv/bin/python -m pip. Native shell networking is "
            + ("enabled. " if config.get("harness", {}).get("shell", {}).get("network_access", False) else "disabled. ") +
            "Hidden simulator state, evaluator files, result logs and other "
            "episodes are not mounted into your container and are not public observations.",
    }
    template = Path(config["harness"]["instruction"]).read_text()
    def replace(match):
        key = match.group(1).strip()
        if key not in values:
            raise ValueError(f"unknown prompt placeholder: {key}")
        return values[key]
    rendered = re.sub(r"\{\{\s*([^{}]+?)\s*\}\}", replace, template)
    return rendered + ("\n\n" + RESOURCE_DISCOVERY + "\n" if mode == "eval" else "")
