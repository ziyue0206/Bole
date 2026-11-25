"""A working external policy example, not a trained robot controller.

It returns only user-supplied candidate actions. It neither executes actions nor
reads native evaluation state. The benchmark validates any accepted candidate
when the harness separately invokes execute.
"""
from copy import deepcopy
import json


class RecordedActions:
    def __init__(self):
        self.actions = None

    def load(self, config, resources):
        actions = config.get("actions")
        if not isinstance(actions, list) or not actions or not all(isinstance(item, dict) for item in actions):
            raise ValueError("recorded-actions requires policy.actions: a nonempty list of action objects")
        # Reject values that cannot be represented in the public tool protocol.
        json.dumps(actions, allow_nan=False)
        self.actions = deepcopy(actions)

    def infer(self, observation):
        if self.actions is None:
            raise RuntimeError("load must be called before infer")
        return {"actions": deepcopy(self.actions),
                "source": "explicit_policy_configuration",
                "observation_id": observation.get("observation_id"),
                "executed": False}

    def close(self):
        self.actions = None
