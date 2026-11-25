"""Worker-only pose diagnostics v1; candidate grasp is not a slip verdict.

Old v2 grasp_slip means height drop from the first detected lift; old
transport_tilt means change of body-z from that lift, not gravity tilt.
This module adds separately named raw quantities without changing those scores.
"""

import math


def unit_quaternion(q):
    if len(q) != 4 or not all(map(math.isfinite, q)):
        raise ValueError("invalid xyzw quaternion")
    n = math.hypot(*q)
    if n < 1e-8:
        raise ValueError("zero quaternion")
    return [v / n for v in q]


def conjugate(q):
    x, y, z, w = q
    return [-x, -y, -z, w]


def multiply(a, b):
    x, y, z, w = a
    X, Y, Z, W = b
    return [w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X,
            w*Z+x*Y-y*X+z*W, w*W-x*X-y*Y-z*Z]


def rotate(q, v):
    return multiply(multiply(q, [*v, 0]), conjugate(q))[:3]


def rotation_deg(a, b):
    d = abs(sum(x*y for x, y in zip(unit_quaternion(a), unit_quaternion(b))))
    return math.degrees(2*math.acos(max(-1., min(1., d))))


def relative_pose(obj, link):
    q = conjugate(unit_quaternion(link[3:]))
    return {"position_m": rotate(q, [a-b for a, b in zip(obj["position_m"], link[:3])]),
            "quaternion_xyzw": multiply(q, unit_quaternion(obj["quaternion_xyzw"]))}


class PoseDiagnostics:
    """Conservative continuous-contact candidate gate with explicit unknowns."""

    def __init__(self, baseline, roles):
        self.baseline, self.roles = baseline, roles
        self.reference = {}
        self.lift_reference = {}

    def sample(self, poses, robots, contacts, contact_status):
        result = {"schema_version": "robodojo-pose-diagnostics-v2",
                  "implementation_version": "v2-missing-phase-boundary-2026-11-16",
                  "frame": "world_m_xyzw; relative poses in measured link6 frame, no TCP offset applied",
                  "axis_definition": "local +z; stack_bowls native is_axis_up axis=[0,0,1]",
                  "up_reference": "world +Z; native IsaacRLEnvCfg gravity=(0,0,-9.81) m/s^2",
                  "reference_scope": {"height_drop_from_initial_lift_m": "episode first candidate lift per object/arm",
                                      "candidate_relative_motion": "contiguous bilateral-contact lift candidate phase"},
                  "thresholds": {"version": "candidate-v2-unvalidated-both-fingers-closed", "finger_contact_n": 0.01,
                                 "max_finger_position_m": 0.040, "lift_from_baseline_m": 0.01},
                  "legacy_definitions": {"grasp_slip": "height_drop_from_first_detected_lift",
                                         "transport_tilt": "body_z_change_from_first_detected_lift"},
                  "objects": {}, "slip_event": None, "drop_event": None,
                  "classification_status": "unvalidated_no_event_verdict"}
        measured = {r["arm"]: r for r in robots or []}
        observable_objects = {name for name, pose in (poses or {}).items() if pose is not None}
        # A missing pose/robot interrupts the continuous reference. The legacy
        # episode-first lift height is intentionally retained and separately named.
        self.reference = {key: value for key, value in self.reference.items()
                          if key[0] in observable_objects and key[1] in measured}
        fingers = {}
        for pair in contacts or []:
            if pair["normal_force_n"] <= 0.01:
                continue
            for a, b in (pair["bodies"], pair["bodies"][::-1]):
                if a["role"] == "gripper" and b["name"] in self.roles:
                    fingers.setdefault((b["name"], a["arm"]), set()).add(a["name"])
        for name in sorted(set(self.baseline) | set(self.roles) | set(poses or {})):
            pose = (poses or {}).get(name)
            if pose is None:
                result["objects"][name] = {"status": "missing"}
                continue
            base = self.baseline.get(name)
            axis = rotate(unit_quaternion(pose["quaternion_xyzw"]), [0., 0., 1.])
            obj = {"status": "valid", "role": self.roles.get(name, "unknown"),
                   "body_z_to_world_up_deg": math.degrees(math.acos(max(-1., min(1., axis[2])))),
                   "rotation_from_baseline_deg": rotation_deg(base["quaternion_xyzw"], pose["quaternion_xyzw"]) if base else None,
                   "height_change_from_baseline_m": pose["position_m"][2] - base["position_m"][2] if base else None,
                   "arms": {}}
            for arm, robot in measured.items():
                key = (name, arm)
                rel = relative_pose(pose, robot["ee_transform_xyzw"])
                positions = robot.get("measured_finger_position_m") or []
                closed = bool(positions) and all(map(math.isfinite, positions)) and max(positions) <= .040
                opened = bool(positions) and all(map(math.isfinite, positions)) and min(positions) > .040
                bilateral = fingers.get(key) == {"link7", "link8"}
                lifted = base is not None and obj["height_change_from_baseline_m"] >= .01
                # Positive bilateral reports remain evidence when resting support
                # is partial; missing reports never establish absence of contact.
                candidate = bilateral and closed and lifted
                if candidate:
                    self.reference.setdefault(key, rel)
                    self.lift_reference.setdefault(key, pose)
                    stage = "bilateral_contact_lift_candidate"
                else:
                    self.reference.pop(key, None)
                    stage = "open_measured" if opened else "unknown_contact_or_hold" if contact_status != "ok" else "no_bilateral_lift_evidence"
                anchor = self.reference.get(key)
                lift = self.lift_reference.get(key)
                obj["arms"][arm] = {"stage": stage, "bilateral_finger_contact_observed": bilateral,
                                     "finger_positions_m": positions, "relative_to_link6": rel,
                                     "candidate_relative_translation_change_m": math.dist(anchor["position_m"], rel["position_m"]) if anchor else None,
                                     "candidate_relative_rotation_change_deg": rotation_deg(anchor["quaternion_xyzw"], rel["quaternion_xyzw"]) if anchor else None,
                                     "height_drop_from_initial_lift_m": max(0., lift["position_m"][2] - pose["position_m"][2]) if lift else None,
                                     "grasp_confirmed": None, "release_intent": "unknown"}
            result["objects"][name] = obj
        return result
