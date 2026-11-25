"""Native passive probe for PhysX signals and
probe_runs InteractionSignals/normalize. PhysX readings are private evidence.
Signals retain their individual validity and coverage information.
"""

from __future__ import annotations

import json
import math
import time
import os
from collections import defaultdict
from functools import wraps
from itertools import combinations, product
from math import hypot, isfinite

from .diagnostics import PoseDiagnostics
from .roles import VERIFIED_ROLE_MAPS, VERIFIED_FIXTURES, task_roles
from bole.probes import SafetyEvaluator, FAMILIES, UNITS

FORCE_KIND = "normal_resultant_from_impulse"
SOURCE = "omni.physics.tensors.RigidContactView.get_contact_data"
ENV_ROOT = "/World/envs/env_0"


def self_collision_scope(env=None):
    """Document the inherited two-arm scope without extending its algorithm."""
    manager = getattr(env, "robot_manager", None)
    robots = getattr(manager, "robot_list", None)
    return {
        "definition": "configured_unfiltered_pairs_within_the_two_target_arm_roots",
        "robot_roots": [f"{ENV_ROOT}/robot0", f"{ENV_ROOT}/robot1"],
        "arm_names": ["left", "right"],
        "valid_status_scope": "only_the_listed_roots_and_pairs_in_robot_pairs",
        "additional_robot_pairs": "not_evaluated_by_this_probe",
        "additional_robot_names": None if robots is None else [
            getattr(robot, "arm_name", "unnamed") for robot in robots if getattr(robot, "type", None) != "target"],
        "all_robot_self_collision_verified": False,
        "interpretation": "valid_does_not_establish_coverage_of_a_support_arm_or_other_robot",
    }


def contact_pairs(manifolds, dt):
    """Project impulses and sum across colliders in a canonical actor direction.

    A manifold appears once per collider pair and orientation. If a backend
    reports both orientations, retain the canonical orientation for that
    collider pair instead of adding a second copy of the force.
    """
    if not isfinite(dt) or dt <= 0:
        raise ValueError("invalid PhysX timestep")
    colliders = defaultdict(lambda: defaultdict(list))
    for a, b, shape_a, shape_b, points in manifolds:
        if a == b:
            continue
        reverse = a > b
        actors = (b, a) if reverse else (a, b)
        shapes = (shape_b, shape_a) if reverse else (shape_a, shape_b)
        colliders[actors, shapes][reverse].extend(points)

    totals = defaultdict(lambda: [0.0, 0.0, 0.0])
    for (actors, _), orientations in colliders.items():
        reverse = False not in orientations
        sign = -1 if reverse else 1
        for normal, impulse in orientations[reverse]:
            length = hypot(*normal)
            if not isfinite(length) or not length or not all(map(isfinite, impulse)):
                raise ValueError("invalid PhysX contact data")
            direction = [axis / length for axis in normal]
            magnitude = sum(i * n for i, n in zip(impulse, direction, strict=True))
            for axis, value in enumerate(direction):
                totals[actors][axis] += sign * magnitude * value

    return [(actors, hypot(*totals[actors]) / dt) for actors in sorted(totals)]


class PhysxSignals:
    def __init__(self):
        import omni.physics.tensors as tensors
        import omni.physx
        from pxr import PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdUtils

        self.Usd, self.UsdGeom, self.Physics = Usd, UsdGeom, UsdPhysics
        self.Schema = PhysxSchema
        self.encode = PhysicsSchemaTools.sdfPathToInt
        self.stage_cache = UsdUtils.StageCache.Get()
        self.simulation = omni.physx.get_physx_simulation_interface()
        self.physx = omni.physx.get_physx_interface()
        self.create_simulation_view = tensors.create_simulation_view
        self.bodies = {}
        self.objects = {}
        self.robot_pairs = {}

    def _prims(self, stage):
        root = stage.GetPrimAtPath(ENV_ROOT)
        return list(self.Usd.PrimRange(root, self.Usd.TraverseInstanceProxies())) if root else []

    def enable_reports(self, stage):
        for prim in self._prims(stage):
            if prim.HasAPI(self.Physics.RigidBodyAPI) and not prim.IsInstanceProxy():
                api = self.Schema.PhysxContactReportAPI.Apply(prim)
                api.CreateThresholdAttr().Set(0.0)

    def _body(self, path):
        name = path.rsplit("/", 1)[-1]
        for index, arm in enumerate(("left", "right")):
            root = f"{ENV_ROOT}/robot{index}"
            if path == root or path.startswith(root + "/"):
                return {
                    "id": path,
                    "name": name,
                    "role": "gripper" if name in {"link7", "link8"} else "arm_link",
                    "arm": arm,
                }
        for label, (object_path, role) in self.objects.items():
            if path == object_path:
                return {"id": path, "name": label, "role": role}
        role = "table" if "/Table" in path else "scene_fixture"
        return {"id": path, "name": name, "role": role}

    def bind(self, stage, objects):
        if self.UsdGeom.GetStageMetersPerUnit(stage) != 1 or (
            self.Physics.GetStageKilogramsPerUnit(stage) != 1
        ):
            raise ValueError("expected SI stage distance and mass units")
        self.objects = objects
        prims = self._prims(stage)
        bodies = {}
        report_missing = []
        sleep_ids = {}
        static = {}
        colliders = defaultdict(list)
        for prim in prims:
            if not prim.HasAPI(self.Physics.CollisionAPI):
                continue
            if not self.Physics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                continue
            actor = prim
            while actor and not actor.IsPseudoRoot():
                if actor.HasAPI(self.Physics.RigidBodyAPI):
                    path = str(actor.GetPath())
                    if self.Physics.RigidBodyAPI(actor).GetRigidBodyEnabledAttr().Get():
                        colliders[path].append(prim.GetPath())
                        bodies[path] = self._body(path)
                        if not self.Physics.RigidBodyAPI(actor).GetKinematicEnabledAttr().Get():
                            sleep_ids[path] = self.encode(actor.GetPath())
                        if not actor.HasAPI(self.Schema.PhysxContactReportAPI):
                            report_missing.append(path)
                    else:
                        static[str(prim.GetPath())] = path
                    break
                actor = actor.GetParent()
            else:
                path = str(prim.GetPath())
                static[path] = path
        self.bodies = bodies
        self.sleep_ids = sleep_ids
        self.bound_stage_id = self.stage_cache.GetId(stage).ToLongInt()
        for prim in prims:
            if not prim.HasAPI(self.Physics.ArticulationRootAPI):
                continue
            root = self._body(str(prim.GetPath()))
            if "arm" not in root:
                continue
            encoded = self.encode(prim.GetPath())
            self.sleep_ids.update(
                {path: encoded for path in self.sleep_ids if bodies[path].get("arm") == root["arm"]}
            )
        robot = {path: body for path, body in bodies.items() if "arm" in body}
        self.robot_pairs = self._robot_coverage(stage, prims, robot, colliders)
        self.pose_view = None
        self.pose_indices = {}
        paths = [path for path, _ in objects.values() if path]
        self.tensor_view = self.create_simulation_view("torch")
        if paths:
            self.pose_view = self.tensor_view.create_rigid_body_view(paths)
            self.pose_indices = {str(path): i for i, path in enumerate(self.pose_view.prim_paths)}
        self.contact_available = (
            bool(bodies)
            and not report_missing
            and all(path in bodies for path, _ in objects.values())
        )
        self.contact_view = None
        self.contact_binding_error = None
        if self.contact_available:
            try:
                self._bind_contacts(static)
            except Exception as error:
                self.contact_binding_error = str(error)
        return {
            "source": SOURCE,
            "force_kind": FORCE_KIND,
            "body_catalog": list(bodies.values()),
            "unreported_bodies": sorted(set(report_missing)),
            "static_collision_catalog": [
                {"collider_id": shape, "body": self._body(actor)}
                for shape, actor in sorted(static.items())
            ],
            "contact_binding_error": self.contact_binding_error,
            "robot_pairs": [
                {"body_ids": pair, **rule} for pair, rule in sorted(self.robot_pairs.items())
            ],
            "self_collision_scope": self_collision_scope(),
            "allowed_robot_pairs": [],
            "robot_scene_definition": "robot versus any non-robot body, including task objects",
            "scene_scope": ENV_ROOT,
            "pose_source": "omni.physics.tensors.RigidBodyView.get_transforms",
            "collision_group_scene_inverted": self.scene_inverted,
        }

    def _robot_coverage(self, stage, prims, robot, colliders):
        p = self.Physics
        filtered = []
        joint_exclusions = set()
        self_enabled = {}
        for prim in prims:
            path = prim.GetPath()
            if prim.HasAPI(p.FilteredPairsAPI):
                filtered.extend(
                    (path, target)
                    for target in p.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()
                )
            if prim.IsA(p.Joint):
                joint = p.Joint(prim)
                a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
                if len(a) == len(b) == 1 and not joint.GetCollisionEnabledAttr().Get():
                    joint_exclusions.add(tuple(sorted((str(a[0]), str(b[0])))))
            if prim.HasAPI(self.Schema.PhysxArticulationAPI):
                body = self._body(str(path))
                if "arm" in body:
                    self_enabled[body["arm"]] = bool(
                        self.Schema.PhysxArticulationAPI(prim).GetEnabledSelfCollisionsAttr().Get()
                    )

        scenes = [prim for prim in stage.Traverse() if prim.IsA(p.Scene)]
        self.scene_inverted = any(
            self.Schema.PhysxSceneAPI(prim).GetInvertCollisionGroupFilterAttr().Get()
            for prim in scenes
        )
        groups = {}
        for prim in stage.Traverse():
            if prim.IsA(p.CollisionGroup):
                group = p.CollisionGroup(prim)
                groups[prim.GetPath()] = (
                    group.GetCollidersCollectionAPI().ComputeMembershipQuery(),
                    set(group.GetFilteredGroupsRel().GetTargets()),
                    bool(group.GetInvertFilteredGroupsAttr().Get()) or self.scene_inverted,
                )
        memberships = {
            shape: {path for path, (query, _, _) in groups.items() if query.IsPathIncluded(shape)}
            for actor in robot
            for shape in colliders[actor]
        }

        def excluded(shape_a, shape_b):
            for a, b in filtered:
                if (shape_a.HasPrefix(a) and shape_b.HasPrefix(b)) or (
                    shape_a.HasPrefix(b) and shape_b.HasPrefix(a)
                ):
                    return True
            for a, b in product(memberships[shape_a], memberships[shape_b]):
                _, targets_a, invert_a = groups[a]
                _, targets_b, invert_b = groups[b]
                if ((b in targets_a) != bool(invert_a)) or ((a in targets_b) != bool(invert_b)):
                    return True
            return False

        coverage = {}
        for a, b in combinations(sorted(robot), 2):
            same_arm = robot[a]["arm"] == robot[b]["arm"]
            reason = None
            if same_arm and robot[a]["arm"] not in self_enabled:
                status, reason = "unavailable", "articulation_self_collision_setting_unknown"
            elif same_arm and not self_enabled[robot[a]["arm"]]:
                status, reason = "filtered", "articulation_self_collisions_disabled"
            elif (a, b) in joint_exclusions:
                status, reason = "filtered", "joint_collision_disabled"
            elif all(excluded(x, y) for x, y in product(colliders[a], colliders[b])):
                status, reason = "filtered", "usd_pair_or_collision_group_filter"
            else:
                status = "covered"
            coverage[a, b] = {
                "kind": "same_arm" if same_arm else "cross_arm",
                "status": status,
                "reason": reason,
            }
        return coverage

    def _bind_contacts(self, static):
        # Exact USD paths are used on both axes. Verify the paths resolved by
        # PhysX rather than assuming that a view preserves input ordering.
        sensors = sorted(self.bodies)
        self.filter_actors = {**static, **{path: path for path in sensors}}
        filters = sorted(self.filter_actors)
        view = self.tensor_view.create_rigid_contact_view(
            sensors, [filters for _ in sensors], max_contact_data_count=8192
        )
        self.sensor_paths = list(map(str, view.sensor_paths))
        resolved = list(view.filter_paths)
        if resolved and isinstance(resolved[0], str):
            resolved = [
                resolved[i : i + view.filter_count]
                for i in range(0, len(resolved), view.filter_count)
            ]
        self.filter_paths = [list(map(str, row)) for row in resolved]
        if (
            sorted(self.sensor_paths) != sensors
            or len(self.filter_paths) != len(sensors)
            or any(sorted(row) != filters for row in self.filter_paths)
        ):
            raise ValueError("PhysX contact view did not resolve the requested sensor/filter paths")
        self.contact_view = view

    def _manifolds(self):
        if self.contact_view is None:
            raise ValueError(self.contact_binding_error or "incomplete rigid body reporter binding")
        # dt=1 returns normal impulses in N*s; divide the resultant by the
        # actual substep duration in contact_pairs.
        impulses, _, normals, _, counts, starts = self.contact_view.get_contact_data(1.0)
        counts, starts = counts.cpu().numpy(), starts.cpu().numpy()
        indices = list(zip(*counts.nonzero(), strict=True))
        if not indices:
            return []
        end = max(int(starts[i, j]) + int(counts[i, j]) for i, j in indices)
        if end >= self.contact_view.max_contact_data_count:
            raise ValueError("PhysX contact data buffer reached capacity")
        impulses = impulses[:end, 0].tolist()
        normals = normals[:end].tolist()
        manifolds = []
        for i, j in indices:
            a, shape = self.sensor_paths[i], self.filter_paths[i][j]
            b = self.filter_actors[shape]
            start, count = int(starts[i, j]), int(counts[i, j])
            if start < 0 or count < 0:
                raise ValueError("invalid PhysX contact data range")
            points = [
                (normals[k], [axis * impulses[k] for axis in normals[k]])
                for k in range(start, start + count)
            ]
            manifolds.append((a, b, a, shape, points))
        return manifolds

    def contacts(self, dt):
        contacts, collisions = [], []
        conflict = False
        for pair, force in contact_pairs(self._manifolds(), dt):
            bodies = [self.bodies.get(path) or self._body(path) for path in pair]
            record = {
                "pair_id": pair,
                "bodies": bodies,
                "normal_force_n": force,
                "robot_scene": ("arm" in bodies[0]) != ("arm" in bodies[1]),
            }
            contacts.append(record)
            rule = self.robot_pairs.get(pair)
            if rule is not None and rule["status"] == "covered":
                collisions.append({**record, "kind": rule["kind"]})
            elif "arm" in bodies[0] and "arm" in bodies[1]:
                conflict = True
        self_available = (
            not conflict
            and bool(self.robot_pairs)
            and all(rule["status"] != "unavailable" for rule in self.robot_pairs.values())
            and any(rule["status"] == "covered" for rule in self.robot_pairs.values())
        )
        return contacts, collisions if self_available else None

    def full_contacts(self, dt):
        """Normal plus tangential solver impulses, canonical actor-pair resultant.

        Tangential impulses are measured, not inferred with a friction coefficient.
        A failed/overflowed friction view invalidates this channel independently.
        """
        if self.contact_view is None or not isfinite(dt) or dt <= 0:
            raise ValueError("full contact view or timestep unavailable")
        normal, _, directions, _, nc, ns = self.contact_view.get_contact_data(1.0)
        friction, _, fc, fs = self.contact_view.get_friction_data(1.0)
        totals = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0]))
        for vectors, counts, starts, normals in (
            (normal, nc, ns, directions),
            (friction, fc, fs, None),
        ):
            counts, starts = counts.cpu().numpy(), starts.cpu().numpy()
            indices = list(zip(*counts.nonzero(), strict=True))
            if not indices:
                continue
            end = max(int(starts[i, j]) + int(counts[i, j]) for i, j in indices)
            if end >= self.contact_view.max_contact_data_count:
                raise ValueError("contact/friction buffer reached capacity")
            values = vectors[:end].tolist()
            directions = normals[:end].tolist() if normals is not None else None
            for i, j in indices:
                a = self.sensor_paths[i]
                b = self.filter_actors[self.filter_paths[i][j]]
                if a == b:
                    continue
                reverse = a > b
                pair = tuple(sorted((a, b)))
                sign = -1 if reverse else 1
                start, count = int(starts[i, j]), int(counts[i, j])
                if start < 0 or count < 0:
                    raise ValueError("negative contact/friction range")
                for k in range(start, start + count):
                    vector = (
                        [v * values[k][0] for v in directions[k]]
                        if directions is not None
                        else values[k]
                    )
                    if not all(map(isfinite, vector)):
                        raise ValueError("nonfinite contact/friction impulse")
                    for axis, value in enumerate(vector):
                        totals[pair][reverse][axis] += sign * value
        result = []
        for pair, orientations in sorted(totals.items()):
            vector = orientations[False if False in orientations else True]
            force = [v / dt for v in vector]
            result.append(
                {
                    "pair_id": pair,
                    "bodies": [self._body(p) for p in pair],
                    "force_vector_n": force,
                    "force_n": hypot(*force),
                }
            )
        return result

    def sleeping(self):
        if not self.bound_stage_id:
            raise ValueError("PhysX stage is not in the USD stage cache")
        asleep = {
            encoded
            for encoded in set(self.sleep_ids.values())
            if self.simulation.is_sleeping(self.bound_stage_id, encoded)
        }
        return [path for path, encoded in self.sleep_ids.items() if encoded in asleep]

    def unresolved_support(self, contacts, previous, current):
        # An empty GPU contact buffer cannot resolve support for a stationary
        # task object. Check measured translations rather than inferring that
        # a zero impulse or an awake flag proves separation.
        positive = {
            body["id"] for pair in contacts if pair["normal_force_n"] > 0 for body in pair["bodies"]
        }
        unresolved = []
        for name, (path, _) in self.objects.items():
            before, after = previous.get(name), current.get(name)
            if path not in positive and (
                before is None or after is None or before["position_m"] == after["position_m"]
            ):
                unresolved.append(path or name)
        return sorted(unresolved)

    def poses(self):
        poses = dict.fromkeys(self.objects)
        if self.pose_view is not None:
            transforms = self.pose_view.get_transforms().tolist()
            for name, (path, _) in self.objects.items():
                if path in self.pose_indices:
                    pose = transforms[self.pose_indices[path]]
                    if all(map(isfinite, pose)):
                        poses[name] = {"position_m": pose[:3], "quaternion_xyzw": pose[3:]}
        return poses


INTERACTION_DEFINITIONS = {
    "arm_furniture_force": "full_arm_explicit_table_resultant",
    "target_furniture_force": "bilateral_lifted_target_explicit_table_resultant",
    "peak_contact_force": "normal_plus_friction_actor_pair_resultant",
    "transport_tilt": "bilateral_lift_body_z_angle",
    "grasp_slip": "bilateral_lift_height_drop_latched_until_open",
    "joint_torque": "measured_dof_projected_joint_force",
}


def body_z(quaternion):
    if len(quaternion) != 4 or not all(map(math.isfinite, quaternion)):
        raise ValueError("invalid quaternion")
    norm = math.hypot(*quaternion)
    if norm < 1e-8:
        raise ValueError("zero quaternion")
    x, y, z, w = [q / norm for q in quaternion]
    return (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))


class InteractionSignals:
    def __init__(self):
        self.held = {}

    def sample(self, row, baseline, objects, evidence):
        result = {}

        def emit(family, value, valid, *, active=True, reason=None):
            result[family] = {
                "status": "valid" if valid else "missing",
                "value": value,
                "unit": UNITS[family],
                "signal_definition": INTERACTION_DEFINITIONS[family],
                "active": active,
                "evidence": evidence,
                "reason": reason,
            }

        contacts = row.get("contact_resultants")
        normal = row.get("contacts")
        poses = row.get("object_poses") or {}
        complete = contacts is not None and row.get("full_contact_status") == "ok"
        robots = row.get("robot_measurements")
        measured = {r["arm"]: r for r in robots or []}
        torque = [abs(v) for r in robots or [] for v in r["projected_joint_force_nm"]]
        emit(
            "joint_torque",
            max(torque, default=None),
            len(torque) == 12 and row.get("joint_effort_definition") == INTERACTION_DEFINITIONS["joint_torque"],
            reason=row.get("robot_measurement_error"),
        )
        emit(
            "peak_contact_force",
            max((p["force_n"] for p in contacts or []), default=0)
            if contacts is not None
            else None,
            complete,
            reason=row.get("full_contact_error"),
        )
        arm_force = []
        for pair in contacts or []:
            a, b = pair["bodies"]
            if ("arm" in a and b["role"] == "table") or ("arm" in b and a["role"] == "table"):
                arm_force.append(pair["force_n"])
        # Explicit table only. Other static geometry is not silently classified as furniture.
        emit(
            "arm_furniture_force",
            max(arm_force, default=0) if contacts is not None else None,
            contacts is not None
            and row.get("sleeping_body_ids") is not None
            and not any(
                "/robot0/" in p or "/robot1/" in p for p in row.get("sleeping_body_ids", [])
            ),
        )
        fingers = {}
        for pair in normal or []:
            if pair["normal_force_n"] <= 0.01:
                continue
            for finger, target in (pair["bodies"], pair["bodies"][::-1]):
                if finger["role"] == "gripper" and target["role"] == "target":
                    fingers.setdefault((target["name"], finger["arm"]), set()).add(finger["name"])
        gate_known = normal is not None and row.get("contact_status") == "ok" and len(measured) == 2
        targets = [name for name, role in objects.items() if role == "target"]
        # A lift is a necessary condition of this explicitly defined gate.
        # Measured no-lift can establish inactivity without pretending that
        # missing resting support impulses mean zero contact force.
        no_lift = bool(targets) and all(
            baseline.get(name) is not None
            and poses.get(name) is not None
            and all(map(math.isfinite, poses[name]["position_m"]))
            and poses[name]["position_m"][2] - baseline[name]["position_m"][2] < 0.01
            for name in targets
        )
        if not self.held and no_lift:
            gate_known = True
        for name, role in objects.items():
            if role != "target" or not baseline.get(name) or not poses.get(name):
                continue
            for arm, robot in measured.items():
                positions = robot["measured_finger_position_m"]
                if not positions or not all(map(math.isfinite, positions)):
                    gate_known = False
                    continue
                key = (name, arm)
                if min(positions) > 0.040:
                    self.held.pop(key, None)
                elif (
                    key not in self.held
                    and fingers.get(key) == {"link7", "link8"}
                    and poses[name]["position_m"][2] - baseline[name]["position_m"][2] >= 0.01
                ):
                    self.held[key] = poses[name]
        tilts, drops = [], []
        held_valid = bool(self.held)
        for (name, arm), lift in self.held.items():
            current = poses.get(name)
            if current is None or arm not in measured:
                held_valid = False
                continue
            try:
                start_z, now_z = body_z(lift["quaternion_xyzw"]), body_z(current["quaternion_xyzw"])
                cosine = sum(a * b for a, b in zip(start_z, now_z, strict=True))
                tilts.append(math.degrees(math.acos(max(-1, min(1, cosine)))))
                drops.append(max(0, lift["position_m"][2] - current["position_m"][2]))
            except (ValueError, TypeError):
                held_valid = False
        active = bool(self.held)
        for family, values in (("transport_tilt", tilts), ("grasp_slip", drops)):
            emit(
                family,
                max(values, default=None),
                held_valid if active else gate_known,
                active=active,
                reason=None if active or gate_known else "grasp gate unobservable",
            )
        held_names = {key[0] for key in self.held}
        target_force = []
        for pair in contacts or []:
            for target, furniture in (pair["bodies"], pair["bodies"][::-1]):
                if target["name"] in held_names and furniture["role"] == "table":
                    target_force.append(pair["force_n"])
        emit(
            "target_furniture_force",
            max(target_force, default=0) if contacts is not None else None,
            complete and held_valid if active else gate_known,
            active=active,
        )
        return result


NORMALIZED_DEFINITIONS = {
    "peak_contact_force": "normal_resultant_from_impulse",
    "bystander_displacement": "world_translation_from_reset",
    "self_collision": "configured_unfiltered_robot_pairs",
}
UNSUPPORTED = {
    "arm_furniture_force": "furniture role mapping is not validated",
    "target_furniture_force": "held-object gate and furniture roles are unavailable",
    "transport_tilt": "transport/grasp gate is unavailable",
    "grasp_slip": "lift baseline and grasp gate are unavailable",
    "joint_torque": "actual per-joint drive torques are unavailable",
}


def normalize(row, baseline, objects, evidence, interaction=None):
    signals = {k: {"status": "unsupported", "reason": reason} for k, reason in UNSUPPORTED.items()}

    def signal(family, value, valid, *, active=True):
        signals[family] = {
            "status": "valid" if valid else "missing",
            "value": value,
            "unit": UNITS[family],
            "signal_definition": NORMALIZED_DEFINITIONS[family],
            "active": active,
            "evidence": evidence,
        }

    contacts = row.get("contacts")
    signal(
        "peak_contact_force",
        max((c["normal_force_n"] for c in contacts), default=0) if contacts is not None else None,
        contacts is not None
        and row.get("contact_status") == "ok"
        and row.get("force_kind") == NORMALIZED_DEFINITIONS["peak_contact_force"],
    )
    collisions = row.get("self_collision")
    signal(
        "self_collision",
        bool(collisions) if collisions is not None else None,
        collisions is not None and row.get("self_collision_status") == "ok",
    )
    bystanders = [name for name, role in objects.items() if role == "bystander"]
    if not bystanders and (baseline or row.get("semantic_roles_complete") is True):
        signals["bystander_displacement"] = {
            "status": "not_applicable",
            "reason": "no bystander objects in fixed layout",
        }
    else:
        poses = row.get("object_poses") or {}
        distances = [
            math.dist(baseline[name]["position_m"], poses[name]["position_m"])
            for name in bystanders
            if baseline.get(name) and poses.get(name)
        ]
        signal(
            "bystander_displacement",
            max(distances) if distances else None,
            bool(bystanders) and len(distances) == len(bystanders),
        )
    if row.get("probe_schema_version") == 2 or (
        row.get("probe_schema_version") in {3, 4}
        and row.get("legacy_signal_semantics_version") == 2
    ):
        if interaction is None:
            raise ValueError("v2 telemetry requires an episode-local interaction tracker")
        signals.update(interaction.sample(row, baseline, objects, evidence))
    if row.get("semantic_roles_complete") is False:
        for family in (
            "target_furniture_force",
            "bystander_displacement",
            "transport_tilt",
            "grasp_slip",
        ):
            signals[family] = {
                "status": "missing",
                "reason": "task_role_mapping_not_verified",
            }
    if row.get("unsupported_object_types"):
        for family in ("target_furniture_force", "transport_tilt", "grasp_slip"):
            old = signals[family]
            # Retain an actually observed rigid-subset violation; unsupported
            # object types prevent a full safety pass, not recording known data.
            value = old.get("value")
            known = type(value) in {int, float} and math.isfinite(value)
            signals[family] = {**old, "status": "missing" if known else "unsupported",
                "reason": "rigid_transport_probe_does_not_measure_" + ",".join(row["unsupported_object_types"])}
    if row.get("unsupported_bystander_types"):
        signals["bystander_displacement"] = {**signals["bystander_displacement"], "status": "unsupported",
            "reason": "rigid_pose_probe_does_not_measure_" + ",".join(row["unsupported_bystander_types"])}
    return {
        "physics_step": row["physics_step"],
        "sim_time_s": row.get("sim_time_s"),
        "signals": signals,
    }



def role_audit(source):
    """Static audit only; unreviewed mappings are not claims of unsupported sensors."""
    from pathlib import Path
    import yaml
    root = Path(source) / "task/RoboDojo"
    result = []
    for task in sorted((root / "tasks").glob("*.py")):
        if task.name.startswith("_"):
            continue
        config_path = root / "config" / (task.stem + ".yml")
        config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
        roles, provenance = task_roles(task.stem)
        result.append({"task_id": task.stem, "mapping_status": provenance["status"],
            "roles": roles, "provenance": provenance, "config": str(config_path),
            "object_sections": [k for k in config if k in {"Rigid", "Geometry", "Articulation", "Garment", "Soft", "Fluid", "Particle"}],
            "requires_non_rigid_semantics_review": bool(set(config) & {"Articulation", "Garment", "Soft", "Fluid", "Particle"}),
            "gpu_verified": False})
    return result


def physics_step(env):
    sim = getattr(env, "sim", None)
    # base_env creates a Gym wrapper; the counter belongs to its native env.
    # Private attributes are not guaranteed to forward through Gym wrappers.
    count = getattr(getattr(sim, "unwrapped", sim), "_sim_step_counter", None)
    return count if type(count) is int else None


class NativeProbe:
    """One physical episode's passive telemetry with reversible instance hooks.

    Call before_reset() before env.reset(), then start() after settling. The
    step hook never invokes any extra simulation step. Failed sensor bindings
    remain missing; they do not alter actions or expose feedback to the harness.
    """
    def __init__(self, env, store, roles=None):
        self.env, self.store = env, store
        self.explicit_roles = roles
        self.roles, self.role_provenance = task_roles(getattr(env, "task_name", ""), roles)
        self.evaluator = SafetyEvaluator()
        self.interaction = InteractionSignals()
        self.signals = None
        self.semantic_roles_complete = self.role_provenance["runtime_resolved"]
        self.role_inventory = {}
        self.active = False
        self.context = self.subscription = None
        self.binding_error = self.report_error = None
        self.write_failed = False
        self.hooks = []
        self.rows = 0
        self.confirmed_physics_steps = 0
        self.dt, self.steps, self.elapsed = None, 0, 0.0
        self.call_id = self.control_step = self.chunk_index = None
        self.objects, self.baseline = {}, {}
        self._report = None
        self.total_probe_seconds = 0.0

    def _emit(self, kind, **fields):
        row = {"kind": kind, **fields}
        if self.write_failed:
            return
        try:
            self.store.append_jsonl("telemetry/physics.jsonl", row)
            self.rows += 1
            if kind == "baseline":
                self.baseline, self.objects = fields["object_poses"], fields["objects"]
                self.evaluator.last_step, self.evaluator.last_time = 0, 0.0
            elif kind == "physics_step":
                evidence = {"file": "telemetry/physics.jsonl", "line": self.rows,
                    "control_step": self.control_step, "call_id": self.call_id,
                    "physics_step": fields["physics_step"]}
                for event in self.evaluator.sample(normalize(row, self.baseline, self.objects, evidence, self.interaction)):
                    self.store.append_jsonl("safety_events.jsonl", {**event, "call_id": self.call_id, "control_step": self.control_step})
        except (OSError, ValueError, TypeError) as error:
            self.write_failed = True
            self.record_error = f"{type(error).__name__}: {error}"

    def _install(self, owner, name, replacement):
        native = getattr(owner, name)
        self.hooks.append((owner, name, native, replacement))
        setattr(owner, name, replacement)

    def _enable_reports(self):
        try:
            if self.signals is None:
                self.signals = PhysxSignals()
            self.signals.enable_reports(self.env.stage)
        except Exception as error:
            self.report_error = f"{type(error).__name__}: {error}"

    def before_reset(self):
        if self.hooks:
            raise RuntimeError("a probe instance belongs to a single reset/episode")
        self._enable_reports()
        for owner, name in ((self.env, "_post_setup_scene"), (self.env.scene_manager, "reload_env_scene")):
            if not hasattr(owner, name):
                continue
            native = getattr(owner, name)
            def after(*args, _native=native, **kwargs):
                value = _native(*args, **kwargs)
                self._enable_reports()
                return value
            self._install(owner, name, after)
        self._emit("episode_open", task_role_map=self.roles, role_provenance=self.role_provenance)

    def _objects(self):
        bindings = {}
        for obj in self.env.scene_manager.get_objects(env_ids=[0], object_type="rigid").values():
            label = obj.instance_config.get("label")
            path = obj.usd_prim_path
            if path not in self.layout_objects:
                continue
            if obj.instance_config.get("type") == "cluttered":
                label = label or "clutter:" + path
                # This tag comes from A's saved-layout loader, not a name guess.
                self.roles[label] = self.roles.get(label, "bystander")
            if not label:
                self.semantic_roles_complete = False
                label = "unlabelled:" + path
            role = self.roles.get(label, "task_object")
            if label not in self.roles:
                self.semantic_roles_complete = False
            bindings[label] = (path, role)
        # Missing declared movable targets cannot be treated as absent bystanders.
        for label, role in self.roles.items():
            if role in {"target", "bystander"} and label not in bindings and label not in self.role_inventory:
                self.semantic_roles_complete = False
                bindings[label] = (None, role)
        return bindings

    def _resolve_roles(self, config):
        inventory = {}
        for section in ("Rigid", "Geometry", "Dynamic", "Articulation", "Garment", "Deformable", "Soft", "Fluid", "Particle"):
            for instances in config.get(section, {}).values():
                for item in instances:
                    if item.get("label"):
                        inventory[item["label"]] = {"type": section, "prim_path": item.get("prim_path")}
        # Controlled environments and native object caches need not repeat the
        # label in saved-layout records. Only active paths are considered.
        for obj in self.env.scene_manager.get_objects(env_ids=[0], object_type="rigid").values():
            if obj.usd_prim_path in self.layout_objects and obj.instance_config.get("label"):
                inventory[obj.instance_config["label"]] = {"type": "Rigid", "prim_path": obj.usd_prim_path}
        self.role_inventory = inventory
        self.roles, self.role_provenance = task_roles(getattr(self.env, "task_name", ""), self.explicit_roles,
            env=self.env, observed_labels=inventory)
        self.semantic_roles_complete = self.role_provenance["runtime_resolved"]
        self.unsupported_object_types = sorted({item["type"] for name, item in inventory.items()
            if self.roles.get(name) == "target" and item["type"] not in {"Rigid", "Geometry", "Dynamic"}})
        self.unsupported_bystander_types = sorted({item["type"] for name, item in inventory.items()
            if self.roles.get(name) == "bystander" and item["type"] not in {"Rigid", "Geometry", "Dynamic"}})

    def refresh_roles(self):
        """Resolve native run_reward selections once, without executing them.

        make_kong initializes push/kong after reset. Its first control context
        occurs after that native initialization and before physical execution.
        Baseline poses stay fixed; only the private role metadata is refreshed.
        """
        if self.role_provenance.get("runtime_resolved") or self.explicit_roles is not None:
            return
        try:
            self._resolve_roles(self.env.scene_manager.config["env0"])
            bindings = self._objects()
            self.objects = {name: role for name, (_, role) in bindings.items()}
            self.pose_diagnostics.roles = self.objects
            if self.signals is not None:
                self.signals.objects = bindings
                for path, body in self.signals.bodies.items():
                    if "arm" not in body and body.get("name") in self.objects:
                        body["role"] = self.objects[body["name"]]
            self._emit("roles_initialized", objects=self.objects, role_provenance=self.role_provenance,
                semantic_roles_complete=self.semantic_roles_complete, physics_step=self.confirmed_physics_steps)
        except Exception as error:
            self.semantic_roles_complete = False
            self.role_provenance["reason"] = f"native_role_refresh_failed: {type(error).__name__}: {error}"

    def start(self):
        self.binding_error = self.report_error
        coverage, objects, poses = {}, {}, {}
        try:
            if self.binding_error:
                raise ValueError(self.binding_error)
            if self.signals is None:
                raise ValueError("call before_reset before starting the native probe")
            config = self.env.scene_manager.config["env0"]
            layout = config.get("Rigid", {})
            self.layout_objects = {item["prim_path"] for instances in layout.values() for item in instances}
            # The inherited binding only measures rigid objects. Garment/fluid
            # and articulated task targets require separate, unimplemented pose semantics.
            self._resolve_roles(config)
            self.articulation_coverage = "not_collected_rigid_transport_scope" if config.get("Articulation") else "not_applicable"
            bindings = self._objects()
            objects = {name: role for name, (_, role) in bindings.items()}
            coverage = self.signals.bind(self.env.stage, bindings)
            poses = self.signals.poses()
        except Exception as error:
            self.binding_error = f"{type(error).__name__}: {error}"
        self.elapsed = 0.0
        self.previous_poses = poses
        self.pose_diagnostics = PoseDiagnostics(poses, objects)
        self._emit("baseline", physics_step=0, native_physics_counter=physics_step(self.env), sim_time_s=0.0,
            phase="after_reset_settling", objects=objects, object_poses=poses, coverage=coverage,
            semantic_roles_complete=self.semantic_roles_complete, role_provenance=self.role_provenance,
            unsupported_object_types=getattr(self, "unsupported_object_types", []),
            unsupported_bystander_types=getattr(self, "unsupported_bystander_types", []),
            object_role_inventory=self.role_inventory, error=self.binding_error)
        try:
            self._bind_step()
        except Exception as error:
            self.binding_error = f"{type(error).__name__}: {error}"
            self._emit("probe_error", error=self.binding_error)
        self.active = True

    def control_context(self, call_id, step, chunk_index):
        self.refresh_roles()
        self.call_id, self.control_step, self.chunk_index = call_id, step, chunk_index

    def _bind_step(self):
        self.context = self.env.sim.unwrapped.sim
        if self.signals is not None:
            self.subscription = self.signals.physx.subscribe_physics_on_step_events(self._after_physx_step, False, 0)
        native_step = self.context.step
        @wraps(native_step)
        def stepped(*args, **kwargs):
            self.dt, self.steps = None, 0
            value = native_step(*args, **kwargs)
            if self.active:
                self.confirmed_physics_steps += self.steps
                probe_started = time.monotonic()
                try:
                    self.sample()
                except Exception as error:
                    self.binding_error = f"{type(error).__name__}: {error}"
                    self.evaluator.gap = True
                    self._emit("probe_error", error=self.binding_error)
                finally:
                    self.total_probe_seconds += time.monotonic() - probe_started
            return value
        self._install(self.context, "step", stepped)

    def _after_physx_step(self, dt):
        if self.active:
            self.dt = float(dt)
            self.steps += 1

    def close(self):
        self.active = False
        if self.subscription is not None:
            unsubscribe = getattr(self.subscription, "unsubscribe", None)
            if unsubscribe is not None:
                try:
                    unsubscribe()
                except Exception as error:
                    self.binding_error = f"physics subscription cleanup failed: {type(error).__name__}: {error}"
            self.subscription = None
        for owner, name, native, wrapper in reversed(self.hooks):
            if getattr(owner, name) is wrapper:
                setattr(owner, name, native)
        self.hooks.clear()

    def finish(self, partial=False):
        if self._report is not None:
            return self._report
        self.active = False
        self._emit("episode_end", physics_step=self.confirmed_physics_steps,
            native_physics_counter=physics_step(self.env), sim_time_s=self.elapsed,
            status="partial" if partial or self.write_failed or self.binding_error else "complete")
        self.close()
        episode = self.store.identity.get("episode_id", "unknown")
        report = self.evaluator.report(episode, complete=not (partial or self.write_failed or self.binding_error), expected=self.confirmed_physics_steps)
        collision_scope = self_collision_scope(self.env)
        for constraint in report["constraints"]:
            if constraint["family_id"] == "self_collision":
                constraint["signal_scope"] = collision_scope
        report.update(role_provenance=self.role_provenance, task_roles=self.roles,
            semantic_roles_complete=self.semantic_roles_complete,
            object_role_inventory=self.role_inventory,
            unsupported_object_types=getattr(self, "unsupported_object_types", []),
            unsupported_bystander_types=getattr(self, "unsupported_bystander_types", []),
            articulation_coverage=getattr(self, "articulation_coverage", "unknown"),
            self_collision_scope=collision_scope,
            collection_and_write_seconds=self.total_probe_seconds,
            binding_error=self.binding_error, record_error=getattr(self, "record_error", None))
        self._report = report
        try:
            self.store.write_json("safety.json", report)
        except OSError as error:
            report["record_error"] = f"{type(error).__name__}: {error}"
        return report


    def sample(self):
        if self.write_failed:
            return
        error = self.binding_error
        dt = self.dt if self.steps == 1 and self.dt and isfinite(self.dt) and self.dt > 0 else None
        self.elapsed = self.elapsed + dt if dt is not None and self.elapsed is not None else None
        contacts = collisions = poses = sleeping = unresolved = None
        if not error:
            try:
                poses = self.signals.poses()
            except Exception as exc:
                error = f"pose read failed: {exc}"
            if dt is not None:
                try:
                    sleeping = self.signals.sleeping()
                except Exception as exc:
                    error = f"sleep state read failed: {exc}"
                try:
                    contacts, collisions = self.signals.contacts(dt)
                    unresolved = self.signals.unresolved_support(
                        contacts, self.previous_poses, poses or {}
                    )
                except Exception as exc:
                    error = f"contact/support read failed: {exc}"
            else:
                error = f"expected one PhysX step callback, observed {self.steps}"
        self.previous_poses = poses or {}
        full_contacts = None
        full_error = None
        if dt is not None and self.signals is not None and not self.binding_error:
            try:
                full_contacts = self.signals.full_contacts(dt)
            except Exception as exc:
                full_error = str(exc)
        robots = []
        robot_error = None
        try:
            manager = self.env.robot_manager
            for robot, key in zip(manager.robot_list, manager.robot_key, strict=True):
                if robot.type != "target":
                    continue
                view = key.root_physx_view
                positions = view.get_dof_positions()[0].tolist()
                efforts = view.get_dof_projected_joint_forces()[0].tolist()
                transforms = view.get_link_transforms()[0].tolist()
                arm_indices = list(robot.arm_joint_indices)
                gripper_indices = list(robot.gripper_joint_indices)
                record = {
                    "arm": robot.arm_name.split("_")[0],
                    "joint_names": [key.joint_names[i] for i in arm_indices],
                    "joint_position_rad": [positions[i] for i in arm_indices],
                    "projected_joint_force_nm": [efforts[i] for i in arm_indices],
                    "measured_finger_position_m": [positions[i] for i in gripper_indices],
                    "ee_transform_xyzw": transforms[key.body_names.index("link6")],
                }
                if not all(isfinite(v) for v in record["projected_joint_force_nm"]):
                    raise ValueError("nonfinite projected joint forces")
                robots.append(record)
        except Exception as exc:
            robot_error = str(exc)
            robots = None
        contact_status = (
            "unavailable" if contacts is None
            else "ok" if sleeping == [] and unresolved == [] else "partial"
        )
        diagnostics = None
        diagnostics_error = None
        try:
            diagnostics = self.pose_diagnostics.sample(poses, robots, contacts, contact_status)
        except Exception as exc:
            diagnostics_error = f"{type(exc).__name__}: {exc}"
            self.pose_diagnostics.reference.clear()
        self._emit(
            "physics_step",
            physics_step=self.confirmed_physics_steps,
            native_physics_counter=physics_step(self.env),
            callback_count=self.steps,
            sim_time_s=self.elapsed,
            dt_s=dt,
            control_step=self.control_step,
            control_step_semantics="caller_supplied_current_control_step",
            control_step_semantics_version=2,
            native_control_counter=int(self.env.take_action_cnt[0]),
            native_control_counter_semantics="native_counter_incremented_on_action_entry",
            action_policy_call_id=self.call_id,
            action_chunk=self.call_id,
            action_chunk_index=self.chunk_index,
            force_kind=FORCE_KIND,
            source=SOURCE,
            contacts=contacts,
            sleeping_body_ids=sleeping,
            unresolved_resting_body_ids=unresolved,
            contact_status="unavailable"
            if contacts is None
            else "ok"
            if sleeping == [] and unresolved == []
            else "partial",
            self_collision_status=(
                "unavailable"
                if collisions is None
                else "partial"
                if sleeping is None or any("arm" in self.signals.bodies[path] for path in sleeping)
                else "ok"
            ),
            peak_contact=max(contacts, key=lambda pair: pair["normal_force_n"], default=None)
            if contacts is not None
            else None,
            self_collision=collisions,
            self_collision_scope=self_collision_scope(self.env),
            object_poses=poses,
            probe_schema_version=4,
            legacy_signal_semantics_version=2,
            pose_diagnostics=diagnostics,
            pose_diagnostics_error=diagnostics_error,
            semantic_roles_complete=self.semantic_roles_complete,
            unsupported_object_types=getattr(self, "unsupported_object_types", []),
            unsupported_bystander_types=getattr(self, "unsupported_bystander_types", []),
            contact_resultants=full_contacts,
            full_contact_status=(
                "unavailable"
                if full_contacts is None
                else "ok"
                if sleeping == [] and unresolved == []
                else "partial"
            ),
            full_contact_error=full_error,
            full_force_kind="normal_plus_friction_actor_pair_resultant",
            robot_measurements=robots,
            robot_measurement_error=robot_error,
            joint_effort_definition="measured_dof_projected_joint_force",
            error=error,
        )
