from ._base_task import *
from .put_bottle_in_shelf import Task as BasePutBottleInShelfTask, TaskCfg as BasePutBottleInShelfTaskCfg
import numpy as np


@configclass
class TaskCfg(BasePutBottleInShelfTaskCfg):
    pass


class Task(BasePutBottleInShelfTask):
    variant_name = "D1"

    shelf_xy_range = np.array([0.08, 0.03], dtype=float)
    shelf_yaw_range = np.deg2rad(5.0)

    # bottle 仍然围绕 D0 的世界系初始位姿采样，但增加 shelf-local 的接受约束，
    # 排除“离 shelf 太近”或“横向偏得太夸张”的退化初始状态。
    bottle_xy_range = np.array([0.015, 0.05], dtype=float)
    bottle_yaw_range = np.deg2rad(8.0)

    preferred_min_forward_clearance = 0.37
    preferred_max_forward_clearance = 0.47
    safe_min_forward_clearance = 0.33
    safe_max_forward_clearance = 0.49
    max_lateral_offset_in_shelf_frame = 0.06
    max_reset_sample_attempts = 128
    hidden_grasp_height_min = 0.11
    hidden_grasp_height_max = 0.12
    hidden_grasp_pitch_noise = np.pi / 18

    def _sample_scalar(self, low: float, high: float) -> float:
        return float(self.rng.uniform(low, high))

    def _sample_symmetric(self, magnitude: float) -> float:
        return float(self.rng.uniform(-magnitude, magnitude))

    def _sample_shelf_pose(self) -> tuple[Pose, np.ndarray, float]:
        shelf_xy = self.rng.uniform(-self.shelf_xy_range, self.shelf_xy_range)
        shelf_yaw = self._sample_symmetric(self.shelf_yaw_range)
        shelf_pose = (
            Pose([0.9 + float(shelf_xy[0]), float(shelf_xy[1]), 0.01], [1, 0, 0, 0])
            .add_rotation([0.0, 0.0, shelf_yaw])
        )
        return shelf_pose, shelf_xy, shelf_yaw

    def _measure_bottle_pose(self, bottle_pose: Pose, shelf_pose: Pose) -> tuple[float, float]:
        rel_pose = bottle_pose.rebase(shelf_pose)
        forward_clearance = -float(rel_pose.p[0])
        lateral_offset = float(rel_pose.p[1])
        return forward_clearance, lateral_offset

    def _is_preferred_bottle_pose(self, forward_clearance: float, lateral_offset: float) -> bool:
        return (
            self.preferred_min_forward_clearance
            <= forward_clearance
            <= self.preferred_max_forward_clearance
            and abs(lateral_offset) <= self.max_lateral_offset_in_shelf_frame
        )

    def _is_safe_fallback_bottle_pose(self, forward_clearance: float, lateral_offset: float) -> bool:
        return (
            self.safe_min_forward_clearance <= forward_clearance <= self.safe_max_forward_clearance
            and abs(lateral_offset) <= self.max_lateral_offset_in_shelf_frame
        )

    def _safe_fallback_score(self, forward_clearance: float, lateral_offset: float) -> float:
        if forward_clearance < self.preferred_min_forward_clearance:
            forward_penalty = self.preferred_min_forward_clearance - forward_clearance
        elif forward_clearance > self.preferred_max_forward_clearance:
            forward_penalty = forward_clearance - self.preferred_max_forward_clearance
        else:
            forward_penalty = 0.0
        return 2.0 * forward_penalty + abs(lateral_offset)

    def _sample_bottle_pose(
        self, shelf_pose: Pose
    ) -> tuple[Pose, np.ndarray, float, float, float, int, int, str, bool]:
        best_safe_candidate = None
        best_safe_score = None

        for attempt_idx in range(1, self.max_reset_sample_attempts + 1):
            bottle_xy = self.rng.uniform(-self.bottle_xy_range, self.bottle_xy_range)
            bottle_yaw = self._sample_symmetric(self.bottle_yaw_range)
            bottle_pose = (
                Pose([0.5 + float(bottle_xy[0]), float(bottle_xy[1]), 0.01], [1, 0, 0, 0])
                .add_rotation([0.0, 0.0, bottle_yaw])
            )

            forward_clearance, lateral_offset = self._measure_bottle_pose(bottle_pose, shelf_pose)
            if self._is_preferred_bottle_pose(forward_clearance, lateral_offset):
                return (
                    bottle_pose,
                    bottle_xy,
                    bottle_yaw,
                    forward_clearance,
                    lateral_offset,
                    attempt_idx,
                    attempt_idx,
                    "preferred",
                    False,
                )

            if self._is_safe_fallback_bottle_pose(forward_clearance, lateral_offset):
                safe_score = self._safe_fallback_score(forward_clearance, lateral_offset)
                if best_safe_score is None or safe_score < best_safe_score:
                    best_safe_score = safe_score
                    best_safe_candidate = (
                        bottle_pose,
                        np.array(bottle_xy, dtype=float),
                        float(bottle_yaw),
                        float(forward_clearance),
                        float(lateral_offset),
                        attempt_idx,
                    )

        if best_safe_candidate is not None:
            (
                bottle_pose,
                bottle_xy,
                bottle_yaw,
                bottle_forward,
                bottle_lateral,
                selected_sample_attempt,
            ) = best_safe_candidate
            return (
                bottle_pose,
                bottle_xy,
                bottle_yaw,
                bottle_forward,
                bottle_lateral,
                self.max_reset_sample_attempts,
                selected_sample_attempt,
                "safe_fallback",
                True,
            )

        raise RuntimeError(
            "Failed to sample a bottle pose that satisfies even the safety clearance constraints."
        )

    def _reset_actors(self):
        shelf_pose, shelf_xy, shelf_yaw = self._sample_shelf_pose()
        (
            bottle_pose,
            bottle_xy,
            bottle_yaw,
            bottle_forward,
            bottle_lateral,
            reset_sample_attempts,
            selected_sample_attempt,
            reset_acceptance_tier,
            used_reset_fallback,
        ) = self._sample_bottle_pose(shelf_pose)

        self.shelf.set_pose(shelf_pose)
        self.bottle.set_pose(bottle_pose)

        self.metadata["variant"] = self.variant_name
        self.metadata["reset_design"] = "sample_bottle_near_d0_then_filter_by_shelf_local_clearance"
        self.metadata["shelf_xy_offset"] = [float(shelf_xy[0]), float(shelf_xy[1])]
        self.metadata["shelf_yaw_offset"] = float(shelf_yaw)
        self.metadata["bottle_xy_offset"] = [float(bottle_xy[0]), float(bottle_xy[1])]
        self.metadata["reset_sample_attempts"] = int(reset_sample_attempts)
        self.metadata["selected_reset_sample_attempt"] = int(selected_sample_attempt)
        self.metadata["reset_acceptance_tier"] = reset_acceptance_tier
        self.metadata["used_reset_fallback"] = bool(used_reset_fallback)
        self.metadata["bottle_forward_offset_in_shelf_frame"] = float(bottle_forward)
        self.metadata["bottle_lateral_offset_in_shelf_frame"] = float(bottle_lateral)
        self.metadata["bottle_yaw_offset"] = float(bottle_yaw)
        self.metadata["bottle_pose"] = bottle_pose.tolist()

    def pre_move(self):
        self.delay(10)

        bottle_pose = self.bottle.get_pose()
        grasp_height = float(self.rng.uniform(self.hidden_grasp_height_min, self.hidden_grasp_height_max))
        target_pose = bottle_pose.add_bias([0, 0, grasp_height])
        target_pose = construct_grasp_pose(
            target_pose.p,
            [0, 0, 1],
            [1, 0, 0]
        )
        self.grasp_noise = Pose()
        if self.hidden_grasp_pitch_noise > 0:
            self.grasp_noise = self.create_noise(euler=[0, self.hidden_grasp_pitch_noise, 0])
            target_pose = target_pose.add_offset(self.grasp_noise)

        grasp_idx = self.bottle.register_point(
            pose=target_pose,
            type='contact'
        )
        self.move(self.atom.grasp_actor(
            self.bottle, contact_point_id=grasp_idx, pre_dis=0.0, is_close=False
        ))

        base_pose = self.shelf.get_pose()
        self.place_target = base_pose.add_bias([-0.2, 0, 0.21])
        self.move(self.atom.close_gripper())

        self.metadata["expert_design"] = (
            "keep_d0_pitch_noise_only_for_hidden_grasp; difficulty_from_shelf_local_reset_only"
        )
        self.metadata["hidden_grasp_height_range"] = [
            float(self.hidden_grasp_height_min),
            float(self.hidden_grasp_height_max),
        ]
        self.metadata["hidden_grasp_pitch_noise"] = float(self.hidden_grasp_pitch_noise)

    def _play_once(self):
        self.metadata["visible_expert"] = "reuse_d0_lift_reorient_place"
        super()._play_once()
