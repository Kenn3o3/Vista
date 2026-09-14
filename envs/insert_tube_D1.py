from ._base_task import *
from .insert_tube import Task as BaseInsertTubeTask, TaskCfg as BaseInsertTubeTaskCfg
import numpy as np


@configclass
class TaskCfg(BaseInsertTubeTaskCfg):
    pass


class Task(BaseInsertTubeTask):
    variant_name = "D1"

    slot_xy_range = np.array([0.015, 0.03], dtype=float)
    slot_yaw_range = np.deg2rad(8.0)

    # 参考 D0，只保留稳定 top grasp 的高度偏移，不再注入 grasp 旋转扰动。
    grasp_height_range = (0.095, 0.100)
    grasp_longitudinal_range = 0.0
    grasp_lateral_range = 0.0

    # 可见起点直接设为“沿插入方向后退且横向错位”的状态，而不是回到 D0 近目标起点。
    stage_backoff_range = (0.02, 0.035)
    stage_xy_range = np.array([0.005, 0.005], dtype=float)
    stage_approach_pre_dis = 0.10
    stage_approach_dis = 0.05
    stage_refine_pre_dis = 0.02
    stage_refine_dis = 0.002

    pre_stage_lift = 0.15

    prealign_depth_margin = 0.006
    initial_forward_max = 0.04
    initial_forward_delta = 0.008
    approach_time_dilation_factor = 0.4
    realign_time_dilation_factor = 0.5

    final_insert_dis = 0.015
    final_insert_delta = 0.008
    final_push_displacement = -0.04

    def _sample_scalar(self, low: float, high: float) -> float:
        return float(self.rng.uniform(low, high))

    def _sample_symmetric(self, magnitude: float) -> float:
        return float(self.rng.uniform(-magnitude, magnitude))

    def _reset_actors(self):
        slot_xy = self.rng.uniform(-self.slot_xy_range, self.slot_xy_range)
        slot_yaw = self._sample_symmetric(self.slot_yaw_range)
        slot_pose = (
            Pose([0.6 + float(slot_xy[0]), float(slot_xy[1]), self.slot.get_pose()[2]], [1, 0, 0, 0])
            .add_rotation([0.0, 0.0, slot_yaw])
        )
        self.slot.set_pose(slot_pose)

        self.metadata["variant"] = self.variant_name
        self.metadata["slot_xy_offset"] = [float(slot_xy[0]), float(slot_xy[1])]
        self.metadata["slot_yaw_offset"] = float(slot_yaw)

    def _sample_stage_pose(self) -> Pose:
        stage_backoff = self._sample_scalar(*self.stage_backoff_range)
        stage_xy = self.rng.uniform(-self.stage_xy_range, self.stage_xy_range)
        stage_offset = Pose(
            [float(stage_xy[0]), float(stage_xy[1]), float(stage_backoff)],
            [1, 0, 0, 0],
        )
        stage_pose = self.hole_pose.add_offset(stage_offset)

        self.metadata["visible_start_design"] = "backoff_along_insertion_direction_with_xy_misalignment"
        self.metadata["stage_backoff"] = float(stage_backoff)
        self.metadata["stage_xy_offset"] = [float(stage_xy[0]), float(stage_xy[1])]
        self.metadata["stage_pose"] = stage_pose.tolist()
        return stage_pose

    def _grasp_tube_stably(self):
        grasp_longitudinal = self._sample_symmetric(self.grasp_longitudinal_range)
        grasp_lateral = self._sample_symmetric(self.grasp_lateral_range)
        grasp_height = self._sample_scalar(*self.grasp_height_range)

        target_pose = self.prism.get_pose().add_bias(
            [grasp_longitudinal, grasp_lateral, grasp_height]
        )
        contact_pose = construct_grasp_pose(
            target_pose.p,
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        )
        contact_point_id = self.prism.register_point(contact_pose, type="contact")
        self.move(
            self.atom.grasp_actor(
                self.prism,
                contact_point_id=contact_point_id,
                pre_dis=0.0,
                dis=0.0,
            )
        )
        self.origin_inhand_pose = self.prism.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose()
        )

        self.metadata.update(
            {
                "grasp_offset_frame": "prism_local(x=longitudinal, y=lateral, z=vertical)",
                "grasp_longitudinal_offset": float(grasp_longitudinal),
                "grasp_lateral_offset": float(grasp_lateral),
                "grasp_height_offset": float(grasp_height),
                "grasp_rotation_offset": [0.0, 0.0, 0.0],
            }
        )

    def _move_to_stage_pose(self, stage_pose: Pose):
        self.metadata["pre_stage_lift"] = float(self.pre_stage_lift)
        self.move(
            self.atom.move_by_displacement(z=self.pre_stage_lift),
            constraint_pose=[1, 1, 1, 1, 1, 0],
        )
        self.move(
            self.atom.place_actor(
                self.prism,
                target_pose=stage_pose,
                pre_dis=self.stage_approach_pre_dis,
                dis=self.stage_approach_dis,
                is_open=False,
            )
        )
        self.move(
            self.atom.place_actor(
                self.prism,
                target_pose=stage_pose,
                pre_dis=self.stage_refine_pre_dis,
                dis=self.stage_refine_dis,
                is_open=False,
            ),
            time_dilation_factor=self.realign_time_dilation_factor,
            delay=False,
        )

    def _advance_towards_hole(self, dis: float, delta_d: float, tag: str) -> bool:
        if dis <= 0.0:
            return True

        actor_last_pose = self.prism.get_pose()
        max_trials = int(np.ceil(dis / delta_d))
        for _ in range(max_trials):
            self.move(
                self.atom.move_by_displacement(z=delta_d, xyz_coord="local"),
                tag=tag,
                delay=False,
                time_dilation_factor=self.approach_time_dilation_factor,
                constraint_pose=[1, 1, 1, 1, 1, 0],
            )
            actor_pose = self.prism.get_pose()
            if np.linalg.norm(actor_pose.p - actor_last_pose.p) < delta_d:
                return False
            actor_last_pose = actor_pose
        return True

    def _realign_at_current_depth(self):
        rel_pose = self.prism.get_pose().rebase(self.hole_pose)
        raw_rel_depth = float(rel_pose.p[2])
        rel_depth = max(raw_rel_depth, 0.0)
        self.metadata["raw_realign_rel_depth"] = raw_rel_depth
        self.metadata["realign_rel_depth"] = rel_depth
        self.move(
            self.atom.place_actor(
                self.prism,
                target_pose=self.hole_pose,
                pre_dis=rel_depth,
                dis=rel_depth,
                is_open=False,
            ),
            time_dilation_factor=self.realign_time_dilation_factor,
            delay=False,
        )

    def pre_move(self):
        self.delay(10)
        self._grasp_tube_stably()

        base_pose = self.slot.get_pose()
        self.hole_pose = base_pose.add_bias([-0.008, 0.0, 0.077]).add_rotation([0.0, -np.pi / 6, 0.0])
        self.stage_pose = self._sample_stage_pose()
        self.metadata["hidden_carry_design"] = "stable_grasp_then_two_stage_move_to_backoff_stage"
        self._move_to_stage_pose(self.stage_pose)

        self.metadata["visible_start_rel_pose"] = self.prism.get_pose().rebase(self.hole_pose).tolist()
        self.metadata["visible_start_inhand_pose"] = self.prism.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose()
        ).tolist()

    def _play_once(self):
        rel_pose = self.prism.get_pose().rebase(self.hole_pose)
        initial_forward_dis = float(
            np.clip(
                max(0.0, float(rel_pose.p[2]) - self.prealign_depth_margin),
                0.0,
                self.initial_forward_max,
            )
        )
        self.metadata["visible_expert"] = "approach_from_backoff_then_realign_then_insert"
        self.metadata["prealign_depth_margin"] = self.prealign_depth_margin
        self.metadata["initial_forward_dis"] = initial_forward_dis
        self.metadata["initial_forward_rel_pose"] = rel_pose.tolist()

        self._advance_towards_hole(
            dis=initial_forward_dis,
            delta_d=self.initial_forward_delta,
            tag="approach",
        )
        if not self.check_mid_success():
            self._realign_at_current_depth()
            self._advance_towards_hole(
                dis=self.final_insert_dis,
                delta_d=self.final_insert_delta,
                tag="insert",
            )

        self.move(
            self.atom.move_by_displacement(
                z=self.final_push_displacement,
                xyz_coord=self.prism.get_pose(),
            ),
            time_dilation_factor=0.2,
        )
        self.delay(20, is_save=False)
