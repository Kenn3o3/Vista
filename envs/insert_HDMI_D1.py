from ._base_task import *
from .insert_HDMI import Task as BaseInsertHDMITask, TaskCfg as BaseInsertHDMITaskCfg
import numpy as np
import transforms3d as t3d


@configclass
class TaskCfg(BaseInsertHDMITaskCfg):
    pass


class Task(BaseInsertHDMITask):
    variant_name = "D1"

    # D1 只做轻度扩展：
    # 1) slot / hole 增加少量 yaw 扰动；
    # 2) grasp pose 增加少量 local x/y 平移与 pitch/yaw 角度扰动；
    # 3) hidden staging pose 维持 D0 范围，保证 expert 先保持可用。
    slot_xy_range = np.array([0.005, 0.005], dtype=float)
    slot_yaw_range = np.deg2rad(6.0)

    grasp_height_range = (0.011, 0.0135)
    grasp_longitudinal_range = 0.002
    grasp_lateral_range = 0.0005
    grasp_pitch_range = np.deg2rad(15.0)
    grasp_yaw_range = np.deg2rad(5.0)

    staging_xy_range = np.array([0.005, 0.005], dtype=float)
    staging_z_range = (0.0, 0.0)

    def _sample_scalar(self, low: float, high: float) -> float:
        return float(self.rng.uniform(low, high))

    def _sample_symmetric(self, magnitude: float) -> float:
        return float(self.rng.uniform(-magnitude, magnitude))

    def _reset_actors(self):
        slot_xy = self.rng.uniform(-self.slot_xy_range, self.slot_xy_range)
        slot_yaw = self._sample_symmetric(self.slot_yaw_range)
        slot_pose = Pose(
            [0.55 + float(slot_xy[0]), float(slot_xy[1]), self.slot.get_pose()[2]],
            t3d.euler.euler2quat(0.0, 0.0, slot_yaw),
        )
        self.slot.set_pose(slot_pose)
        self.metadata["variant"] = self.variant_name
        self.metadata["slot_xy_offset"] = [float(slot_xy[0]), float(slot_xy[1])]
        self.metadata["slot_yaw_offset"] = float(slot_yaw)

    def pre_move(self):
        self.delay(10)

        self.move(self.atom.open_gripper(0.5))

        # 约定 HDMI local frame:
        # - local x: 近似作为插入前后 / longitudinal 方向
        # - local y: 近似作为宽度 / lateral 方向
        # - local z: 竖直方向
        grasp_longitudinal = self._sample_symmetric(self.grasp_longitudinal_range)
        grasp_lateral = self._sample_symmetric(self.grasp_lateral_range)
        grasp_height = self._sample_scalar(*self.grasp_height_range)
        grasp_pitch = self._sample_symmetric(self.grasp_pitch_range)
        grasp_yaw = self._sample_symmetric(self.grasp_yaw_range)

        self.metadata.update(
            {
                "grasp_offset_frame": "prism_local(x=longitudinal, y=lateral, z=vertical)",
                "grasp_longitudinal_offset": float(grasp_longitudinal),
                "grasp_lateral_offset": float(grasp_lateral),
                "grasp_height_offset": float(grasp_height),
                "grasp_pitch_offset": float(grasp_pitch),
                "grasp_yaw_offset": float(grasp_yaw),
            }
        )

        target_pose = (
            self.prism.get_pose()
            .add_bias([grasp_longitudinal, grasp_lateral, grasp_height])
            .add_rotation([0.0, grasp_pitch, grasp_yaw])
        )
        target_mat = target_pose.to_transformation_matrix()
        cpose = construct_grasp_pose(
            target_pose.p,
            target_mat[:3, 2],
            target_mat[:3, 0],
        )
        cid = self.prism.register_point(cpose, type="contact")
        self.move(
            self.atom.grasp_actor(
                self.prism,
                contact_point_id=cid,
                is_close=False,
            )
        )
        self.move(self.atom.close_gripper())
        self.move(self.atom.move_by_displacement(z=0.02))

        self.target_pose = self.slot.get_pose().add_bias([0.0, 0.0, 0.005])
        self.hole_pose = self.slot.get_pose().add_bias([0.0, 0.0, 0.0128])

        staging_xy = self.rng.uniform(-self.staging_xy_range, self.staging_xy_range)
        staging_z = self._sample_scalar(*self.staging_z_range)
        staging_offset = Pose([float(staging_xy[0]), float(staging_xy[1]), staging_z], [1, 0, 0, 0])
        self.noise_pose = self.hole_pose.add_offset(staging_offset)
        self.metadata["staging_offset"] = [float(staging_xy[0]), float(staging_xy[1]), float(staging_z)]
        self.move(
            self.atom.place_actor(
                self.prism,
                target_pose=self.noise_pose,
                pre_dis=0.02,
                dis=0.01,
                is_open=False,
            )
        )
