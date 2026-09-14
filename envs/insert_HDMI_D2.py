from .insert_HDMI_D1 import Task as BaseInsertHDMID1Task, TaskCfg as BaseInsertHDMID1TaskCfg
from ._base_task import configclass
import numpy as np


@configclass
class TaskCfg(BaseInsertHDMID1TaskCfg):
    pass


class Task(BaseInsertHDMID1Task):
    variant_name = "D2"

    # D2 在 D1 的基础上继续放大：
    # 1) hole yaw 更大；
    # 2) grasp pose 的 local x/y 与 pitch/yaw 扰动更大；
    # 3) hidden staging pose 从 D0 的 ±5 mm 扩大到约 ±12 mm，并加入极小 z 偏移。
    slot_yaw_range = np.deg2rad(12.0)

    grasp_height_range = (0.010, 0.0145)
    grasp_longitudinal_range = 0.0045
    grasp_lateral_range = 0.001
    grasp_pitch_range = np.deg2rad(20.0)
    grasp_yaw_range = np.deg2rad(8.0)

    staging_xy_range = np.array([0.012, 0.012], dtype=float)
    staging_z_range = (0.0, 0.002)
