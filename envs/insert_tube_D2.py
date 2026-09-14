from .insert_tube_D1 import Task as BaseInsertTubeD1Task, TaskCfg as BaseInsertTubeD1TaskCfg
from ._base_task import configclass
import numpy as np


@configclass
class TaskCfg(BaseInsertTubeD1TaskCfg):
    pass


class Task(BaseInsertTubeD1Task):
    variant_name = "D2"

    slot_xy_range = np.array([0.02, 0.04], dtype=float)
    slot_yaw_range = np.deg2rad(15.0)

    grasp_height_range = (0.094, 0.101)
    grasp_longitudinal_range = 0.0
    grasp_lateral_range = 0.0

    stage_backoff_range = (0.03, 0.05)
    stage_xy_range = np.array([0.008, 0.008], dtype=float)

    prealign_depth_margin = 0.008
    initial_forward_max = 0.05
    initial_forward_delta = 0.01

    final_insert_dis = 0.018
    final_insert_delta = 0.01
    final_push_displacement = -0.04
