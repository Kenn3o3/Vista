from .put_bottle_in_shelf_D1 import Task as BasePutBottleInShelfD1Task, TaskCfg as BasePutBottleInShelfD1TaskCfg
from ._base_task import configclass
import numpy as np


@configclass
class TaskCfg(BasePutBottleInShelfD1TaskCfg):
    pass


class Task(BasePutBottleInShelfD1Task):
    variant_name = "D2"

    shelf_xy_range = np.array([0.085, 0.032], dtype=float)
    shelf_yaw_range = np.deg2rad(5.8)

    bottle_xy_range = np.array([0.018, 0.052], dtype=float)
    bottle_yaw_range = np.deg2rad(9.0)

    preferred_min_forward_clearance = 0.35
    preferred_max_forward_clearance = 0.48
    safe_min_forward_clearance = 0.32
    safe_max_forward_clearance = 0.50
    max_lateral_offset_in_shelf_frame = 0.065
