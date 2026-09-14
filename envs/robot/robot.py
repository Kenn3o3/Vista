import yaml
import numpy as np
import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.utils import configclass

from ..utils.transforms import *
from ..utils.atom import GRASP_DIRECTION_DIC
from .robot_cfg import RobotCfg
from .curobo_planner import CuroboPlanner, CuroboPlannerCfg
from .._global import *

if TYPE_CHECKING:
    from curobo.wrap.reacher.motion_gen import MotionGenResult
    from .._base_task import BaseTask

class RobotManager:
    def __init__(self, robot_cfg:RobotCfg, task:'BaseTask', planner_time_dilation_factor:float=1.0):
        self.cfg = robot_cfg
        self.task = task
        self.device = task.device
        self.sensor_type = task.cfg.tactile_sensor_type
        if self.sensor_type in ['gsmini', 'gf225', 'xensews']: # franka panda
            self.robot_type = 'franka_panda'

        self.robot = Articulation(self.cfg.robot)
        self.task.scene.articulations['robot'] = self.robot
        self.planner_time_dilation_factor = planner_time_dilation_factor

        self.gripper_max_qpos = 0.039
        self.last_arm_velocity = None
        self.last_gripper_velocity = None

        if self.robot_type == 'franka_panda':
            self.hand_name = 'panda_hand'
            self._arm_joint_names = [
                'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
                'panda_joint5', 'panda_joint6', 'panda_joint7'
            ]
            self._gripper_joint_names = [
                'panda_finger_joint1', 'panda_finger_joint2'
            ]
            self.gripper_max_qpos = self.cfg.gripper_max_qpos
            self.yaml_path = str(EMBODIMENTS_ROOT / 'franka' / 'curobo.yml')
            offset = self.cfg.gripper_offset
        else:
            raise NotImplementedError(f"Robot type {self.robot_type} not implemented.")

        # offset from end-effector to gripper center frame
        self._offset = Pose(p=[0, 0, -offset], q=[1, 0, 0, 0])
        self._offset_pos = torch.tensor([0.0, 0.0, offset], device=self.device).repeat(self.task.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.task.num_envs, 1)

    def setup(self):
        """设置机器人属性"""
        body_ids, body_names = self.robot.find_bodies(self.hand_name)
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        self._jacobi_body_idx = self._body_idx - 1

        joint_names = self.robot.joint_names
        self.joint_name_to_id = {name: i for i, name in enumerate(joint_names)}

        self._arm_ids = torch.tensor([
            self.joint_name_to_id[n] for n in self._arm_joint_names
        ], device=self.device)
        self._gripper_ids = torch.tensor([
            self.joint_name_to_id[n] for n in self._gripper_joint_names
        ], device=self.device)
        self.origin_pose = self.get_gripper_center_pose()
        self._all_ids = torch.cat([self._arm_ids, self._gripper_ids], dim=0)

        self.root_pose = Pose.from_list(self.robot.data.root_link_pos_w[0])
        planner_cfg = CuroboPlannerCfg(
            dt=self.task.cfg.sim.dt,
            all_joints_name=self.robot.joint_names,
            active_joints_name=self._arm_joint_names,
            robot_prime_path=self.cfg.robot.prim_path,
            yaml_path=self.yaml_path
        )
        self.planner = CuroboPlanner(
            task=self.task,
            cfg=planner_cfg,
            robot_origin_pose=self.root_pose,
        )
        self.diff_ik = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
                ik_params={"lambda_val": 0.05},
            ),
            num_envs=self.task.num_envs,
            device=self.device,
        )
        self.diff_ik = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
                ik_params={"lambda_val": 0.05},
            ),
            num_envs=self.task.num_envs,
            device=self.device,
        )

    def ee_to_gripper_center(self, ee_pose:Pose) -> Pose:
        """将夹爪中心位姿转换为末端执行器目标位姿"""
        return ee_pose.add_offset(self._offset.inv())

    def gripper_center_to_ee(self, gripper_center_pose:Pose) -> Pose:
        """将夹爪中心位姿转换为末端执行器目标位姿"""
        return gripper_center_pose.add_offset(self._offset)

    def get_gripper_center_pose(self, env_ids:slice=None) -> Pose:
        """获取当前夹爪中心位姿"""
        return self.ee_to_gripper_center(self.get_ee_pose())

    def get_inhand_pose(self, actor:'Actor') -> Pose:
        return actor.get_pose().rebase(self.get_gripper_center_pose())

    def get_ee_pose(self, env_ids:slice=None) -> Pose:
        """获取当前末端执行器目标位姿（target_pose）"""
        if env_ids is None:
            env_ids = [0]
        ee_pos_w = self.robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self.robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self.robot.data.root_link_pos_w
        root_quat_w = self.robot.data.root_link_quat_w
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        return Pose(ee_pose_b[0].cpu().numpy(), ee_quat_b[0].cpu().numpy())

    def get_ee_pose_tensor(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self.robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self.robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self.robot.data.root_link_pos_w
        root_quat_w = self.robot.data.root_link_quat_w
        return math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

    def get_arm_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos[:, self._arm_ids]

    def get_arm_joint_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        limits = self.robot.data.soft_joint_pos_limits[:, self._arm_ids]
        return limits[..., 0], limits[..., 1]

    def get_ee_jacobian(self) -> torch.Tensor:
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._arm_ids]
        root_quat_w = self.robot.data.root_link_quat_w
        base_rot = math_utils.matrix_from_quat(math_utils.quat_inv(root_quat_w))
        jacobian_pos = torch.bmm(base_rot, jacobian[:, :3, :])
        jacobian_rot = torch.bmm(base_rot, jacobian[:, 3:, :])
        return torch.cat([jacobian_pos, jacobian_rot], dim=1)

    def reset_diff_ik(self):
        if hasattr(self, "diff_ik"):
            self.diff_ik.reset()

    def compute_ee_pose_error(self, target_pose: Pose | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(target_pose, Pose):
            command = target_pose.totensor(device=self.device).reshape(1, 7)
        else:
            command = torch.as_tensor(target_pose, dtype=torch.float32, device=self.device).reshape(-1, 7)

        ee_pos_b, ee_quat_b = self.get_ee_pose_tensor()
        target_pos = command[:, :3]
        target_quat = command[:, 3:7]
        pos_err, rot_err = math_utils.compute_pose_error(
            ee_pos_b,
            ee_quat_b,
            target_pos,
            target_quat,
            rot_error_type="axis_angle",
        )
        if pos_err.shape[0] == 1:
            pos_err = pos_err[0]
            rot_err = rot_err[0]
        return pos_err, rot_err

    # def compute_arm_qpos_from_ee_target(
    #     self,
    #     target_pose: Pose | torch.Tensor,
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     if isinstance(target_pose, Pose):
    #         command = target_pose.totensor(device=self.device).reshape(1, 7)
    #     else:
    #         command = torch.as_tensor(target_pose, dtype=torch.float32, device=self.device).reshape(-1, 7)

    #     ee_pos_b, ee_quat_b = self.get_ee_pose_tensor()
    #     joint_pos = self.get_arm_joint_pos()
    #     jacobian = self.get_ee_jacobian()

    #     self.diff_ik.set_command(command, ee_pos_b, ee_quat_b)
    #     joint_pos_des = self.diff_ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

    #     # joint_lo, joint_hi = self.get_arm_joint_limits()
    #     # joint_pos_des = torch.clamp(joint_pos_des, joint_lo, joint_hi)

    #     # joint_step_limit = float(getattr(self.task.cfg, "ee_servo_joint_step_limit", 0.0))
    #     # if joint_step_limit > 0:
    #     #     joint_delta = torch.clamp(joint_pos_des - joint_pos, -joint_step_limit, joint_step_limit)
    #     #     joint_pos_des = joint_pos + joint_delta

    #     position_error, axis_angle_error = math_utils.compute_pose_error(
    #         ee_pos_b,
    #         ee_quat_b,
    #         self.diff_ik.ee_pos_des,
    #         self.diff_ik.ee_quat_des,
    #         rot_error_type="axis_angle",
    #     )

    #     if joint_pos_des.shape[0] == 1:
    #         joint_pos_des = joint_pos_des[0]
    #         position_error = position_error[0]
    #         axis_angle_error = axis_angle_error[0]
    #     return joint_pos_des, position_error, axis_angle_error

    def get_ee_pose_tensor(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self.robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self.robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self.robot.data.root_link_pos_w
        root_quat_w = self.robot.data.root_link_quat_w
        return math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

    def get_arm_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos[:, self._arm_ids]

    def get_arm_joint_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        limits = self.robot.data.soft_joint_pos_limits[:, self._arm_ids, :]
        return limits[..., 0], limits[..., 1]

    def get_ee_jacobian(self) -> torch.Tensor:
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._arm_ids]
        root_quat_w = self.robot.data.root_link_quat_w
        base_rot = math_utils.matrix_from_quat(math_utils.quat_inv(root_quat_w))
        jacobian_pos = torch.bmm(base_rot, jacobian[:, :3, :])
        jacobian_rot = torch.bmm(base_rot, jacobian[:, 3:, :])
        return torch.cat([jacobian_pos, jacobian_rot], dim=1)

    def reset_diff_ik(self):
        self.diff_ik.reset()

    def compute_ee_pose_error(self, target_pose: Pose | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(target_pose, Pose):
            command = target_pose.totensor(device=self.device).reshape(1, 7)
        else:
            command = torch.as_tensor(target_pose, dtype=torch.float32, device=self.device).reshape(-1, 7)

        ee_pos_b, ee_quat_b = self.get_ee_pose_tensor()
        target_pos = command[:, :3]
        target_quat = command[:, 3:7]
        return math_utils.compute_pose_error(
            ee_pos_b,
            ee_quat_b,
            target_pos,
            target_quat,
            rot_error_type="axis_angle",
        )

    def compute_arm_qpos_from_ee_target(
        self,
        target_pose: Pose | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(target_pose, Pose):
            command = target_pose.totensor(device=self.device).reshape(1, 7)
        else:
            command = torch.as_tensor(target_pose, dtype=torch.float32, device=self.device).reshape(-1, 7)

        ee_pos_b, ee_quat_b = self.get_ee_pose_tensor()
        joint_pos = self.get_arm_joint_pos()
        jacobian = self.get_ee_jacobian()

        self.diff_ik.set_command(command, ee_pos_b, ee_quat_b)
        joint_pos_des = self.diff_ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # joint_lo, joint_hi = self.get_arm_joint_limits()
        # joint_pos_des = torch.clamp(joint_pos_des, joint_lo, joint_hi)

        # step_limit = float(getattr(self.task.cfg, "ee_servo_joint_step_limit", 0.0))
        # if step_limit > 0:
        #     joint_delta = torch.clamp(joint_pos_des - joint_pos, min=-step_limit, max=step_limit)
        #     joint_pos_des = joint_pos + joint_delta

        position_error, axis_angle_error = math_utils.compute_pose_error(
            ee_pos_b,
            ee_quat_b,
            self.diff_ik.ee_pos_des,
            self.diff_ik.ee_quat_des,
            rot_error_type="axis_angle",
        )
        return joint_pos_des, position_error, axis_angle_error

    def get_qpos(self):
        return self.robot.data.joint_pos.clone().cpu()

    def get_gripper_qpos(self):
        return self.get_qpos()[0, self._gripper_ids[0]].clone().cpu().item()
    def get_gripper_percentage(self):
        return self.get_gripper_qpos().item() / self.gripper_max_qpos

    def _record_physx_joint_target(self, *, source: str) -> None:
        if not hasattr(self.task, "_record_physx_joint_write"):
            return

        joint_pos_target = self.robot._data.joint_pos_target.detach().cpu()
        if joint_pos_target.ndim > 1:
            joint_pos_target = joint_pos_target[0]

        self.task._record_physx_joint_write(
            {
                "source": source,
                "sim_step": int(getattr(self.task, "step_count", 0)) + 1,
                "take_action_cnt": int(getattr(self.task, "take_action_cnt", 0)),
                "joint_names": list(self.robot.joint_names),
                "joint_pos_target": joint_pos_target.tolist(),
            }
        )

    def set_arm(self, pos:torch.Tensor, vel:torch.Tensor=None, env_ids:slice=None, force:bool=True):
        '''设置目标位姿'''
        pos = torch.as_tensor(pos, dtype=torch.float32, device=self.device)
        if vel is not None:
            vel = torch.as_tensor(vel, dtype=torch.float32, device=self.device)
        self.robot.set_joint_position_target(pos, joint_ids=self._arm_ids, env_ids=env_ids)
        if vel is not None:
            self.robot.set_joint_velocity_target(vel, joint_ids=self._arm_ids, env_ids=env_ids)
        if force:
            self.robot.root_physx_view.set_dof_positions(
                self.robot._data.joint_pos_target,
                self.robot._ALL_INDICES
            )
            self._record_physx_joint_target(source="set_arm")

    def set_gripper(self, pos:torch.Tensor, vel:torch.Tensor=None, env_ids:slice=None, force:bool=True):
        '''设置目标位姿'''
        pos = torch.as_tensor(pos, dtype=torch.float32, device=self.device)
        if vel is not None:
            vel = torch.as_tensor(vel, dtype=torch.float32, device=self.device)
        self.robot.set_joint_position_target(pos, joint_ids=self._gripper_ids, env_ids=env_ids)
        if vel is not None:
            self.robot.set_joint_velocity_target(vel, joint_ids=self._gripper_ids, env_ids=env_ids)
        if force:
            self.robot.root_physx_view.set_dof_positions(
                self.robot._data.joint_pos_target,
                self.robot._ALL_INDICES
            )
            self._record_physx_joint_target(source="set_gripper")

    def plan_arm(self, target_pose:Pose, constraint_pose=None, pre_dis=None, time_dilation_factor=None):
        result:MotionGenResult = self.planner.plan_path(
            curr_joint_pos=self.robot.data.joint_pos[0, :self.robot.num_joints-2],
            curr_joint_vel=self.robot.data.joint_vel[0, :self.robot.num_joints-2],
            target_ee_pose=target_pose,
            real_robot_pose=self.root_pose,
            pre_dis=pre_dis,
            constraint_pose=constraint_pose,
            time_dilation_factor=time_dilation_factor
        )

        if result.success.item():
            return {
                'status': 'Success',
                'num_steps': result.interpolated_plan.position.shape[0],
                'position': result.interpolated_plan.position.detach(),
                'velocity': result.interpolated_plan.velocity.detach()
            }
        else:
            return {'status': 'Fail', 'num_steps': 0, 'position': None, 'velocity': None}

    def gripper_percent2qpos(self, percentage:float):
        gripper_range = [0, self.gripper_max_qpos]
        target_pos = gripper_range[0] + (gripper_range[1] - gripper_range[0]) * percentage
        return target_pos

    def plan_gripper(self, pos:float, type:Literal['percent', 'qpos'] = 'percent'):
        if type == 'percent':
            target_pos = self.gripper_percent2qpos(pos)
        else:
            target_pos = pos
        gripper_pos = self.robot.data.joint_pos[0, self._gripper_ids][0]
        num_steps = np.ceil(abs(target_pos - gripper_pos.cpu().item()) / 0.0005).astype(int)
        position = torch.linspace(gripper_pos, target_pos, num_steps, device=self.device)
        velocity = torch.clip((position - gripper_pos)/self.task.cfg.sim.dt, -0.0001, 0.0001)

        return {
            'status': 'Success',
            'num_steps': num_steps,
            'position': position.detach(),
            'velocity': velocity.detach()
        }

    def _reset_idx(self, env_ids: torch.Tensor | None=None):
        """重置环境"""
        if not hasattr(self, 'origin_pose'):
            self._setup_robot_properties()
        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(joint_pos)

        self.planner.reset()
        self.reset_diff_ik()
        self.robot.set_joint_position_target(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

    def get_observations(self, data_type:list[str]=['joint', 'ee']) -> dict:
        obs = {}
        if 'ee' in data_type:
            obs['ee'] = self.get_ee_pose().totensor(device=self.device)
        if 'joint' in data_type:
            obs['joint'] = self.robot.data.joint_pos.squeeze(0)
        return obs

    def get_grasp_perfect_direction(self):
        return 'top_down'
