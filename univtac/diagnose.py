from __future__ import annotations

import argparse
import importlib
import os
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive terminal diagnose tool for UniVTAC tasks."
    )
    parser.add_argument("task_name", type=str)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--translation-step", type=float, default=0.0025, help="Translation delta in meters.")
    parser.add_argument("--rotation-step-deg", type=float, default=3.0, help="Rotation delta in degrees.")
    parser.add_argument("--gripper-step", type=float, default=0.002, help="Gripper delta in joint position.")
    parser.add_argument("--print-hz", type=float, default=8.0, help="Terminal refresh rate.")
    parser.add_argument("--step-limit", type=int, default=5000, help="Manual control horizon before reset is needed.")
    parser.add_argument(
        "--translation-frame",
        type=str,
        choices=("local", "world"),
        default="local",
        help="Frame used for translation keys.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


ARGS_CLI = build_parser().parse_args()
ARGS_CLI.enable_cameras = True
ARGS_CLI.num_envs = 1

APP_LAUNCHER = AppLauncher(ARGS_CLI)
SIMULATION_APP = APP_LAUNCHER.app


import numpy as np
import torch
import transforms3d as t3d

from envs.utils.transforms import Pose
from univtac.data.collect import DEFAULT_OBS_DATA_TYPE
from univtac.experiments import load_experiment_config
from univtac.paths import require_experiment_name
from univtac.registry import require_known_task


KEY_HELP = [
    "Move: w/s x, a/d y, e/c z",
    "Rotate (local): i/k pitch, j/l yaw, u/o roll",
    "Gripper: [ close, ] open",
    "Reset: r same seed, n next seed",
    "Frame: f toggle translation frame",
    "Other: p refresh, h help, q quit",
]


def _quat_to_deg_xyz(quat_wxyz: np.ndarray) -> np.ndarray:
    euler = np.array(t3d.euler.quat2euler(quat_wxyz), dtype=np.float64)
    return np.rad2deg(euler)


def _format_vec(vec: np.ndarray, scale: float = 1.0) -> str:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1) * scale
    return "[" + ", ".join(f"{item: .3f}" for item in arr) + "]"


def _format_pose(pose: Pose) -> str:
    pos_mm = np.asarray(pose.p, dtype=np.float64) * 1000.0
    rpy_deg = _quat_to_deg_xyz(np.asarray(pose.q, dtype=np.float64))
    return f"pos_mm={_format_vec(pos_mm)} rpy_deg={_format_vec(rpy_deg)}"


@contextmanager
def raw_terminal_mode():
    if not sys.stdin.isatty():
        raise RuntimeError("stdin is not a TTY. Run diagnose from an interactive terminal.")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        sys.stdout.write("\x1b[?25l")
        sys.stdout.flush()
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        sys.stdout.write("\x1b[?25h\x1b[0m\n")
        sys.stdout.flush()


class TerminalKeyboard:
    def __init__(self):
        self._fd = sys.stdin.fileno()

    def read_keys(self) -> list[str]:
        keys: list[str] = []
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                break
            chunk = os.read(self._fd, 64).decode("utf-8", errors="ignore")
            if not chunk:
                break
            for char in chunk:
                if char == "\x03":
                    raise KeyboardInterrupt
                keys.append(char)
        return keys


class DiagnoseSession:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.task_name = require_known_task(args.task_name)
        self.experiment_name = require_experiment_name(args.experiment_name)
        self.experiment_cfg = load_experiment_config(self.experiment_name)
        self.translation_frame = args.translation_frame
        self.current_seed = int(args.seed)
        self.last_message = "Ready."
        self.last_render_t = 0.0
        self.print_period_s = 1.0 / max(float(args.print_hz), 0.5)
        self._step_render_enabled = False

        os.environ["UNIVTAC_EXPERIMENT_NAME"] = self.experiment_name

        task_module = importlib.import_module(f"envs.{self.task_name}")
        env_cfg = task_module.TaskCfg()
        collection_cfg = self.experiment_cfg.get("collection", {})

        runtime_root = REPO_ROOT / ".cache" / "diagnose" / self.experiment_name / self.task_name
        env_cfg.tactile_sensor_type = self.experiment_cfg.get("tactile_sensor_type", "gsmini")
        env_cfg.save_dir = runtime_root / "save"
        env_cfg.video_save_dir = runtime_root / "videos"
        env_cfg.runtime_dir = runtime_root / "runtime"
        env_cfg.auto_write_metadata = False
        env_cfg.decimation = collection_cfg.get("decimation", env_cfg.decimation)
        env_cfg.save_frequency = 1
        env_cfg.video_frequency = 0
        env_cfg.render_frequency = 1
        env_cfg.video_size = tuple(collection_cfg.get("video_size", env_cfg.video_size))
        env_cfg.obs_data_type = dict(DEFAULT_OBS_DATA_TYPE)
        env_cfg.random_texture = collection_cfg.get("random_texture", False)
        env_cfg.scene.num_envs = 1
        env_cfg.step_lim = max(int(args.step_limit), int(env_cfg.step_lim))

        self.task = task_module.Task(env_cfg, mode="collect")
        self._orig_task_step = self.task._step
        self.task._step = self._hooked_task_step
        self.keyboard = TerminalKeyboard()
        self.reset(self.current_seed, reason="Initial reset completed.")

    def close(self) -> None:
        self.task._step = self._orig_task_step
        self.task.close()

    def _hooked_task_step(self, *args, **kwargs):
        result = self._orig_task_step(*args, **kwargs)
        if self._step_render_enabled:
            self._render_status(force=True)
        return result

    def reset(self, seed: int, *, reason: str) -> None:
        self.current_seed = int(seed)
        self.task.reset(seed=self.current_seed)
        self.task.plan_success = True
        self.task.eval_success = False
        self.task.force_stop = False
        self.task.force_stop_reason = None
        self.last_message = f"{reason} seed={self.current_seed}"
        self._render_status(force=True)

    def _translation_delta(self, key: str) -> np.ndarray:
        step = float(self.args.translation_step)
        mapping = {
            "w": np.array([step, 0.0, 0.0], dtype=np.float64),
            "s": np.array([-step, 0.0, 0.0], dtype=np.float64),
            "a": np.array([0.0, step, 0.0], dtype=np.float64),
            "d": np.array([0.0, -step, 0.0], dtype=np.float64),
            "e": np.array([0.0, 0.0, step], dtype=np.float64),
            "c": np.array([0.0, 0.0, -step], dtype=np.float64),
        }
        return mapping[key]

    def _rotation_delta(self, key: str) -> np.ndarray:
        step = np.deg2rad(float(self.args.rotation_step_deg))
        mapping = {
            "i": np.array([0.0, step, 0.0], dtype=np.float64),
            "k": np.array([0.0, -step, 0.0], dtype=np.float64),
            "j": np.array([0.0, 0.0, step], dtype=np.float64),
            "l": np.array([0.0, 0.0, -step], dtype=np.float64),
            "u": np.array([step, 0.0, 0.0], dtype=np.float64),
            "o": np.array([-step, 0.0, 0.0], dtype=np.float64),
        }
        return mapping[key]

    def _build_action(self, *, translation_key: str | None = None, rotation_key: str | None = None) -> torch.Tensor:
        ee_pose = self.task._robot_manager.get_ee_pose().clone()
        if translation_key is not None:
            ee_pose = ee_pose.add_bias(
                self._translation_delta(translation_key),
                coord=self.translation_frame,
            )
        if rotation_key is not None:
            ee_pose = ee_pose.add_rotation(self._rotation_delta(rotation_key), coord="local")

        gripper_qpos = self.task._robot_manager.get_gripper_qpos()
        return torch.tensor(
            ee_pose.tolist() + [float(gripper_qpos)],
            dtype=torch.float32,
            device=self.task.device,
        )

    def _step_gripper(self, delta: float) -> None:
        ee_pose = self.task._robot_manager.get_ee_pose()
        current = float(self.task._robot_manager.get_gripper_qpos())
        target = np.clip(current + delta, 0.0, float(self.task._robot_manager.gripper_max_qpos))
        action = torch.tensor(
            ee_pose.tolist() + [float(target)],
            dtype=torch.float32,
            device=self.task.device,
        )
        success, done = self._execute_action(action)
        self.last_message = f"Gripper -> {target:.4f} ({'ok' if success else 'fail'}, success_check={done})"

    def _idle_step(self) -> None:
        self.task._step(is_save=False)

    def _execute_action(self, action: torch.Tensor) -> tuple[bool, bool]:
        self.task.eval_success = False
        success, done = self.task.take_action(action, action_type="ee_servo")
        self.task.eval_success = False
        return success, done

    def _relative_pose_lines(self) -> list[str]:
        lines: list[str] = []
        actor_names = list(self.task._actor_manager.actors.keys())
        if "prism" in actor_names:
            prism = self.task._actor_manager.actors["prism"]
            inhand_pose = self.task._robot_manager.get_inhand_pose(prism)
            lines.append(f"prism@gripper {_format_pose(inhand_pose)}")
            if hasattr(self.task, "target_pose"):
                rel_target = prism.get_pose().rebase(getattr(self.task, "target_pose"))
                lines.append(f"prism@target  {_format_pose(rel_target)}")
            if hasattr(self.task, "hole_pose"):
                rel_hole = prism.get_pose().rebase(getattr(self.task, "hole_pose"))
                lines.append(f"prism@hole    {_format_pose(rel_hole)}")
        return lines

    def _status_text(self) -> str:
        ee_pose = self.task._robot_manager.get_ee_pose()
        grip_pose = self.task._robot_manager.get_gripper_center_pose()
        joint_pos = self.task._robot_manager.robot.data.joint_pos[0].detach().cpu().numpy()
        joint_vel = self.task._robot_manager.robot.data.joint_vel[0].detach().cpu().numpy()
        lines = [
            f"UniVTAC diagnose | experiment={self.experiment_name} task={self.task_name} seed={self.current_seed}",
            f"message: {self.last_message}",
            f"step_count={self.task.step_count} take_action_cnt={self.task.take_action_cnt} plan_success={self.task.plan_success} eval_success={self.task.eval_success}",
            f"translation_frame={self.translation_frame} translation_step={self.args.translation_step:.4f}m rotation_step={self.args.rotation_step_deg:.1f}deg gripper_step={self.args.gripper_step:.4f}",
            "",
            f"joint_pos  {_format_vec(joint_pos)}",
            f"joint_vel  {_format_vec(joint_vel)}",
            f"ee        {_format_pose(ee_pose)}",
            f"gripper   {_format_pose(grip_pose)}",
            f"gripper_qpos={self.task._robot_manager.get_gripper_qpos():.4f} / {self.task._robot_manager.gripper_max_qpos:.4f}",
        ]

        tactile_depth = self.task._tactile_manager.get_min_depth().detach().cpu().numpy()
        lines.append(f"tactile_min_depth_mm={_format_vec(tactile_depth)}")

        for name, actor in self.task._actor_manager.actors.items():
            lines.append(f"{name:<9} {_format_pose(actor.get_pose())}")

        lines.append("")
        lines.extend(self._relative_pose_lines())
        lines.append("")
        lines.append("Controls:")
        lines.extend(KEY_HELP)
        return "\n".join(lines)

    def _render_status(self, *, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and (now - self.last_render_t) < self.print_period_s:
            return
        self.last_render_t = now
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(self._status_text())
        sys.stdout.write("\n")
        sys.stdout.flush()

    def handle_key(self, key: str) -> bool:
        if key in {"\r", "\n", "\t"}:
            return True
        if key == "q":
            self.last_message = "Quit requested."
            return False
        if key == "h":
            self.last_message = "Help refreshed."
            self._render_status(force=True)
            return True
        if key == "p":
            self.last_message = "Manual refresh."
            self._render_status(force=True)
            return True
        if key == "f":
            self.translation_frame = "world" if self.translation_frame == "local" else "local"
            self.last_message = f"Translation frame -> {self.translation_frame}"
            return True
        if key == "r":
            self.reset(self.current_seed, reason="Reset same seed.")
            return True
        if key == "n":
            self.reset(self.current_seed + 1, reason="Reset next seed.")
            return True
        if key == "[":
            self._step_gripper(-float(self.args.gripper_step))
            return True
        if key == "]":
            self._step_gripper(float(self.args.gripper_step))
            return True
        if key in {"w", "s", "a", "d", "e", "c"}:
            success, done = self._execute_action(self._build_action(translation_key=key))
            self.last_message = f"Move {key} ({'ok' if success else 'fail'}, success_check={done})"
            return True
        if key in {"i", "k", "j", "l", "u", "o"}:
            success, done = self._execute_action(self._build_action(rotation_key=key))
            self.last_message = f"Rotate {key} ({'ok' if success else 'fail'}, success_check={done})"
            return True
        return True

    def run(self) -> None:
        self._step_render_enabled = True
        self._render_status(force=True)
        try:
            while SIMULATION_APP.is_running():
                keys = self.keyboard.read_keys()
                keep_running = True
                for key in keys:
                    keep_running = self.handle_key(key)
                    self._render_status(force=True)
                    if not keep_running:
                        return
                if not keys:
                    self._idle_step()
                    self._render_status()
        finally:
            self._step_render_enabled = False


def main() -> None:
    session: DiagnoseSession | None = None
    try:
        session = DiagnoseSession(ARGS_CLI)
        with raw_terminal_mode():
            session.run()
    except KeyboardInterrupt:
        pass
    finally:
        if session is not None:
            session.close()
        SIMULATION_APP.close()


if __name__ == "__main__":
    main()
