from shutil import ExecError
import sys

import time
import json
import yaml
import torch
import argparse
import os
import traceback
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
policy_root = REPO_ROOT / "policy"
if str(policy_root) not in sys.path:
    sys.path.append(str(policy_root))

from isaaclab.app import AppLauncher
from univtac.experiments import load_experiment_config

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Eval Policy"
)
parser.add_argument(
    "task_name",
    type=str,
    help="Task name",
)
parser.add_argument(
    "task_config",
    type=str,
    help="Task config",
)
parser.add_argument(
    "deploy_config",
    type=str,
    help="Deploy file name",
)
parser.add_argument(
    "--expert_check",
    action='store_true',
    help="Whether to do expert check before eval"
)
parser.add_argument(
    "--start_seed",
    type=int,
    default=-1
)
parser.add_argument(
    "--max_seed",
    type=int,
    default=-1
)
parser.add_argument(
    "--total_num",
    type=int,
    default=100
)
parser.add_argument(
    "--print_only",
    action='store_true',
)
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.livestream = int(os.environ.get("VISTA_LIVESTREAM", "0"))
args_cli.num_envs = 1

# launch omniverse app, must done before importing anything from omni.isaac
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import traceback
import importlib
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from envs._base_task import BaseTask, BaseTaskCfg
    from policy._base_policy import BasePolicy

log_path = Path('./log')
DEFAULT_OBS_DATA_TYPE = {
    "camera": ["rgb"],
    "tactile": ["rgb", "rgb_marker", "marker", "depth", "pose"],
    "embodiment": ["joint", "ee"],
    "actor": True,
}

def log(msg):
    global log_path, args_cli
    msg = f"[{time.strftime(r'%Y-%m-%d %H:%M:%S')}] {msg}"
    if not args_cli.print_only:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')
    print(msg)


def finalize_policy_rollout(policy, task, result: str):
    try:
        policy.on_rollout_end(task, result=result)
    except Exception as exc:
        log(f"Policy rollout finalization failed: {exc}\n{traceback.format_exc()}")

def eval_policy(
    task: 'BaseTask', policy: 'BasePolicy', expert_check,
    start_seed, max_seed, test_total_num, instructions, instruciton_type:Literal['seen', 'unseen']='seen'
):
    test_num, succ_num, seed = 0, 0, start_seed

    seed_path = task.save_root.parent / 'seeds.json'
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    if seed_path.exists():
        with open(seed_path, 'r') as f:
            seed_status = json.load(f)
    else:
        seed_status = {}

    while test_num < test_total_num and (max_seed == -1 or seed <= max_seed):
        if not seed_status.get(str(seed), True):
            seed += 1
            continue

        if expert_check and str(seed) not in seed_status:
            test_start = time.perf_counter()
            task.mode = 'eval_test'
            try:
                task.reset(seed=seed)
                task.play_once()
                if not task.check_success() or not task.plan_success:
                    raise ExecError(f'seed {seed} Expert check failed, check {task.check_success()}, plan {task.plan_success}.')
                else:
                    seed_status[seed] = True
                    with open(seed_path, 'w') as f:
                        json.dump(seed_status, f)
                test_cost = time.perf_counter() - test_start
                task.clean_cache(result='test_success')
                log(f'Expert check succ, seed {seed}, cost {test_cost:.2f}s')
            except Exception as e:
                test_cost = time.perf_counter() - test_start
                log(f'Expert check failed, seed {seed}, cost {test_cost:.2f}s, with exception {e}')
                task.clean_cache(result='test_fail')
                seed_status[seed] = False
                with open(seed_path, 'w') as f:
                    json.dump(seed_status, f)
                seed += 1
                continue
        test_num += 1

        succ = False
        eval_start = time.perf_counter()
        task.mode = 'eval'
        try:
            task.reset(seed=seed, instructions=instructions[instruciton_type])
            task.mean_steps = task.cfg.step_lim
            policy.reset()
            while task.take_action_cnt < task.cfg.step_lim:
                observation = task._get_observations()
                policy.eval(task, observation)
                if task.eval_success:
                    succ = True
                    break
                if getattr(task, 'force_stop', False):
                    task.metadata['force_stop_reason'] = getattr(task, 'force_stop_reason', None)
                    break
                if task.check_early_stop():
                    break
        except Exception as e:
            log(f"[{test_num:<3d}] Seed {seed} occurred exception: {e}\n{traceback.format_exc()}")
            succ_status = 'error'
            finalize_policy_rollout(policy, task, succ_status)
            task.clean_cache(result=succ_status)
            test_num -= 1
        else:
            eval_cost = time.perf_counter() - eval_start

            if succ:
                succ_num += 1
            succ_status = 'success' if succ else 'failed'
            finalize_policy_rollout(policy, task, succ_status)
            task.clean_cache(result=succ_status)
            log(f"[{test_num:<3d}] Seed {seed} {succ_status} after {eval_cost:.2f} s.\n"
                f"steps: {task.step_count:<5d}, actions: {task.take_action_cnt:<5d}.\n"
                f"Instruction: {task.instruction}\n"
                f"Total {succ_num}/{test_num}({succ_num/test_num*100:.2f}%) success.")
        finally:
            seed += 1

    return {
        'test_num': test_num,
        'succ_num': succ_num
    }

def get_config(file, default_root:Path, type:Literal['yaml', 'json']):
    input_path = Path(file)
    if type == 'yaml':
        if input_path.suffix in {'.yml', '.yaml'}:
            file = input_path
        elif input_path.is_absolute() or '/' in file:
            file = Path(f"{file}.yml")
        else:
            file = default_root / f'{file}.yml'
        if not file.is_absolute():
            file = REPO_ROOT / file
        if not file.exists() and default_root.name == 'task_config':
            experiment_name = os.environ.get('UNIVTAC_EXPERIMENT_NAME', os.environ.get('EXPERIMENT_NAME', 'BIGFOV_EXPERIMENT'))
            experiment_cfg = load_experiment_config(experiment_name)
            evaluation_cfg = experiment_cfg.get("evaluation", {})
            config = {
                "decimation": evaluation_cfg.get("decimation", 1),
                "save_frequency": evaluation_cfg.get("save_frequency", 2),
                "video_frequency": evaluation_cfg.get("video_frequency", 1),
                "render_frequency": evaluation_cfg.get("render_frequency", 0),
                "video_size": evaluation_cfg.get("video_size", [1280, 480]),
                "random_texture": evaluation_cfg.get("random_texture", False),
                "observations": evaluation_cfg.get("observations", DEFAULT_OBS_DATA_TYPE),
            }
            return config, file
        with open(file, 'r', encoding='utf-8') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config, file
    else:
        if input_path.suffix == '.json':
            file = input_path
        elif input_path.is_absolute() or '/' in file:
            file = Path(f"{file}.json")
        else:
            file = default_root / f'{file}.json'
        if not file.is_absolute():
            file = REPO_ROOT / file
        with open(file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config, file


def resolve_policy_module_name(policy_name: str) -> str:
    if policy_name.startswith("policy."):
        return policy_name

    first_segment = policy_name.split(".", 1)[0]
    if (policy_root / first_segment).exists() or (policy_root / f"{first_segment}.py").exists():
        return f"policy.{policy_name}"

    return policy_name

task_module, policy_module = None, None
def main():
    global args_cli, task_module, policy_module, log_path

    task_file_name = args_cli.task_name
    task_config, task_config_file = get_config(
        args_cli.task_config, default_root=REPO_ROOT / 'task_config', type='yaml'
    )
    deploy_config, deploy_config_file = get_config(
        args_cli.deploy_config, default_root=REPO_ROOT / 'policy', type='yaml'
    )
    policy_name = deploy_config['policy_name']
    deploy_config['task_name'] = task_file_name
    deploy_config['task_config'] = task_config_file.stem if task_config_file.exists() else 'clean'

    deploy_config['instuction_file'] = deploy_config.get('instuction_file', task_file_name)
    if deploy_config['instuction_file'] is not None:
        instructions, _ = get_config(
            deploy_config['instuction_file'], default_root=REPO_ROOT / 'instructions', type='json'
        )
    else:
        instructions = {'seen': ['Empty'], 'unseen': ['Empty']}

    task_module = importlib.import_module(f"envs.{task_file_name}")
    policy_module = importlib.import_module(resolve_policy_module_name(policy_name))

    curr_time = time.strftime(r'%Y-%m-%d_%H:%M:%S')

    env_cfg:BaseTaskCfg = task_module.TaskCfg()
    experiment_name = os.environ.get('UNIVTAC_EXPERIMENT_NAME', os.environ.get('EXPERIMENT_NAME', 'BIGFOV_EXPERIMENT'))
    experiment_cfg = load_experiment_config(experiment_name)
    save_dir_exact = os.environ.get('UNIVTAC_EVAL_SAVE_DIR')
    save_dir_prefix = os.environ.get('UNIVTAC_EVAL_SAVE_DIR_PREFIX')
    if save_dir_exact:
        env_cfg.save_dir = Path(save_dir_exact)
    elif save_dir_prefix:
        env_cfg.save_dir = Path(save_dir_prefix) / task_file_name / deploy_config_file.stem / curr_time
    else:
        env_cfg.save_dir = Path('eval_result') / policy_name / task_file_name / deploy_config_file.stem / curr_time
    env_cfg.tactile_sensor_type = experiment_cfg.get("tactile_sensor_type", env_cfg.tactile_sensor_type)
    env_cfg.decimation = task_config.get("decimation", env_cfg.decimation)
    env_cfg.obs_data_type = task_config.get("observations", {})
    env_cfg.save_frequency = task_config.get("save_frequency", env_cfg.save_frequency)
    env_cfg.video_frequency = task_config.get("video_frequency", env_cfg.video_frequency)
    env_cfg.render_frequency = task_config.get("render_frequency", env_cfg.render_frequency)
    env_cfg.video_size = tuple(task_config.get("video_size", env_cfg.video_size))
    env_cfg.random_texture = task_config.get("random_texture", False)
    reset_time_limit = os.environ.get("UNIVTAC_RESET_TIME_LIMIT")
    if reset_time_limit:
        env_cfg.reset_time_limit = float(reset_time_limit)

    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None \
        else env_cfg.sim.device
    seed = deploy_config.get("seed", 0)

    init_start = time.perf_counter()
    policy:BasePolicy = policy_module.Policy(deploy_config)
    policy_init_cost = time.perf_counter() - init_start

    init_start = time.perf_counter()
    task:BaseTask = task_module.Task(env_cfg, mode='eval')
    task_init_cost = time.perf_counter() - init_start

    if os.environ.get('TRAIN_CONFIG'):
        deploy_config['train_config'] = os.environ['TRAIN_CONFIG']

    log_path = task.save_root / f"log.log"
    log(f"Task Name: {task_file_name}")
    log(f"Task Config: {task_config_file.absolute()}")
    log(f"Eval Config: {json.dumps(deploy_config, ensure_ascii=False, indent=4)}\n{'-' * 20}\n")
    log(f"Task init finish in {task_init_cost:.2f} seconds.")
    log(f"Policy init finish in {policy_init_cost:.2f} seconds.")

    results = eval_policy(
        task=task, policy=policy,
        expert_check=args_cli.expert_check,
        start_seed=1000000 * (1 + seed) if args_cli.start_seed == -1 else args_cli.start_seed,
        max_seed=args_cli.max_seed,
        test_total_num=args_cli.total_num,
        instructions=instructions,
        instruciton_type=deploy_config.get("instruction_type", "seen")
    )
    log(f"Final Result: {results['succ_num']}/{results['test_num']}({results['succ_num']/results['test_num']*100:.2f}%) success.")

    task.close()
    policy.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
