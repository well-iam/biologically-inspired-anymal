# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# add WS argparse
parser.add_argument("--sim_steps", type=int, default=1000, help="Number of simulation steps to run.")
parser.add_argument("--log_active", action="store_true", default=False, help="Log robot data during play.")



# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
algorithm = args_cli.algorithm.lower()


def main():
    """Play with skrl agent."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    task_name = args_cli.task.split(":")[-1]

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    try:
        experiment_cfg = load_cfg_from_registry(task_name, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(task_name, "skrl_cfg_entry_point")

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
        print(f"[INFO_WS] Loading checkpoint from directory: {resume_path}")
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    print(f"[INFO_WS] log_dir is: {log_dir}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": False,
        }
        print(f"[INFO_WS] Saving video in dir: {video_kwargs['video_folder']}")
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    robot = env.unwrapped.scene.articulations["robot"]
    contact_sensor = env.unwrapped.scene.sensors["contact_sensor"]
    # from omni.isaac.core.utils.viewports import set_camera_view, get_viewport_interface

    # # Get the viewport interface
    # viewport = get_viewport_interface()
    # viewport.set_active_viewport("Viewport")

    # # Set the camera to look at the robot (hardcoded offset example)
    # set_camera_view(
    #     eye=[3.0, 0.0, 2.0],              # camera position (x, y, z)
    #     target=[0.0, 0.0, 0.5]            # what the camera looks at
    # )
    timestep = 0

    # Define log data (TODO: Pre-allocate)
    if args_cli.log_active:
        # Time
        logged_time = []
        # Joints
        logged_joint_pos = []
        logged_joint_vel = []
        logged_computed_torque = []
        logged_applied_torque = []
        logged_joint_pos_target = []
        logged_joint_vel_target = []
        logged_joint_effort_target = []
        joint_pos_limits = robot.data.joint_pos_limits.cpu().numpy()
        joint_vel_limits = robot.data.joint_vel_limits.cpu().numpy()
        joint_effort_limits = robot.data.joint_effort_limits.cpu().numpy()
        # Root link
        logged_root_link_state_w = []
        logged_root_com_state_w = []
        # GR Forces
        logged_gr_forces = []
        # Command and Actions
        logged_commands = []
        logged_actions = []

    step_counter = 0
    # simulate environment
    while simulation_app.is_running(): #and step_counter <= args_cli.sim_steps:
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)

        # WS
        if args_cli.log_active:
            # example for single robot, single environment
            #print(f"[DEBUG] Object registry keys: {env.unwrapped.scene.articulations.keys()}")
            # you may need to adapt based on your robot's class
            # Log Time
            logged_time.append(step_counter*dt)
            # Log Joints
            print(f"Joint values: {robot.data.joint_pos.cpu().numpy()}")
            logged_joint_pos.append(robot.data.joint_pos.cpu().numpy())
            logged_joint_vel.append(robot.data.joint_vel.cpu().numpy())
            logged_computed_torque.append(robot.data.computed_torque.cpu().numpy())
            logged_applied_torque.append(robot.data.applied_torque.cpu().numpy())
            logged_joint_pos_target.append(robot.data.joint_pos_target.cpu().numpy())
            logged_joint_vel_target.append(robot.data.joint_vel_target.cpu().numpy())
            logged_joint_effort_target.append(robot.data.joint_effort_target.cpu().numpy())
            # Log Root Link
            logged_root_link_state_w.append(robot.data.root_link_state_w.cpu().numpy())
            logged_root_com_state_w.append(robot.data.root_com_state_w.cpu().numpy())
            # Log GR Forces
            logged_gr_forces.append(contact_sensor.data.net_forces_w.cpu().numpy())
            # TODO: Log Command and Actions
            logged_commands.append(env.unwrapped._commands.cpu().numpy())
            logged_actions.append(actions.cpu().numpy())

        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
        step_counter += 1

    ### WS: Save logged data in files
    if args_cli.log_active:
        import numpy as np
        from scipy.io import savemat
        # stack and save
        data = {
            "time": np.stack(logged_time),

            "joint_pos_limits": np.stack(joint_pos_limits),
            "joint_vel_limits": np.stack(joint_vel_limits),
            "joint_effort_limits": np.stack(joint_effort_limits),
            "joint_pos": np.stack(logged_joint_pos),
            "joint_vel": np.stack(logged_joint_vel),
            "computed_torque": np.stack(logged_computed_torque),
            "applied_torque": np.stack(logged_applied_torque),
            "joint_pos_target": np.stack(logged_joint_pos_target),
            "joint_vel_target": np.stack(logged_joint_vel_target),
            "joint_effort_target": np.stack(logged_joint_effort_target),

            "root_link_state_w": np.stack(logged_root_link_state_w),
            "root_com_state_w": np.stack(logged_root_com_state_w),

            "contact_forces": np.stack(logged_gr_forces),

            "commands": np.stack(logged_commands),
            "actions": np.stack(logged_actions)
        }
        savemat("isaaclab_robot_data_flat_curriculum.mat", data)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
