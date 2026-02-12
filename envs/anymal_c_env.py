# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

# ----------- IMPORT FOR MARKERS 
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

def define_markers() -> VisualizationMarkers:
        """Define markers with various different shapes."""
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/myMarkers",
            markers={
                    "forward": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                        scale=(0.25, 0.25, 0.5),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                    ),
                    "command": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                        scale=(0.25, 0.25, 0.5),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
            },
        )
        return VisualizationMarkers(cfg=marker_cfg)
# -------------------------------------
import gymnasium as gym
import torch
import torch.nn as nn

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster, Camera

from .anymal_c_env_cfg import AnymalCFlatEnvCfg, AnymalCRoughEnvCfg


class AnymalCEnv(DirectRLEnv):
    cfg: AnymalCFlatEnvCfg | AnymalCRoughEnvCfg

    def __init__(self, cfg: AnymalCFlatEnvCfg | AnymalCRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # v------------------------------------------------ WS: Debug counter for printing DEBUG every N
        self.episode_count = 0
        self.is_debug = 0
        self.debug_counter = 0
        self.N = 100
        # ^----------------------------------------------------------------------------------------------
        # v------------------------------------------------ WS: Add min_vel and max_vel for bound commands
        self.min_vel = 0.5
        self.max_vel = 2.0
        # ^------------------------------------------------------------------------------------------------

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                # v--------------------------------------- WS: ADDED TERMS, since Bound training
                "bound_gait",
                "hind_leg_symmetry",
                "front_leg_symmetry",
                "hip_lateral_penalty",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*FOOT")
        print(f"[WS_INFO]: self._feet_ids: {self._feet_ids}")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*THIGH")

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        #self._camera = Camera(self.cfg.camera)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        #self.scene.sensors["camera"] = self._camera
        if isinstance(self.cfg, AnymalCRoughEnvCfg):
            # we add a height scanner for perceptive locomotion
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # v--------------------------------------------------------- WS: Add markers
        self.visualization_markers = define_markers()
        # setting aside useful variables for later
        self.up_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)#.cuda()
        self.yaws = torch.zeros((self.cfg.scene.num_envs, 1), device=self.device)#.cuda()
        self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3), device=self.device)#.cuda()
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3), device=self.device)#.cuda()
        self.marker_offset[:,-1] = 0.5
        self.forward_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4), device=self.device)#.cuda()
        self.command_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4), device=self.device)#.cuda()

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos
        self._visualize_markers()   # <--- WS: Visualize markers
        eye_pos = torch.tensor([15.0, 0.0, 10.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        #self._camera.set_world_poses_from_view(eye_pos, self._robot.data.root_link_pos_w)

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        height_data = None
        if isinstance(self.cfg, AnymalCRoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    height_data,
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        # v---------------------------------------- WS Added bound reward
        
        bound_gait = self._get_bound_gait_reward()
        # ----------------------------------------------------------------
        
        # v------------------------------------------------------------------------------ WS: Override lin_vel_error
        # Project current velocity onto commanded direction (dot product)
        cmd_dir = nn.functional.normalize(self._commands[:, :2], dim=1)
        current_vel = self._robot.data.root_lin_vel_b[:, :2]
        projected_vel = torch.sum(current_vel * cmd_dir, dim=1)

        # Penalize deviation from desired speed in the desired direction
        speed_error = torch.square(torch.norm(self._commands[:, :2], dim=1) - projected_vel)
        lin_vel_error_mapped = torch.exp(-speed_error / 0.25)
        # ^-------------------------------------------------------------------------------------------------------
        
        # v------------------------------------------------------------------------------ WS: Add symmetry reward for HLegs
        # Assuming (indeces of hind joints):
        LH_ids = [3, 4, 5]
        RH_ids = [9, 10, 11]

        lh_joint_pos = self._robot.data.joint_pos[:, LH_ids]
        rh_joint_pos = self._robot.data.joint_pos[:, RH_ids]

        symmetry_reward_hind = torch.exp(-torch.norm(lh_joint_pos - rh_joint_pos, dim=1))
        # ^-------------------------------------------------------------------------------------------------------
        # v------------------------------------------------------------------------------ WS: Add symmetry reward for FLegs
        # Assuming (indeces of front joints):
        LF_ids = [0, 1, 2]
        RF_ids = [6, 7, 8]

        lf_joint_pos = self._robot.data.joint_pos[:, LF_ids]
        rf_joint_pos = self._robot.data.joint_pos[:, RF_ids]

        symmetry_reward_front = torch.exp(-torch.norm(lf_joint_pos - rf_joint_pos, dim=1))
        # ^-------------------------------------------------------------------------------------------------------
        # v------------------------------------------------------------------------------ WS: Add penalty for hip abduction
        # Penalize lateral joint angles (e.g., HAA joints at indices 0, 3, 6, 9 for LF, RF, LH, RH)
        haa_indices = torch.tensor([0, 6], device=self.device)
        haa_pos = self._robot.data.joint_pos[:, haa_indices]
        hip_lateral_penalty = torch.sum(torch.square(haa_pos), dim=1)
        # ^-------------------------------------------------------------------------------------------------------
        if self.is_debug and self.debug_counter%self.N == 0:
            print("[DEBUG] projected_vel mean:", projected_vel.mean().item())
            print("[DEBUG] lin_vel_reward mean:", lin_vel_error_mapped.mean().item())
            print("[DEBUG] hind symmetry mean:", symmetry_reward_hind.mean().item())
            print("[DEBUG] front symmetry mean:", symmetry_reward_front.mean().item())
            print("[DEBUG] hip lateral mean:", hip_lateral_penalty.mean().item())
            print("[DEBUG] lin_vel_cmd:", self._commands[:, 0].mean().item(), "vel_actual:", self._robot.data.root_lin_vel_b[:, 0].mean().item())

        self.debug_counter += 1
        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            # v---------------------------------------- WS Added bound reward
            #"bound_gait": bound_gait * self.cfg.bound_gait_reward_scale * self.step_dt,
            # v---------------------------------------- WS Added hind legs symmetry
            #"hind_leg_symmetry": symmetry_reward_hind * self.cfg.hind_symmetry_reward_scale * self.step_dt,
            # v---------------------------------------- WS Added front legs symmetry
            #"front_leg_symmetry": symmetry_reward_front * self.cfg.hind_symmetry_reward_scale * self.step_dt,
            # v---------------------------------------- WS Added front legs symmetry
            #"hip_lateral_penalty": hip_lateral_penalty * self.cfg.hip_lateral_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        # Never reset due to termination or timeout
        #died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        #time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new commands
        # OLD (randomized):
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        self._commands[:,-1] = 0    # Set yaw_rate=0
        # v------------------------------------------------------------------------------------------------------------
        # NEW (fixed command) [given wrt robot frame]:
        self._commands[env_ids] = torch.tensor([2.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        # BOUND commands
        #self.curriculum_speed = min(self.min_vel + self.episode_count / 3000, self.max_vel)
        #self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(self.min_vel, self.curriculum_speed)
        #self._commands[:,1:] = 0    # Set y_dot and yaw_rate = 0

        # ^------------------------------------------------------------------------------------------------------------
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # v--------------------------------------------------------------------------------------- WS: Update markers
        self._visualize_markers()
        # _commands: will be used as command for robot; command: is the direction versor of _commands
        self.commands = self._commands/torch.linalg.norm(self._commands, dim=1, keepdim=True)
        # offsets to account for atan range and keep things on [-pi, pi]
        ratio = self.commands[:,1]/(self.commands[:,0]+1E-8)
        gzero = torch.where(self.commands > 0, True, False)
        lzero = torch.where(self.commands < 0, True, False)
        plus = lzero[:,0]*gzero[:,1]
        minus = lzero[:,0]*lzero[:,1]
        offsets = torch.pi*plus - torch.pi*minus
        self.yaws = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1) + torch.pi/2

        self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3), device=self.device)#.cuda()
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3), device=self.device)#.cuda()
        self.marker_offset[:,-1] = 0.5
        self.forward_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4), device=self.device)#.cuda()
        self.command_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4), device=self.device)#.cuda()
        # ^ -----------------------------------------------------------------------------------------
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
        # v---------- WS: increment episode_count
        self.episode_count += 1
        #print("[DEBUG] EPISODE COUNT:", self.episode_count)


    # v------------------------------------------- WS
    def _visualize_markers(self):
        # get marker locations and orientations
        self.marker_locations = self._robot.data.root_pos_w
        curr_yaws = self._compute_curr_yaws()
        self.forward_marker_orientations = math_utils.quat_from_angle_axis(curr_yaws, self.up_dir).squeeze()
        self.command_marker_orientations = math_utils.quat_from_angle_axis(self.yaws, self.up_dir).squeeze()

        # offset markers so they are above the robot
        loc = self.marker_locations + self.marker_offset
        loc = torch.vstack((loc, loc))
        rots = torch.vstack((self.forward_marker_orientations, self.command_marker_orientations))

        # render the markers
        all_envs = torch.arange(self.cfg.scene.num_envs)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))
        self.visualization_markers.visualize(loc, rots, marker_indices=indices)

    def _compute_curr_yaws(self):
        curr_vel = self._robot.data.root_lin_vel_w[:, :2]
        ratio = curr_vel[:,1]/(curr_vel[:,0]+1E-8)
        gzero = torch.where(curr_vel > 0, True, False)
        lzero = torch.where(curr_vel < 0, True, False)
        plus = lzero[:,0]*gzero[:,1]
        minus = lzero[:,0]*lzero[:,1]
        offsets = torch.pi*plus - torch.pi*minus
        curr_yaws = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)
        return curr_yaws
    # ^--------------------------------------------------------------- WS: We are defining visualization markers
    # v--------------------------------------------------------------- WS: additional reward
    def _get_bound_gait_reward(self) -> torch.Tensor:
        # Get contact forces: shape (N_envs, T_history, Bodies_per_sensor, 3)
        net_contact_forces = self._contact_sensor.data.net_forces_w_history

        # Compute the norm of contact forces. Take maximum along temporal history (dim=1)
        # Verify if over threshold to determine contact.
        is_contact = torch.max(torch.norm(net_contact_forces[:, :, self._feet_ids], dim=-1), dim=1)[0] > 1.0

        # Convert booleans (True/False) to float (1.0/0.0) for following computations
        contacts = is_contact.float()

        # Nominal bound gait: 
        # - front feet (LF=0, RF=2) in phase -> same contact state
        # - hind feet (LH=1, RH=3) in phase
        # - front and hind out of phase -> difference between their average contact
        contact_anterior_L = contacts[:, 0]
        contact_anterior_R = contacts[:, 2]
        contact_posterior_L = contacts[:, 1]
        contact_posterior_R = contacts[:, 3]

        # 1. Reward for leg pair synchronization 
        # i.e., if feet of the same pair are in the same state (both on ground or both in air)
        sync_front = torch.exp(-torch.square(contact_anterior_L - contact_anterior_R))
        sync_hind = torch.exp(-torch.square(contact_posterior_L - contact_posterior_R))
        sync_reward = (sync_front + sync_hind) / 2.0

        # 2. Reward for leg pair desynchronization
        # i.e., if one leg pair is on ground while the other is in air
        phase_front = (contact_anterior_L + contact_anterior_R) / 2.0
        phase_hind = (contact_posterior_L + contact_posterior_R) / 2.0
        # Get reward when the sum of the phases is close do 1 (one pair is 1, the other is 0)
        desync_reward = torch.exp(-torch.square(phase_front + phase_hind - 1.0))

        # 3. Reward for flight phase (all feet in air)
        # i.e., adds a strong reward when the agent is in a complete flight phase (crucial for dynamic gaits)
        num_feet_in_contact = torch.sum(is_contact, dim=1)
        flight_reward = (num_feet_in_contact == 0)

        # Combine bound rewards
        # Apply only at high velocities, where bound gait makes sense
        #is_moving_fast = torch.norm(self._commands[:, :2], dim=1) > 2.0 # Discontinuous threshold
        speed_mag = torch.norm(self._commands[:, :2], dim=1)
        speed_weight = torch.clamp(speed_mag / 2.0, 0.0, 1.0)  # Smooth transition
        bound_reward = (sync_reward + desync_reward + flight_reward) * speed_weight
        #bound_reward = sync_reward * desync_reward * flight_reward * is_moving_fast
        # Total gait reward: encourage front/back sync, and front-hind alternation
        if self.is_debug and self.debug_counter%self.N == 0:
            print("[DEBUG] mean sync:", sync_reward.mean().item(), "desync:", desync_reward.mean().item(), "flight:", flight_reward.float().mean().item())
        return bound_reward
    # ^-----------------------------------------------------------------------------------------------