import gymnasium as gym
import numpy as np
import sapien
import pygame
import time
import math
import cv2
from PIL import Image
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import RecordEpisode


import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PushCube-v1"
    """The environment ID of the task you want to simulate"""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "sensor_data"
    """Observation mode - changed to sensor_data to get camera images"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
    """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = None
    """Control mode"""

    render_mode: str = "rgb_array"
    """Render mode - using rgb_array to get camera images"""

    shader: str = "default"
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    record_dir: Optional[str] = None
    """Directory to save recordings"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = None
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""

def get_mapped_joints(robot):
    """
    Get the current joint positions from the robot and map them correctly to the target joints.
    
    The mapping is:
    - full_joints[0,2] → current_joints[0,1] (base x position and base rotation)
    - full_joints[3,6,9,11,13] → current_joints[2,3,4,5,6] (first arm joints)
    - full_joints[4,7,10,12,14] → current_joints[7,8,9,10,11] (second arm joints)
    
    Returns:
        np.ndarray: Mapped joint positions with shape matching the target_joints
    """
    if robot is None:
        return np.zeros(16)  # Default size for action
        
    # Get full joint positions
    full_joints = robot.get_qpos()
    
    # Convert tensor to numpy array if needed
    if hasattr(full_joints, 'numpy'):
        full_joints = full_joints.numpy()
    
    # Handle case where it's a 2D tensor/array
    if full_joints.ndim > 1:
        full_joints = full_joints.squeeze()
    
    # Create the mapped joints array with correct size
    mapped_joints = np.zeros(16)
    
    # Map the joints according to the specified mapping
    if len(full_joints) >= 15:
        # Base joints: [0,2] → [0,1]
        mapped_joints[0] = full_joints[0]  # Base X position
        mapped_joints[1] = full_joints[2]  # Base rotation
        
        # First arm: [3,6,9,11,13] → [2,3,4,5,6]
        mapped_joints[2] = full_joints[3]
        mapped_joints[3] = full_joints[6]
        mapped_joints[4] = full_joints[9]
        mapped_joints[5] = full_joints[11]
        mapped_joints[6] = full_joints[13]
        
        # Second arm: [4,7,10,12,14] → [7,8,9,10,11]
        mapped_joints[7] = full_joints[4]
        mapped_joints[8] = full_joints[7]
        mapped_joints[9] = full_joints[10]
        mapped_joints[10] = full_joints[12]
        mapped_joints[11] = full_joints[14]
        mapped_joints[12] = full_joints[15]
        mapped_joints[13] = full_joints[16]

        mapped_joints[14] = full_joints[5]
        mapped_joints[15] = full_joints[8]
    
    return mapped_joints

def inverse_kinematics(x, y, l1=0.1159, l2=0.1350):
    """
    Calculate inverse kinematics for a 2-link robotic arm, considering joint offsets
    
    Parameters:
        x: End effector x coordinate
        y: End effector y coordinate
        l1: Upper arm length (default 0.1159 m)
        l2: Lower arm length (default 0.1350 m)
        
    Returns:
        joint2, joint3: Joint angles in radians as defined in the URDF file
    """
    # Calculate joint2 and joint3 offsets in theta1 and theta2
    theta1_offset = math.atan2(0.028, 0.11257)  # theta1 offset when joint2=0
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset  # theta2 offset when joint3=0
    
    # Calculate distance from origin to target point
    r = math.sqrt(x**2 + y**2)
    r_max = l1 + l2  # Maximum reachable distance
    
    # If target point is beyond maximum workspace, scale it to the boundary
    if r > r_max:
        scale_factor = r_max / r
        x *= scale_factor
        y *= scale_factor
        r = r_max
    
    # If target point is less than minimum workspace (|l1-l2|), scale it
    r_min = abs(l1 - l2)
    if r < r_min and r > 0:
        scale_factor = r_min / r
        x *= scale_factor
        y *= scale_factor
        r = r_min
    
    # Use law of cosines to calculate theta2
    cos_theta2 = -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    
    # Calculate theta2 (elbow angle)
    theta2 = math.pi - math.acos(cos_theta2)
    
    # Calculate theta1 (shoulder angle)
    beta = math.atan2(y, x)
    gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta + gamma
    
    # Convert theta1 and theta2 to joint2 and joint3 angles
    joint2 = theta1 + theta1_offset
    joint3 = theta2 + theta2_offset
    
    # Ensure angles are within URDF limits
    joint2 = max(-0.1, min(3.45, joint2))
    joint3 = max(-0.2, min(math.pi, joint3))
    
    return joint2, joint3

def convert_tensor_to_numpy_image(tensor_image):
    """Convert tensor image to numpy array for display"""
    # Handle PyTorch tensors
    if hasattr(tensor_image, 'cpu'):
        # Move tensor to CPU first if it's on GPU
        tensor_image = tensor_image.cpu()
    
    if hasattr(tensor_image, 'numpy'):
        image = tensor_image.numpy()
    else:
        image = tensor_image
    
    # Handle different image formats
    if image.dtype == np.float32 or image.dtype == np.float64:
        # Normalize to 0-255 range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Remove batch dimension if present
    if image.ndim == 4:
        image = image[0]  # Take first batch
    
    # Convert RGBA to RGB if needed
    if image.shape[-1] == 4:
        image = image[..., :3]
    
    return image

def main(args: Args):
    pygame.init()
    pygame.joystick.init()
    
    # Check controller connection
    if pygame.joystick.get_count() == 0:
        print("No controller detected, please ensure Xbox controller is connected")
        return
    
    # Get the first controller
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Connected controller: {joystick.get_name()}")
    print(f"Number of buttons: {joystick.get_numbuttons()}")
    print(f"Number of axes: {joystick.get_numaxes()}")
    print(f"Number of hats: {joystick.get_numhats()}")
    
    # Remove Pygame interface display, only keep controller input reading
    
    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene,
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
        if len(env_kwargs["robot_uids"]) == 1:
            env_kwargs["robot_uids"] = env_kwargs["robot_uids"][0]
    env: BaseEnv = gym.make(
        args.env_id,
        **env_kwargs
    )
    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=gym_utils.find_max_episode_steps_value(env))

    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        if env.unwrapped.agent is not None:
            print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)

    obs, _ = env.reset(seed=args.seed, options=dict(reconfigure=True))
    if args.seed is not None and env.action_space is not None:
            env.action_space.seed(args.seed[0])
    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()
    
    action = env.action_space.sample() if env.action_space is not None else None
    action = np.zeros_like(action)
    
    # Initialize target joint positions with zeros
    target_joints = np.zeros_like(action)
    target_joints[3] = 0.303
    target_joints[4] = 0.556
    target_joints[6] = 1.57  # Set joint 5 for first arm to 1.57
    target_joints[8] = 0.303
    target_joints[9] = 0.556
    target_joints[11] = 1.57  # Set joint 5 for second arm to 1.57

    # Initialize end effector positions for both arms
    initial_ee_pos_arm1 = np.array([0.247, -0.023])  # Initial position for first arm
    initial_ee_pos_arm2 = np.array([0.247, -0.023])  # Initial position for second arm
    ee_pos_arm1 = initial_ee_pos_arm1.copy()
    ee_pos_arm2 = initial_ee_pos_arm2.copy()
    
    # Initialize pitch adjustments for end effector orientation
    initial_pitch_1 = 0.0  # Initial pitch adjustment for first arm
    initial_pitch_2 = 0.0  # Initial pitch adjustment for second arm
    pitch_1 = initial_pitch_1
    pitch_2 = initial_pitch_2
    pitch_step = 0.02  # Step size for pitch adjustment
    
    # Define tip length for vertical position compensation
    tip_length = 0.108  # Length from wrist to end effector tip
    
    # Define the step size for changing target joints and end effector positions
    joint_step = 0.01
    ee_step = 0.005  # Step size for end effector position control
    
    # Define the gain for the proportional controller as a list for each joint
    p_gain = np.ones_like(action)  # Default all gains to 1.0
    # Specific gains can be adjusted here
    p_gain[0] = 2     # Base forward/backward
    p_gain[1] = 0.5     # Base rotation - lower gain for smoother turning
    p_gain[2:7] = 1.0   # First arm joints
    p_gain[7:12] = 1.0  # Second arm joints
    p_gain[12:14] = 0.05  # Gripper joints
    p_gain[14:16] = 2  # Head motor joints
    
    # Get initial joint positions if available
    current_joints = np.zeros_like(action)
    robot = None
    
    # Try to get the robot instance for direct access
    if hasattr(env.unwrapped, "agent"):
        robot = env.unwrapped.agent.robot
    elif hasattr(env.unwrapped, "agents") and len(env.unwrapped.agents) > 0:
        robot = env.unwrapped.agents[0]  # Get the first robot if multiple exist
    
    print("robot", robot)
    
    # Get the correctly mapped joints
    current_joints = get_mapped_joints(robot)
    
    # Ensure target_joints is a numpy array with the same shape as current_joints
    target_joints = np.zeros_like(current_joints)
    target_joints[6] = 1.57 
    target_joints[11] = 1.57 
    
    # Set initial joint positions based on inverse kinematics from initial end effector positions
    try:
        target_joints[3], target_joints[4] = inverse_kinematics(ee_pos_arm1[0], ee_pos_arm1[1])
        target_joints[8], target_joints[9] = inverse_kinematics(ee_pos_arm2[0], ee_pos_arm2[1])
    except Exception as e:
        print(f"Error calculating initial inverse kinematics: {e}")
    
    # Add step counter for warmup phase
    step_counter = 0
    warmup_steps = 50
    
    # Controller control parameters
    joystick_sensitivity = 0.8  # Joystick sensitivity
    trigger_sensitivity = 0.3   # Trigger sensitivity
    deadzone = 0.1             # Deadzone size
    
    while True:
        # Remove Pygame event handling, only keep controller input reading
        
        # Use direct reading method from test_xbox
        pygame.event.pump()
        
        # Get axis states
        right_stick_x = joystick.get_axis(3)  # Right stick X axis
        right_stick_y = joystick.get_axis(4)  # Right stick Y axis
        right_trigger = joystick.get_axis(5)  # Right trigger
        left_stick_x = joystick.get_axis(0)   # Left stick X axis
        left_stick_y = joystick.get_axis(1)   # Left stick Y axis
        left_trigger = joystick.get_axis(2)   # Left trigger
        
        # Apply deadzone
        if abs(right_stick_x) < deadzone: right_stick_x = 0.0
        if abs(right_stick_y) < deadzone: right_stick_y = 0.0
        if abs(right_trigger) < deadzone: right_trigger = 0.0
        if abs(left_stick_x) < deadzone: left_stick_x = 0.0
        if abs(left_stick_y) < deadzone: left_stick_y = 0.0
        if abs(left_trigger) < deadzone: left_trigger = 0.0
        
        # Get button states
        a_button = joystick.get_button(0)  # A button
        b_button = joystick.get_button(1)  # B button
        x_button = joystick.get_button(2)  # X button
        y_button = joystick.get_button(3)  # Y button
        lb_pressed = joystick.get_button(4)  # LB button
        rb_pressed = joystick.get_button(5)  # RB button
        start_button = joystick.get_button(7)  # Start button
        left_stick_pressed = joystick.get_button(9)  # Left stick pressed
        right_stick_pressed = joystick.get_button(10)  # Right stick pressed
        
        # Get hat (d-pad) state
        hat = joystick.get_hat(0)
        hat_x, hat_y = hat
        
        # Check reset button (Start button)
        if start_button:
            # Reset end effector positions
            ee_pos_arm1 = initial_ee_pos_arm1.copy()
            ee_pos_arm2 = initial_ee_pos_arm2.copy()
            
            # Reset pitch adjustments
            pitch_1 = initial_pitch_1
            pitch_2 = initial_pitch_2
            
            # Reset target joints
            target_joints = np.zeros_like(target_joints)
            
            # Calculate initial joint positions based on inverse kinematics
            try:
                compensated_y1 = ee_pos_arm1[1] - tip_length * math.sin(pitch_1)
                target_joints[3], target_joints[4] = inverse_kinematics(ee_pos_arm1[0], compensated_y1)
                
                compensated_y2 = ee_pos_arm2[1] - tip_length * math.sin(pitch_2)
                target_joints[8], target_joints[9] = inverse_kinematics(ee_pos_arm2[0], compensated_y2)
                
                # Apply pitch adjustment to joint 5 and 10
                target_joints[5] = target_joints[3] - target_joints[4] + pitch_1
                target_joints[10] = target_joints[8] - target_joints[9] + pitch_2
            except Exception as e:
                print(f"Error calculating inverse kinematics during reset: {e}")
            
            print("All positions reset to initial values")
        
        # Update target joint positions based on controller input - only after warmup
        if step_counter >= warmup_steps:
            # Base control - using d-pad
            if hat_y == -1:  # Up
                action[0] = -0.1  # Forward
            elif hat_y == 1:  # Down
                action[0] = 0.1  # Backward
            else:
                action[0] = 0.0  # Stop forward/backward movement
                
            # Base turning - using d-pad left/right
            if hat_x == -1:  # Left
                target_joints[1] += joint_step * 2  # Turn left
            elif hat_x == 1:  # Right
                target_joints[1] -= joint_step * 2  # Turn right
            
            # Arm control logic
            if not right_stick_pressed and not rb_pressed:
                # Right stick controls left arm up/down and forward/backward
                if abs(right_stick_y) > 0.1:  # Up/down control
                    ee_pos_arm1[1] -= right_stick_y * ee_step * joystick_sensitivity
                if abs(right_stick_x) > 0.1:  # Forward/backward control
                    ee_pos_arm1[0] += right_stick_x * ee_step * joystick_sensitivity
                    
                # Calculate left arm inverse kinematics
                
                # target_joints[5] = target_joints[3] - target_joints[4] + pitch_1
            
            if not left_stick_pressed and not lb_pressed:
                # Left stick controls right arm up/down and forward/backward
                if abs(left_stick_y) > 0.1:  # Up/down control
                    ee_pos_arm2[1] -= left_stick_y * ee_step * joystick_sensitivity
                if abs(left_stick_x) > 0.1:  # Forward/backward control
                    ee_pos_arm2[0] += left_stick_x * ee_step * joystick_sensitivity
                    
                
            
            # Left stick pressed controls left arm first joint
            if left_stick_pressed:
                print("left_stick_pressed")
                if abs(left_stick_x) > 0.1:
                    target_joints[7] += left_stick_x * joint_step * joystick_sensitivity
            
            # Right stick pressed controls right arm first joint
            if right_stick_pressed:
                print("right_stick_pressed")
                if abs(right_stick_x) > 0.1:
                    target_joints[2] += right_stick_x * joint_step * joystick_sensitivity
            
            # LB pressed controls left arm pitch and wrist roll
            if lb_pressed:
                print("lb_pressed")
                if abs(left_stick_y) > 0.1:  # Up/down controls pitch
                    pitch_2 += left_stick_y * pitch_step * joystick_sensitivity
                if abs(left_stick_x) > 0.1:  # Left/right controls wrist roll
                    target_joints[11] += left_stick_x * joint_step * 3 * joystick_sensitivity
                
                # Apply pitch adjustment to joint 5
            target_joints[5] = target_joints[3] - target_joints[4] + pitch_1
            compensated_y = ee_pos_arm1[1] + tip_length * math.sin(pitch_1)
            target_joints[3], target_joints[4] = inverse_kinematics(ee_pos_arm1[0], compensated_y)
            
            # RB pressed controls right arm pitch and wrist roll
            if rb_pressed:
                print("rb_pressed")
                if abs(right_stick_y) > 0.1:  # Up/down controls pitch
                    pitch_1 += right_stick_y * pitch_step * joystick_sensitivity
                if abs(right_stick_x) > 0.1:  # Left/right controls wrist roll
                    target_joints[6] += right_stick_x * joint_step * 3 * joystick_sensitivity
                
                # Apply pitch adjustment to joint 10
            target_joints[10] = target_joints[8] - target_joints[9] + pitch_2
            # Calculate right arm inverse kinematics
            compensated_y = ee_pos_arm2[1] + tip_length * math.sin(pitch_2)
            target_joints[8], target_joints[9] = inverse_kinematics(ee_pos_arm2[0], compensated_y)

        
            # Trigger controls gripper - press to close, release to open
            if abs(left_trigger) > trigger_sensitivity:
                # Left trigger controls left gripper - press to close(0.1), release to open(2.5)
                target_joints[12] = 0.1 if left_trigger > 0 else 2.5
            
            if abs(right_trigger) > trigger_sensitivity:
                # Right trigger controls right gripper - press to close(0.1), release to open(2.5)
                target_joints[13] = 0.1 if right_trigger > 0 else 2.5
            
            # ABXY controls head
            if a_button:  # A button - head up/down
                target_joints[15] += joint_step * 2
            if b_button:  # B button - head up/down
                target_joints[14] -= joint_step * 2
            if x_button:  # X button - head left/right
                target_joints[14] += joint_step * 2
            if y_button:  # Y button - head left/right
                target_joints[15] -= joint_step * 2
        
        # Get current joint positions using our mapping function
        current_joints = get_mapped_joints(robot)
        
        # Simple P controller for arm joints only (not base)
        if step_counter < warmup_steps:
            action = np.zeros_like(action)
        else:
            # Apply P control to turning (index 1) and arm joints (indices 2-11) and grippers (indices 12-13)
            # Base forward/backward (index 0) is already set directly above
            for i in range(1, len(action)):
                action[i] = p_gain[i] * (target_joints[i] - current_joints[i])
        
        # Remove Pygame interface display, only keep controller input reading functionality
        
        obs, reward, terminated, truncated, info = env.step(action)
        step_counter += 1
        
        if args.render_mode is not None:
            env.render()
        
        time.sleep(0.01)
        
        if args.render_mode is None or args.render_mode != "human":
            if (terminated | truncated).any():
                break
    
    env.close()

    if record_dir:
        print(f"Saving video to {record_dir}")


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args) 