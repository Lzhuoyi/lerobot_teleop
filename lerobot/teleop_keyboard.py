"""
Simple script to control a robot from teleoperation.

Example:

```shell
python -m lerobot.teleop_keyboard --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=robot1 --robot.cameras='{}' --display_data=true
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus
import numpy as np
import rerun as rr

from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.common.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.common.model.kinematics import RobotKinematics  # <-- Add this import

from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import init_logging, move_cursor_up
from lerobot.common.utils.visualization_utils import _init_rerun

# For Keyboard teleop:
import pygame as pg
import math
import sys

# This is a python way to create data-storing classes, similar to C++ structs.
# It automatically generates __init__, __repr__, __eq__, and other methods for the class
@dataclass
class TeleoperateConfig:
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False


def teleop_loop(
    robot: Robot, fps: int, display_data: bool = False, duration: float | None = None
):
    pg.init()
    screen_width, screen_height = 600, 750
    screen = pg.display.set_mode((screen_width, screen_height))
    pg.display.set_caption("Control Window - Use keys to move")
    font = pg.font.SysFont(None, 24)

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    gripper_pos = 0.0  # Start at zero pos

    # --- Initialize kinematics for SO-101 ---
    kinematics = RobotKinematics(robot_type="so_new_calibration")

    # --- Get initial joint positions and Action ---
    observation = robot.get_observation()
    joint_order = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
    joint_positions = np.array([observation[j] for j in joint_order], dtype=np.float32)
    initial_joint_positions = joint_positions.copy()  # <-- Store initial joint positions

    # Get Transformation Matrix for End-Effector (EE)
    ee_pose = kinematics.forward_kinematics(joint_positions, frame="gripper_tip")
    target_ee_pos = ee_pose[:3, 3].copy()  # Target EE position (XYZ)
    initial_ee_pos = target_ee_pos.copy()  # <-- Store initial EE position

    # Velocity-based teleop
    velocity = np.zeros(3, dtype=np.float32)  # vx, vy, vz
    vel_step = 0.05  # m/s, change as needed

    # For inidividual joint control:
    wrist_flex_vel = 0.0
    wrist_roll_vel = 0.0
    joint_vel_step = 151  # degree

    clock = pg.time.Clock()

    while True:
        loop_start = time.perf_counter()
        dt = clock.tick(fps) / 1000.0  # seconds since last loop

        # Init Action Dict
        action = {key: 0.0 for key in robot.action_features}

        # Reset velocity
        velocity[:] = 0.0
        wrist_flex_vel = 0.0
        wrist_roll_vel = 0.0

        # Handle key holds for velocity control
        keys = pg.key.get_pressed()
        if keys[pg.K_MINUS]:        # '-' decrease X
            velocity[0] -= vel_step
        if keys[pg.K_EQUALS]:       # '=' increase X
            velocity[0] += vel_step
        if keys[pg.K_LEFTBRACKET]:  # '[' decrease Y
            velocity[1] -= vel_step
        if keys[pg.K_RIGHTBRACKET]: # ']' increase Y
            velocity[1] += vel_step
        if keys[pg.K_SEMICOLON]:    # ';' decrease Z
            velocity[2] -= vel_step
        if keys[pg.K_QUOTE]:        # ''' increase Z
            velocity[2] += vel_step


        reset_flag = False  # <-- Add a flag for reset

        # Handle gripper open/close (discrete) and reset
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_COMMA:
                    gripper_pos -= 30
                if event.key == pg.K_PERIOD:
                    gripper_pos += 30
                if event.key == pg.K_r:  # <-- Reset on 'r'
                    reset_flag = True

        if reset_flag:
            # Set action to initial joint positions
            for i, joint in enumerate(joint_order):
                action[joint] = float(initial_joint_positions[i])
            action['gripper.pos'] = gripper_pos
            target_ee_pos = initial_ee_pos.copy()  # Reset target EE position

            robot.send_action(action)
            print("Arm reset to initial position.")
            # Optionally, wait a bit to let the arm move before continuing
            pg.time.wait(500)
            continue

        # Update target EE position by velocity * dt
        target_ee_pos += velocity * dt

        # Get current joint positions
        observation = robot.get_observation()
        joint_positions = np.array([observation[j] for j in joint_order], dtype=np.float32)

        # Compute current EE pose
        ee_pose = kinematics.forward_kinematics(joint_positions, frame="gripper_tip")

        # Build target pose (keep orientation, change position)
        target_pose = ee_pose.copy()
        target_pose[:3, 3] = target_ee_pos

        # Inverse kinematics to get new joint positions (position only)
        goal_joint_positions = kinematics.ik(joint_positions, target_pose, position_only=True, frame="gripper_tip")

        # Fill action dict with new joint positions
        for i, joint in enumerate(joint_order):
            if joint in action:
                action[joint] = float(goal_joint_positions[i])
        action['gripper.pos'] = gripper_pos

        # Check if we are controlling the robot with individual joint velocities
        joint_vel_keys = (
            keys[pg.K_q] or keys[pg.K_a] or
            keys[pg.K_w] or keys[pg.K_s] 
        )

        if joint_vel_keys:
            # Wrist flex (w/s)
            if keys[pg.K_q]:
                wrist_flex_vel += joint_vel_step
            if keys[pg.K_a]:
                wrist_flex_vel -= joint_vel_step
            # Wrist roll (e/d)
            if keys[pg.K_w]:
                wrist_roll_vel += joint_vel_step
            if keys[pg.K_s]:
                wrist_roll_vel -= joint_vel_step

            # Update joint positions with individual joint velocities
            observation = robot.get_observation()
            joint_positions = np.array([observation[j] for j in joint_order], dtype=np.float32)
            joint_positions[3] += wrist_flex_vel * dt    # wrist_flex.pos
            joint_positions[4] += wrist_roll_vel * dt    # wrist_roll.pos

            # Override only the manually controlled joints in the action dict
            action["wrist_flex.pos"] = float(joint_positions[3])
            action["wrist_roll.pos"] = float(joint_positions[4])

            # Update target EE pose to match new joint positions
            ee_pose_manual = kinematics.forward_kinematics(joint_positions, frame="gripper_tip")
            target_ee_pos = ee_pose_manual[:3, 3].copy()

        # Print EE position
        print(f"Target EE position (XYZ): {target_ee_pos}")
        print(f"Current EE position (XYZ): {ee_pose[:3, 3]}")

        # Optionally log data
        if display_data:
            for obs, val in observation.items():
                if isinstance(val, float):
                    rr.log(f"observation_{obs}", rr.Scalar(val))
                elif isinstance(val, np.ndarray):
                    rr.log(f"observation_{obs}", rr.Image(val), static=True)
            for act, val in action.items():
                if isinstance(val, float):
                    rr.log(f"action_{act}", rr.Scalar(val))

        robot.send_action(action)
        dt_s = time.perf_counter() - loop_start
        busy_wait(max(0, 1 / fps - dt_s))

        loop_s = time.perf_counter() - loop_start

        print("\n" + "-" * (display_len + 10))
        print(f"{'NAME':<{display_len}} | {'NORM':>7}")
        for motor, value in action.items():
            print(f"{motor:<{display_len}} | {value:>7.2f}")
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return

        move_cursor_up(len(action) + 15)  # Adjust for extra print lines

        
# The @draccus.wrap() decorator turns teleoperate(cfg: TeleoperateConfig) into a command-line interface.
# Under the hood, draccus.wrap():
# Parses sys.argv into a TeleoperateConfig.
@draccus.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        _init_rerun(session_name="teleoperation")

    robot = make_robot_from_config(cfg.robot)

    robot.connect()

    try:
        teleop_loop(robot, cfg.fps, display_data=cfg.display_data, duration=cfg.teleop_time_s)
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        robot.disconnect()


if __name__ == "__main__":
    teleoperate()
