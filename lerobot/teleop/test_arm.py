"""
Simple script to control a robot from teleoperation.

Example:

```shell
python -m lerobot.test_arm --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=robot1 --robot.cameras='{}' --display_data=true
```
"""

import logging
import time
import numpy as np
import pygame as pg
import math
import sys
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus

from lerobot.common.robots import (
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
from lerobot.kin import ArmKinematics

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


def move_to(
    robot: Robot, fps: int, display_data: bool = False, duration: float | None = None
):
    # --- Kinematics ---
    arm_kin = ArmKinematics()
    joint_order = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]

    # --- Initial State ---
    observation = robot.get_observation()
    joint_positions = np.array([observation[j] for j in joint_order], dtype=np.float32)
    init_pos = joint_positions.copy()

    # --- Define Sinusoidal Trajectory Parameters ---
    zero_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Start pose
    amplitude = np.array([0.0, -60.0, 60.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Amplitude for each joint (degrees)
    offset = zero_pose.copy()
    period = 5.0  # seconds for one full sine wave cycle
    omega = 2 * np.pi / period  # angular frequency (rad/s)
    traj_duration = 20.0  # total duration to run the trajectory (seconds)
    steps = int(traj_duration * fps)

    # --- Move arm to zero position with linear trajectory ---
    zero_steps = int(2.0 * fps)  # 2 seconds to reach zero
    print("Moving to zero position (linear trajectory)...")
    observation = robot.get_observation()
    joint_positions = np.array([observation[j] for j in joint_order], dtype=np.float32)
    for step in range(1, zero_steps + 1):
        alpha = step / zero_steps
        pose = (1 - alpha) * joint_positions + alpha * zero_pose
        action = {j: float(pose[i]) for i, j in enumerate(joint_order)}
        robot.send_action(action)
        pg.time.wait(int(1000 / fps))
    print("Arm reached zero position.")

    # --- Sinusoidal Trajectory ---
    print("Starting sinusoidal trajectory...")
    start_time = time.perf_counter()
    quarter_period = period / 4
    epsilon = 1e-4
    for step in range(steps):
        t = (step / fps)
        pose = offset + amplitude * np.sin(omega * t)
        action = {j: float(pose[i]) for i, j in enumerate(joint_order)}
        robot.send_action(action)
        pg.time.wait(int(1000 / fps))

        # Pause for 1 second at every quarter period (excluding t=0)
        if step > 0:
            t_mod = t % quarter_period
            if t_mod < epsilon or abs(t_mod - quarter_period) < epsilon:
                print(f"Pausing at t={t:.2f}s (quarter period)...")
                pg.time.wait(1000)  # Wait 1 second

    print("Sinusoidal trajectory complete.")

    # --- Move back to initial position with linear trajectory ---
    return_steps = int(2.0 * fps)  # 2 seconds to return
    print("Returning to initial position (linear trajectory)...")
    observation = robot.get_observation()
    joint_positions = np.array([observation[j] for j in joint_order], dtype=np.float32)
    for step in range(1, return_steps + 1):
        alpha = step / return_steps
        pose = (1 - alpha) * joint_positions + alpha * init_pos
        action = {j: float(pose[i]) for i, j in enumerate(joint_order)}
        robot.send_action(action)
        pg.time.wait(int(1000 / fps))
    print("Arm returned to initial position.")
    

# The @draccus.wrap() decorator turns teleoperate(cfg: TeleoperateConfig) into a command-line interface.
# Under the hood, draccus.wrap():
# Parses sys.argv into a TeleoperateConfig.
@draccus.wrap()
def teleoperate(cfg: TeleoperateConfig):
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    try:
        move_to(robot, cfg.fps, display_data=cfg.display_data, duration=cfg.teleop_time_s)
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()


if __name__ == "__main__":
    teleoperate()
