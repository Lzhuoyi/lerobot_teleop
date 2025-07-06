"""
Simple script to control a robot from teleoperation.

Example:

```shell
python -m lerobot.teleop_keyboard_v2 --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=robot1 --robot.cameras='{}' --display_data=true
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


def teleop_loop(
    robot: Robot, fps: int, display_data: bool = False, duration: float | None = None
):
    pg.init()
    screen_width, screen_height = 600, 400
    screen = pg.display.set_mode((screen_width, screen_height))
    pg.display.set_caption("Control Window - Use keys to move")
    font = pg.font.SysFont(None, 24)

    # --- Kinematics ---
    arm_kin = ArmKinematics()
    joint_order = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]

    # --- Initial State ---
    observation = robot.get_observation()
    joint_positions = np.array([observation[j] for j in joint_order], dtype=np.float32)
    initial_joint_positions = joint_positions.copy()
    x, y = arm_kin.forward_kinematics(joint_positions[1], joint_positions[2])
    target_xy = np.array([x, y], dtype=np.float32)
    initial_xy = target_xy.copy()
    gripper_pos = joint_positions[5]

    zero_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Zero position for all joints

    # --- Control Params ---
    velocity = np.zeros(2, dtype=np.float32)  # vx, vy
    vel_step = 0.05  # m/s
    joint_vel_step = 151  # deg/s for direct joint control
    p_gain = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # P-gain for each joint

    # --- New: Joint velocity increments for direct control ---
    joint_velocities = np.zeros(3, dtype=np.float32)  # [shoulder_pan, wrist_flex, wrist_roll]

    clock = pg.time.Clock()
    start = time.perf_counter()

    # Move arm to zero position at the beginning
    zero_threshold = 2.0  # degrees, threshold for each joint to consider "at zero"
    max_init_time = 5.0   # seconds, max time to try moving to zero
    init_start = time.perf_counter()
    while True:
        init_action = {j: float(zero_pose[i]) for i, j in enumerate(joint_order)}
        robot.send_action(init_action)
        observation = robot.get_observation()
        joint_positions = np.array([observation[j] for j in joint_order], dtype=np.float32)
        if np.all(np.abs(joint_positions - zero_pose) < zero_threshold):
            print("Arm reached zero position.")
            break
        if time.perf_counter() - init_start > max_init_time:
            print("Warning: Arm did not reach zero position within time limit.")
            break
        pg.time.wait(50)

    while True:
        loop_start = time.perf_counter()
        dt = clock.tick(fps) / 1000.0

        # --- Get latest joint positions ---
        observation = robot.get_observation()
        joint_positions = np.array([observation[j] for j in joint_order], dtype=np.float32)

        # --- Handle events ---
        reset_flag = False
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_COMMA:
                    gripper_pos -= 30
                if event.key == pg.K_PERIOD:
                    gripper_pos += 30
                if event.key == pg.K_r:
                    reset_flag = True

        # --- Handle key holds for velocity control ---
        keys = pg.key.get_pressed()
        velocity[:] = 0.0
        joint_velocities[:] = 0.0
        if keys[pg.K_MINUS]:        # '-' decrease X
            velocity[0] -= vel_step
        if keys[pg.K_EQUALS]:       # '=' increase X
            velocity[0] += vel_step
        if keys[pg.K_LEFTBRACKET]:  # '[' decrease Y
            velocity[1] -= vel_step
        if keys[pg.K_RIGHTBRACKET]: # ']' increase Y
            velocity[1] += vel_step

        # --- New: Direct joint velocity control ---
        # shoulder_pan
        if keys[pg.K_q]:
            joint_velocities[0] += joint_vel_step
        if keys[pg.K_a]:
            joint_velocities[0] -= joint_vel_step
        # wrist_flex
        if keys[pg.K_w]:
            joint_velocities[1] += joint_vel_step
        if keys[pg.K_s]:
            joint_velocities[1] -= joint_vel_step
        # wrist_roll
        if keys[pg.K_e]:
            joint_velocities[2] += joint_vel_step
        if keys[pg.K_d]:
            joint_velocities[2] -= joint_vel_step

        # --- Reset logic ---
        if reset_flag:
            joint_targets = zero_pose.copy()
            target_xy = initial_xy.copy()
            gripper_pos = zero_pose[5]
            action = {j: float(joint_targets[i]) for i, j in enumerate(joint_order)}
            robot.send_action(action)
            print("Arm reset to initial position.")
            pg.time.wait(500)
            continue

        # --- Update target XY by velocity ---
        target_xy += velocity * dt

        # --- Use IK to get joint2 and joint3 (shoulder_lift, elbow_flex) ---
        joint2, joint3 = arm_kin.inverse_kinematics(target_xy[0], target_xy[1], current=(observation["shoulder_lift.pos"], observation["elbow_flex.pos"]))

        # --- Build target joint vector ---
        joint_targets = joint_positions.copy()
        joint_targets[1] = joint2
        joint_targets[2] = joint3
        joint_targets[5] = gripper_pos

        # --- Apply direct joint velocity control ---
        joint_targets[0] += joint_velocities[0] * dt  # shoulder_pan
        joint_targets[3] += joint_velocities[1] * dt  # wrist_flex
        joint_targets[4] += joint_velocities[2] * dt  # wrist_roll

        # --- Simple P-controller for smooth movement ---
        action = {}
        for i, j in enumerate(joint_order):
            action[j] = float(joint_positions[i] + p_gain[i] * (joint_targets[i] - joint_positions[i]))
            #float(joint_positions[i] + p_gain[i] * (joint_targets[i] - joint_positions[i]))

        # --- Send action to robot ---
        robot.send_action(action)

        # --- Print status ---
        x_fk, y_fk = arm_kin.forward_kinematics(action["shoulder_lift.pos"], action["elbow_flex.pos"])
        print(f"Target XY: {target_xy}, FK(XY): ({x_fk:.3f}, {y_fk:.3f})")
        print("Observation:", pformat(observation))
        print("Action:", pformat(action))
        print("\n")

        # --- Draw simple status ---
        screen.fill((0, 0, 0))
        info = [
            f"Target XY: {target_xy}",
            f"FK(XY): ({x_fk:.3f}, {y_fk:.3f})",
            f"Joint2: {action['shoulder_lift.pos']:.2f} deg",
            f"Joint3: {action['elbow_flex.pos']:.2f} deg",
            f"Gripper: {gripper_pos:.2f}",
            "Keys: -/=: X, [/] Y, q/a: shoulder_pan, w/s: wrist_flex, e/d: wrist_roll, ,/. gripper, r reset"
        ]
        for i, txt in enumerate(info):
            screen.blit(font.render(txt, True, (255, 255, 255)), (10, 10 + i * 25))
        pg.display.flip()

        # --- Timing and exit ---
        dt_s = time.perf_counter() - loop_start
        if duration is not None and time.perf_counter() - start >= duration:
            return
        

# The @draccus.wrap() decorator turns teleoperate(cfg: TeleoperateConfig) into a command-line interface.
# Under the hood, draccus.wrap():
# Parses sys.argv into a TeleoperateConfig.
@draccus.wrap()
def teleoperate(cfg: TeleoperateConfig):
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    try:
        teleop_loop(robot, cfg.fps, display_data=cfg.display_data, duration=cfg.teleop_time_s)
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()


if __name__ == "__main__":
    teleoperate()
