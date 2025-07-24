"""
Simple script to control a robot from teleoperation.

Example:

```shell
python -m lerobot.teleop_keyboard_v3 --robot1.type=so101_follower --robot1.port=/dev/ttyACM0 --robot1.id=robot1 --robot2.type=so101_follower --robot2.port=/dev/ttyACM1 --robot2.id=robot2 --display_data=true
```
"""

import logging
import time
import numpy as np
import pygame as pg
import sys
from dataclasses import dataclass
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,  
    koch_follower,
    make_robot_from_config,    # From this import, we can find PID parameters
    so100_follower,
    so101_follower,
)
from lerobot.model.SO101Robot import SO101Kinematics, create_real_robot
from lerobot.utils.robot_utils import busy_wait
import draccus

@dataclass
class DualTeleoperateConfig:
    robot1: RobotConfig
    robot2: RobotConfig
    fps: int = 60
    teleop_time_s: float | None = None
    display_data: bool = False

class TeleopArm:
    def __init__(self, robot, keymap, name="arm"):
        self.robot = robot  # Already a robot instance!
        self.kinematics = RobotKinematics(robot_type="so_new_calibration")
        self.joint_order = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
        self.keymap = keymap
        self.name = name
        self.gripper_pos = 0.0
        self.vel_step = 0.0811
        self.joint_vel_step = 151
        # Store initial joint positions and EE position
        obs = self.robot.get_observation()
        self.initial_joint_positions = np.array([obs[j] for j in self.joint_order], dtype=np.float32)
        ee_pose = self.kinematics.forward_kinematics(self.initial_joint_positions, frame="gripper_tip")
        self.initial_ee_pos = ee_pose[:3, 3].copy()
        self.reset()

    def reset(self):
        # Reset to initial joint positions and EE position
        self.joint_positions = self.initial_joint_positions.copy()
        self.target_ee_pos = self.initial_ee_pos.copy()

    def handle_keys(self, keys, dt):
        velocity = np.zeros(3, dtype=np.float32)
        wrist_flex_vel = 0.0
        wrist_roll_vel = 0.0

        # EE velocity control
        if keys[self.keymap['x-']]: velocity[0] -= self.vel_step
        if keys[self.keymap['x+']]: velocity[0] += self.vel_step
        if keys[self.keymap['y-']]: velocity[1] -= self.vel_step
        if keys[self.keymap['y+']]: velocity[1] += self.vel_step
        if keys[self.keymap['z-']]: velocity[2] -= self.vel_step
        if keys[self.keymap['z+']]: velocity[2] += self.vel_step

        # Joint velocity control
        if keys[self.keymap['wrist_flex+']]: wrist_flex_vel += self.joint_vel_step
        if keys[self.keymap['wrist_flex-']]: wrist_flex_vel -= self.joint_vel_step
        if keys[self.keymap['wrist_roll+']]: wrist_roll_vel += self.joint_vel_step
        if keys[self.keymap['wrist_roll-']]: wrist_roll_vel -= self.joint_vel_step

        self.target_ee_pos += velocity * dt
        obs = self.robot.get_observation()
        self.joint_positions = np.array([obs[j] for j in self.joint_order], dtype=np.float32)
        self.joint_positions[3] += wrist_flex_vel * dt
        self.joint_positions[4] += wrist_roll_vel * dt

        # IK for main arm
        ee_pose = self.kinematics.forward_kinematics(self.joint_positions, frame="gripper_tip")
        target_pose = ee_pose.copy()
        target_pose[:3, 3] = self.target_ee_pos
        goal_joint_positions = self.kinematics.ik(self.joint_positions, target_pose, position_only=True, frame="gripper_tip")

        action = {j: float(goal_joint_positions[i]) for i, j in enumerate(self.joint_order)}
        action["wrist_flex.pos"] = float(self.joint_positions[3])
        action["wrist_roll.pos"] = float(self.joint_positions[4])
        action["gripper.pos"] = self.gripper_pos
        return action

    def handle_events(self, events):
        for event in events:
            if event.type == pg.KEYDOWN:
                if event.key == self.keymap['gripper-']:
                    self.gripper_pos -= 401
                if event.key == self.keymap['gripper+']:
                    self.gripper_pos += 401
                if event.key == self.keymap['reset']:
                    self.reset()

    def send_action(self, action):
        self.robot.send_action(action)

def teleop_loop_dual(robot1, robot2, fps=60, display_data=False, duration=None):
    pg.init()
    screen_width, screen_height = 600, 400
    screen = pg.display.set_mode((screen_width, screen_height))
    pg.display.set_caption("Dual Arm Teleoperation")
    clock = pg.time.Clock()

    # Keymaps for each arm
    keymap1 = {
        'x-': pg.K_MINUS, 'x+': pg.K_EQUALS,
        'y-': pg.K_LEFTBRACKET, 'y+': pg.K_RIGHTBRACKET,
        'z-': pg.K_SEMICOLON, 'z+': pg.K_QUOTE,
        'wrist_flex+': pg.K_q, 'wrist_flex-': pg.K_a,
        'wrist_roll+': pg.K_w, 'wrist_roll-': pg.K_s,
        'gripper-': pg.K_COMMA, 'gripper+': pg.K_PERIOD,
        'reset': pg.K_r
    }
    keymap2 = {
        'x-': pg.K_KP4, 'x+': pg.K_KP6,
        'y-': pg.K_KP2, 'y+': pg.K_KP8,
        'z-': pg.K_KP1, 'z+': pg.K_KP3,
        'wrist_flex+': pg.K_z, 'wrist_flex-': pg.K_x,
        'wrist_roll+': pg.K_c, 'wrist_roll-': pg.K_v,
        'gripper-': pg.K_b, 'gripper+': pg.K_n,
        'reset': pg.K_m
    }

    arm1 = TeleopArm(robot1, keymap1, name="arm1")
    arm2 = TeleopArm(robot2, keymap2, name="arm2")

    while True:
        dt = clock.tick(fps) / 1000.0
        keys = pg.key.get_pressed()
        events = pg.event.get()
        for event in events:
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
        arm1.handle_events(events)
        arm2.handle_events(events)
        action1 = arm1.handle_keys(keys, dt)
        action2 = arm2.handle_keys(keys, dt)
        arm1.send_action(action1)
        arm2.send_action(action2)
        busy_wait(max(0, 1 / fps - dt))

@draccus.wrap()
def dual_teleoperate(cfg: DualTeleoperateConfig):
    from lerobot.common.utils.utils import init_logging
    init_logging()
    logging.info(f"Robot1: {cfg.robot1}")
    logging.info(f"Robot2: {cfg.robot2}")
    robot1 = make_robot_from_config(cfg.robot1)
    robot2 = make_robot_from_config(cfg.robot2)
    robot1.connect()
    robot2.connect()
    try:
        teleop_loop_dual(robot1, robot2, fps=cfg.fps, display_data=cfg.display_data, duration=cfg.teleop_time_s)
    except KeyboardInterrupt:
        pass
    finally:
        robot1.disconnect()
        robot2.disconnect()

# Example usage:
if __name__ == "__main__":
    dual_teleoperate()
