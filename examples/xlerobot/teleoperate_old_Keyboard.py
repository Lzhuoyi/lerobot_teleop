# To Run on the host
'''python
python -m lerobot.robots.xlerobot.xlerobot_host --robot.id=my_xlerobot
'''

# To Run the teleop:
'''python
PYTHONPATH=src python -m examples.xlerobot.teleoperate_old_Keyboard
'''

import time
import numpy as np
import pygame as pg
import sys

from lerobot.robots.xlerobot import XLerobotClient, XLerobotClientConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data
from lerobot.model.SO101Robot import SO101Kinematics, create_real_robot
from lerobot.model.kinematics import RobotKinematics

# Customized parameters
FPS = 30
ip = "192.168.1.123"
robot_name = "my_xlerobot_pc"
# robot_urdf_path = "examples/xlerobot/so101_new_calib.urdf"

# Keymaps for each arm (customize as needed)
LEFT_KEYMAP = {
    'x-': pg.K_j, 'x+': pg.K_l,
    'y-': pg.K_i, 'y+': pg.K_k,
    'z-': pg.K_u, 'z+': pg.K_o,
    'wrist_flex+': pg.K_y, 'wrist_flex-': pg.K_h,
    'wrist_roll+': pg.K_n, 'wrist_roll-': pg.K_m,
    'gripper-': pg.K_COMMA, 'gripper+': pg.K_PERIOD,
    'reset': pg.K_SEMICOLON
}
RIGHT_KEYMAP = {
    'x-': pg.K_KP4, 'x+': pg.K_KP6,
    'y-': pg.K_KP8, 'y+': pg.K_KP5,
    'z-': pg.K_KP7, 'z+': pg.K_KP9,
    'wrist_flex+': pg.K_KP1, 'wrist_flex-': pg.K_KP3,
    'wrist_roll+': pg.K_SLASH, 'wrist_roll-': pg.K_ASTERISK,
    'gripper-': pg.K_MINUS, 'gripper+': pg.K_PLUS,
    'reset': pg.K_KP0
}

LEFT_JOINTS = [
    "left_arm_shoulder_pan.pos",
    "left_arm_shoulder_lift.pos",
    "left_arm_elbow_flex.pos",
    "left_arm_wrist_flex.pos",
    "left_arm_wrist_roll.pos",
    "left_arm_gripper.pos",
]
RIGHT_JOINTS = [
    "right_arm_shoulder_pan.pos",
    "right_arm_shoulder_lift.pos",
    "right_arm_elbow_flex.pos",
    "right_arm_wrist_flex.pos",
    "right_arm_wrist_roll.pos",
    "right_arm_gripper.pos",
]

class TeleopArm:
    def __init__(self, joint_names, kinematics, keymap, obs, name="arm"):
        self.joint_names = joint_names
        self.kinematics = kinematics
        self.keymap = keymap
        self.name = name
        self.vel_step = 0.0811
        self.joint_vel_step = 151
        self.gripper_pos = obs[joint_names[-1]] if joint_names[-1] in obs else 0.0
        self.reset(obs)

    def reset(self, obs):
        self.joint_positions = np.array([obs.get(j, 0.0) for j in self.joint_names], dtype=np.float32)
        ee_pose = self.kinematics.forward_kinematics(self.joint_positions, frame="gripper_tip")
        self.target_ee_pos = ee_pose[:3, 3].copy()

    def handle_keys(self, keys, dt, obs):
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
        self.joint_positions = np.array([obs.get(j, 0.0) for j in self.joint_names], dtype=np.float32)
        self.joint_positions[3] += wrist_flex_vel * dt
        self.joint_positions[4] += wrist_roll_vel * dt

        # IK for main arm
        ee_pose = self.kinematics.forward_kinematics(self.joint_positions, frame="gripper_tip")
        target_pose = ee_pose.copy()
        target_pose[:3, 3] = self.target_ee_pos
        goal_joint_positions = self.kinematics.inverse_kinematics(self.joint_positions, target_pose, position_only=True, frame="gripper_tip")

        action = {j: float(goal_joint_positions[i]) for i, j in enumerate(self.joint_names)}
        action[self.joint_names[3]] = float(self.joint_positions[3])
        action[self.joint_names[4]] = float(self.joint_positions[4])
        action[self.joint_names[5]] = self.gripper_pos
        return action

    def handle_events(self, events):
        for event in events:
            if event.type == pg.KEYDOWN:
                if event.key == self.keymap['gripper-']:
                    self.gripper_pos -= 401
                if event.key == self.keymap['gripper+']:
                    self.gripper_pos += 401
                if event.key == self.keymap['reset']:
                    # Reset requires latest obs, will be handled in main loop
                    pass

def main():
    robot_config = XLerobotClientConfig(remote_ip=ip, id=robot_name)
    robot = XLerobotClient(robot_config)
    robot.connect()
    _init_rerun(session_name="xlerobot_teleop")

    pg.init()
    screen = pg.display.set_mode((600, 400))
    pg.display.set_caption("XLerobot Dual Arm Teleoperation")
    clock = pg.time.Clock()

    # Get initial observation for joint positions
    obs = robot.get_observation()
    kin_left = RobotKinematics(robot_type="so_new_calibration")
    kin_right = RobotKinematics(robot_type="so_new_calibration")
    left_arm = TeleopArm(LEFT_JOINTS, kin_left, LEFT_KEYMAP, obs, name="left_arm")
    right_arm = TeleopArm(RIGHT_JOINTS, kin_right, RIGHT_KEYMAP, obs, name="right_arm")

    while True:
        dt = clock.tick(FPS) / 1000.0
        keys = pg.key.get_pressed()
        events = pg.event.get()
        for event in events:
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            left_arm.handle_events([event])
            right_arm.handle_events([event])
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                pg.quit()
                sys.exit()

        obs = robot.get_observation()

        # Reset if requested
        for event in events:
            if event.type == pg.KEYDOWN:
                if event.key == LEFT_KEYMAP['reset']:
                    left_arm.reset(obs)
                if event.key == RIGHT_KEYMAP['reset']:
                    right_arm.reset(obs)

        left_action = left_arm.handle_keys(keys, dt, obs)
        right_action = right_arm.handle_keys(keys, dt, obs)

        # Base action (use your existing teleop_keys mapping)
        pressed = []
        for k, v in robot.teleop_keys.items():
            if isinstance(v, str):
                try:
                    if keys[getattr(pg, f"K_{v}")]:
                        pressed.append(v)
                except AttributeError:
                    continue
        base_action = robot._from_keyboard_to_base_action(np.array(pressed))

        # Merge actions (The ** operator is used to unpack dictionaries)
        action = {**left_action, **right_action, **base_action}
        log_rerun_data(obs, action)
        robot.send_action(action)
        busy_wait(max(0, 1.0 / FPS - dt))

if __name__ == "__main__":
    main()
