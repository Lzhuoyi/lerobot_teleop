from lerobot.common.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.common.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.common.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.common.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

from lerobot.common.robots.lekiwi.config_lekiwi import lekiwi_cameras_config
from lerobot.common.cameras.opencv.camera_opencv import OpenCVCamera

robot_config = LeKiwiClientConfig(remote_ip="192.168.1.123", id="lekiwi_right_arm")

import rerun as rr
import time
rr.init("lekiwi.opencv_stream", spawn=True)


teleop__arm_config = SO101LeaderConfig(
    port="/dev/ttyACM0",
    id="lekiwi_left_arm",
)

teleop_keyboard_config = KeyboardTeleopConfig(
    id="my_laptop_keyboard",
)

robot = LeKiwiClient(robot_config)
teleop_arm = SO101Leader(teleop__arm_config)
telep_keyboard = KeyboardTeleop(teleop_keyboard_config)
robot.connect()
teleop_arm.connect()
telep_keyboard.connect()

try:
    while True:
        observation = robot.get_observation()

        arm_action = teleop_arm.get_action()
        arm_action = {f"arm_{k}": v for k, v in arm_action.items()}

        keyboard_keys = telep_keyboard.get_action()
        base_action = robot._from_keyboard_to_base_action(keyboard_keys)

        robot.send_action(arm_action | base_action)

        # Display camera data streams
        for cam_key in observation:
            if cam_key.startswith("observation.images."):
                cam_name = cam_key.split(".")[-1]
                frame = observation[cam_key]
                # If frame is a torch tensor, convert to numpy
                if hasattr(frame, "numpy"):
                    frame = frame.numpy()
                rr.log(f"camera/{cam_name}", rr.Image(frame))
        time.sleep(1 / 30)
except KeyboardInterrupt:
    print("Streaming stopped by user.")
finally:
    robot.disconnect()