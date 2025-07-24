"""
Simple script to test camera.

Example:

```shell
python -m lerobot.test_camera
```
"""
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.common.cameras.configs import ColorMode, Cv2Rotation

import cv2
import rerun as rr
rr.init("lerobot.realsense_stream", spawn=True)  # spawn=True opens the viewer automatically

from lerobot.common.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.common.cameras import ColorMode, Cv2Rotation

## This is for RealSense Camera
# Basic: RGBD
# config = RealSenseCameraConfig(serial_number_or_name="125322060037") # Replace with actual SN
# camera = RealSenseCamera(config)
# camera.connect()

# Example with depth capture and custom settings
custom_config = RealSenseCameraConfig(
    serial_number_or_name="125322060037",  # Replace with camera SN
    fps=30,
    width=1280,
    height=720,
    color_mode=ColorMode.BGR, # Request BGR output
    rotation=Cv2Rotation.NO_ROTATION,
    use_depth=True
)
depth_camera = RealSenseCamera(custom_config)
depth_camera.connect()

## This is for OpenCV Camera
config = OpenCVCameraConfig(
    index_or_path=8,  # Replace with camera index found in find_cameras.py
    fps=30,
    width=1280,
    height=720,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION
)
# Instantiate and connect an `OpenCVCamera`, performing a warm-up read (default).
# camera = OpenCVCamera(config)
# camera.connect()


try:
    while True:
        ## This is for the RealSense Camera
        color_frame = depth_camera.read()
        depth_map = depth_camera.read_depth()
        # Stream color image (BGR)
        rr.log("camera/color", rr.Image(color_frame))
        # Stream depth map (normalize for visualization, but also log raw)
        rr.log("camera/depth_raw", rr.Image(depth_map))

        ## This is for the OpenCV Camera
        # frame = camera.async_read(timeout_ms=200)
        # rr.log("camera/opencv", rr.Image(frame))

except KeyboardInterrupt:
    print("Streaming stopped by user.")
finally:
    depth_camera.disconnect()
    # camera.disconnect()



