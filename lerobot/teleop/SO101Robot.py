import math
import numpy as np

from lerobot.common.robots.robot import Robot
from lerobot.common.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.common.robots.utils import make_robot_from_config
import numpy as np
from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.common.cameras import ColorMode, Cv2Rotation
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig

def create_real_robot(port, camera_index, uid: str = "so101") -> Robot:
    """Wrapper function to map string UIDS to real robot configurations. Primarily for saving a bit of code for users when they fork the repository. They can just edit the camera, id etc. settings in this one file."""
    if uid == "so101":
        robot_config = SO101FollowerConfig(
            port= port,
            use_degrees=True,
            # for phone camera users you can use the commented out setting below
            cameras = {
                "base_camera": OpenCVCameraConfig(index_or_path= camera_index,  # Replace with camera index found in find_cameras.py
                fps=30,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
                rotation=Cv2Rotation.NO_ROTATION)
            },
            # for intel realsense camera users you need to modify the serial number or name for your own hardware
            # cameras={
            #     "base_camera": RealSenseCameraConfig(serial_number_or_name="146322070293", fps=30, width=640, height=480)
            # },
            id="robot1",
        )
        real_robot = make_robot_from_config(robot_config)
        return real_robot


class SO101Kinematics:
    """
    A class to represent the kinematics of a SO101 robot arm.
    All public methods use degrees for input/output.
    """

    def __init__(self, l1=0.1159, l2=0.1350):
        self.l1 = l1  # Length of the first link (upper arm)
        self.l2 = l2  # Length of the second link (lower arm)

    def inverse_kinematics(self, x, y, l1=None, l2=None):
        """
        Calculate inverse kinematics for a 2-link robotic arm, considering joint offsets
        
        Parameters:
            x: End effector x coordinate
            y: End effector y coordinate
            l1: Upper arm length (default uses instance value)
            l2: Lower arm length (default uses instance value)
            
        Returns:
            joint2_deg, joint3_deg: Joint angles in degrees (shoulder_lift, elbow_flex)
        """
        # Use instance values if not provided
        if l1 is None:
            l1 = self.l1
        if l2 is None:
            l2 = self.l2
            
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
        
        # Clamp cos_theta2 to valid range [-1, 1] to avoid domain errors
        cos_theta2 = max(-1.0, min(1.0, cos_theta2))
        
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
        
        # Convert from radians to degrees
        joint2_deg = math.degrees(joint2)
        joint3_deg = math.degrees(joint3)

        # Apply coordinate system transformation
        joint2_deg = 90 - joint2_deg
        joint3_deg = joint3_deg - 90
        
        return joint2_deg, joint3_deg
    
    def forward_kinematics(self, joint2_deg, joint3_deg, l1=None, l2=None):
        """
        Calculate forward kinematics for a 2-link robotic arm
        
        Parameters:
            joint2_deg: Shoulder lift joint angle in degrees
            joint3_deg: Elbow flex joint angle in degrees
            l1: Upper arm length (default uses instance value)
            l2: Lower arm length (default uses instance value)
            
        Returns:
            x, y: End effector coordinates
        """
        # Use instance values if not provided
        if l1 is None:
            l1 = self.l1
        if l2 is None:
            l2 = self.l2
            
        # Convert degrees to radians and apply inverse transformation
        joint2_rad = math.radians(90 - joint2_deg)
        joint3_rad = math.radians(joint3_deg + 90)
        
        # Calculate joint2 and joint3 offsets
        theta1_offset = math.atan2(0.028, 0.11257)
        theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset
        
        # Convert joint angles back to theta1 and theta2
        theta1 = joint2_rad - theta1_offset
        theta2 = joint3_rad - theta2_offset
        
        # Forward kinematics calculations
        x = l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2 - math.pi)
        y = l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2 - math.pi)
        
        return x, y
    
    def is_within_workspace(self, x, y, l1=None, l2=None):
        """
        Check if a point (x, y) is within the robot's workspace
        
        Parameters:
            x: Target x coordinate
            y: Target y coordinate
            l1: Upper arm length (default uses instance value)
            l2: Lower arm length (default uses instance value)
            
        Returns:
            bool: True if point is within workspace, False otherwise
        """
        # Use instance values if not provided
        if l1 is None:
            l1 = self.l1
        if l2 is None:
            l2 = self.l2
            
        r = math.sqrt(x**2 + y**2)
        r_max = l1 + l2
        r_min = abs(l1 - l2)
        
        return r_min <= r <= r_max
    
    def get_workspace_boundary(self, num_points=100, l1=None, l2=None):
        """
        Generate points on the workspace boundary
        
        Parameters:
            num_points: Number of points to generate
            l1: Upper arm length (default uses instance value)
            l2: Lower arm length (default uses instance value)
            
        Returns:
            outer_boundary: List of (x, y) points on outer boundary
            inner_boundary: List of (x, y) points on inner boundary (if exists)
        """
        # Use instance values if not provided
        if l1 is None:
            l1 = self.l1
        if l2 is None:
            l2 = self.l2
            
        r_max = l1 + l2
        r_min = abs(l1 - l2)
        
        outer_boundary = []
        inner_boundary = []
        
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            
            # Outer boundary
            x_outer = r_max * math.cos(angle)
            y_outer = r_max * math.sin(angle)
            outer_boundary.append((x_outer, y_outer))
            
            # Inner boundary (only if r_min > 0)
            if r_min > 0:
                x_inner = r_min * math.cos(angle)
                y_inner = r_min * math.sin(angle)
                inner_boundary.append((x_inner, y_inner))
        
        return outer_boundary, inner_boundary