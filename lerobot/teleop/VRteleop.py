#!/usr/bin/env python3
"""
VR Teleoperation for SO101 Robot
python -m lerobot.teleop.VRteleop
"""

import time
import logging
import traceback
import math
import threading
import asyncio
import numpy as np
import sys
import os
from lerobot.teleop.SO101Robot import SO101Kinematics, create_real_robot
from lerobot.common.utils.visualization_utils import _init_rerun, log_rerun_data

# Add VR monitor path
sys.path.append(os.path.abspath("/home/jellyfish/lerobot_ws/XLeRobot/XLeVR"))
from vr_monitor import VRMonitor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Joint calibration coefficients - manually edit
# Format: [joint_name, zero_position_offset(degrees), scale_factor]
JOINT_CALIBRATION = [
    ['shoulder_pan', 6.0, 1.0],      # Joint1: zero position offset, scale factor
    ['shoulder_lift', 2.0, 0.97],   # Joint2: zero position offset, scale factor
    ['elbow_flex', 0.0, 1.05],      # Joint3: zero position offset, scale factor
    ['wrist_flex', 0.0, 0.94],      # Joint4: zero position offset, scale factor
    ['wrist_roll', 0.0, 0.5],       # Joint5: zero position offset, scale factor
    ['gripper', 0.0, 1.0],          # Joint6: zero position offset, scale factor
]

class SO101VRTeleopController:
    """SO101 Robot VR Teleoperation Controller"""
    
    def __init__(self, robot, kinematics=None):
        self.robot = robot
        self.kinematics = kinematics if kinematics else SO101Kinematics()
        
        # Control parameters
        self.kp = 0.5  # Proportional gain
        self.control_freq = 50  # Control frequency (Hz)
        
        # End effector initial position
        self.current_x = 0.1629
        self.current_y = 0.1131
        
        # Pitch control parameters
        self.pitch = 0.0
        self.pitch_step = 1.0
        
        # Record start positions
        self.start_positions = {}
        
        # Target positions
        self.target_positions = {
            'shoulder_pan': 0.0,
            'shoulder_lift': 0.0,
            'elbow_flex': 0.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 0.0
        }
        
        # VR control parameters
        self.vr_scale_x = 0.5
        self.vr_scale_y = 0.6
        self.vr_scale_z = -0.6
        self.vr_pitch_scale = 1.3
        self.vr_roll_scale = 1.3
        self.vr_rotation_scale = 180/ math.pi  # Convert radians to degrees
        
        # VR activity tracking
        self.last_vr_activity_time = time.time()
        self.auto_reset_timeout = 30.0  # Reset after 30 seconds of no VR activity
        
        # Initial values for reset
        self.initial_x = self.current_x
        self.initial_y = self.current_y
        self.initial_pitch = 0.0
        
    def apply_joint_calibration(self, joint_name, raw_position):
        """
        Apply joint calibration coefficients
        
        Args:
            joint_name: Joint name
            raw_position: Raw position value
        
        Returns:
            calibrated_position: Calibrated position value
        """
        for joint_cal in JOINT_CALIBRATION:
            if joint_cal[0] == joint_name:
                offset = joint_cal[1]  # Zero position offset
                scale = joint_cal[2]   # Scale factor
                calibrated_position = (raw_position - offset) * scale
                return calibrated_position
        return raw_position  # If no calibration coefficient found, return raw value

    def move_to_zero_position(self, duration=3.0):
        """
        Use P control to slowly move robot to zero position
        
        Args:
            duration: Time to move to zero position (seconds)
        """
        print("Moving robot to zero position using P control...")
        
        # Get current robot state
        current_obs = self.robot.get_observation()
        
        # Extract current joint positions
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                current_positions[motor_name] = value
        
        # Zero position target
        zero_positions = {
            'shoulder_pan': 0.0,
            'shoulder_lift': 0.0,
            'elbow_flex': 0.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 0.0
        }
        
        # Calculate control steps
        total_steps = int(duration * self.control_freq)
        step_time = 1.0 / self.control_freq
        
        print(f"Moving to zero position in {duration} seconds using P control, frequency: {self.control_freq}Hz, gain: {self.kp}")
        
        for step in range(total_steps):
            # Get current robot state
            current_obs = self.robot.get_observation()
            current_positions = {}
            for key, value in current_obs.items():
                if key.endswith('.pos'):
                    motor_name = key.removesuffix('.pos')
                    # Apply calibration coefficients
                    calibrated_value = self.apply_joint_calibration(motor_name, value)
                    current_positions[motor_name] = calibrated_value
            
            # P control calculation
            robot_action = {}
            for joint_name, target_pos in zero_positions.items():
                if joint_name in current_positions:
                    current_pos = current_positions[joint_name]
                    error = target_pos - current_pos
                    
                    # P control: output = Kp * error
                    control_output = self.kp * error
                    
                    # Convert control output to position command
                    new_position = current_pos + control_output
                    robot_action[f"{joint_name}.pos"] = new_position
            
            # Send action to robot
            if robot_action:
                self.robot.send_action(robot_action)
            
            # Show progress
            if step % (self.control_freq // 2) == 0:  # Show progress every 0.5 seconds
                progress = (step / total_steps) * 100
                print(f"Moving to zero position progress: {progress:.1f}%")
            
            time.sleep(step_time)
        
        print("Robot moved to zero position")

    def return_to_start_position(self):
        """
        Use P control to return to start position
        """
        print("Returning to start position...")
        
        control_period = 1.0 / self.control_freq
        max_steps = int(5.0 * self.control_freq)  # Maximum 5 seconds
        
        for step in range(max_steps):
            # Get current robot state
            current_obs = self.robot.get_observation()
            current_positions = {}
            for key, value in current_obs.items():
                if key.endswith('.pos'):
                    motor_name = key.removesuffix('.pos')
                    current_positions[motor_name] = value  # Don't apply calibration coefficients
            
            # P control calculation
            robot_action = {}
            total_error = 0
            for joint_name, target_pos in self.start_positions.items():
                if joint_name in current_positions:
                    current_pos = current_positions[joint_name]
                    error = target_pos - current_pos
                    total_error += abs(error)
                    
                    # P control: output = Kp * error
                    control_output = self.kp * error
                    
                    # Convert control output to position command
                    new_position = current_pos + control_output
                    robot_action[f"{joint_name}.pos"] = new_position
            
            # Send action to robot
            if robot_action:
                self.robot.send_action(robot_action)
            
            # Check if start position reached
            if total_error < 2.0:  # If total error less than 2 degrees, consider reached
                print("Returned to start position")
                break
            
            time.sleep(control_period)
        
        print("Return to start position complete")

    def reset_positions(self):
        """Reset all positions to initial values"""
        self.current_x = self.initial_x
        self.current_y = self.initial_y
        self.pitch = self.initial_pitch
        
        # Reset target positions
        self.target_positions = {
            'shoulder_pan': 0.0,
            'shoulder_lift': 0.0,
            'elbow_flex': 0.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 0.0
        }
        
        print("All positions reset to initial values")

    def handle_vr_input(self, vr_data):
        """
        Handle VR controller input
        
        Args:
            vr_data: VR controller data from VRMonitor
            
        Returns:
            bool: True if VR activity detected, False otherwise
        """
        if not vr_data:
            return False
            
        vr_activity_detected = False
        
        # Get controller data
        left_goal = vr_data.get("left")
        right_goal = vr_data.get("right")
        
        # Process RIGHT controller (primary arm control)
        if right_goal is not None and right_goal.target_position is not None:
            vr_activity_detected = True
            pos = right_goal.target_position
            
            # Convert VR position to robot coordinates
            x_vr = (pos[0] - 0.1) * self.vr_scale_x
            y_vr = (pos[1] - 0.96) * self.vr_scale_y
            z_vr = (pos[2] + 0.4) * self.vr_scale_z

            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"VR Control - Raw Position: {pos}, Scaled Position: ({x_vr:.4f}, {y_vr:.4f}, {z_vr:.4f})")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            
            # Calculate horizontal distance in vertical plane
            r_distance = math.sqrt(x_vr**2 + z_vr**2)
            
            # Update end effector position
            self.current_x = r_distance
            self.current_y = y_vr
            
            # Calculate rotation angle based on controller direction
            if abs(x_vr) > 0.01 or abs(z_vr) > 0.01:  # Small dead zone
                rotation_angle = math.atan2(x_vr, z_vr)
                self.target_positions['shoulder_pan'] = rotation_angle * self.vr_rotation_scale
            
            # Handle wrist flex (pitch control)
            if right_goal.wrist_flex_deg is not None:
                self.pitch = (right_goal.wrist_flex_deg + 60) * self.vr_pitch_scale
            
            # Handle wrist roll
            if right_goal.wrist_roll_deg is not None:
                self.target_positions['wrist_roll'] = (right_goal.wrist_roll_deg) * self.vr_roll_scale
            
            # Handle gripper trigger
            if right_goal.metadata.get('trigger', 0) > 0.5:
                self.target_positions['gripper'] = 0.0  # Open
            else:
                self.target_positions['gripper'] = 90.0 # Close
        
        # Calculate inverse kinematics for end effector position
        if vr_activity_detected:
            try:
                joint2_target, joint3_target = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
                self.target_positions['shoulder_lift'] = joint2_target
                self.target_positions['elbow_flex'] = joint3_target
                
                print(f"VR Control - Position: ({self.current_x:.4f}, {self.current_y:.4f}), "
                      f"Joint2={joint2_target:.3f}, Joint3={joint3_target:.3f}")
            except Exception as e:
                print(f"Error calculating inverse kinematics: {e}")
        
        return vr_activity_detected

    def update_wrist_flex_with_pitch(self):
        """Calculate wrist_flex target position based on shoulder_lift and elbow_flex, plus pitch adjustment"""
        if 'shoulder_lift' in self.target_positions and 'elbow_flex' in self.target_positions:
            self.target_positions['wrist_flex'] = (
                -self.target_positions['shoulder_lift'] 
                - self.target_positions['elbow_flex'] 
                + self.pitch
            )

    def execute_p_control_step(self):
        """Execute one step of P control"""
        # Get current robot state
        current_obs = self.robot.get_observation()
        
        # Extract current joint positions
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                # Apply calibration coefficients
                calibrated_value = self.apply_joint_calibration(motor_name, value)
                current_positions[motor_name] = calibrated_value
        
        # P control calculation
        robot_action = {}
        for joint_name, target_pos in self.target_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                error = target_pos - current_pos
                
                # P control: output = Kp * error
                control_output = self.kp * error
                
                # Convert control output to position command
                new_position = current_pos + control_output
                robot_action[f"{joint_name}.pos"] = new_position
        
        # Send action to robot
        if robot_action:
            self.robot.send_action(robot_action)

        # Log data to Rerun
        log_rerun_data(current_obs, robot_action)

    def control_loop(self, vr_monitor):
        """
        Main control loop
        
        Args:
            vr_monitor: VR monitor instance
        """
        control_period = 1.0 / self.control_freq
        step_counter = 0
        warmup_steps = 50
        
        print(f"Starting VR control loop, frequency: {self.control_freq}Hz, gain: {self.kp}")
        print("VR Control Instructions:")
        print("- Use RIGHT controller to control end effector position")
        print("- Trigger to control gripper (squeeze to close)")
        print("- Wrist rotation controls joint orientation")
        print("- Auto-reset after 30 seconds of inactivity")
        print("="*50)
        
        while True:
            try:
                # Get VR controller data
                vr_data = vr_monitor.get_latest_goal_nowait()
                
                # Skip control during warmup phase
                if step_counter < warmup_steps:
                    step_counter += 1
                    time.sleep(control_period)
                    continue
                
                # Handle VR input
                vr_activity = self.handle_vr_input(vr_data)
                
                # Update activity time if VR active
                if vr_activity:
                    self.last_vr_activity_time = time.time()
                
                # Check for auto-reset
                current_time = time.time()
                if current_time - self.last_vr_activity_time > self.auto_reset_timeout:
                    print("No VR activity detected, resetting positions...")
                    self.reset_positions()
                    self.last_vr_activity_time = current_time
                
                # Update wrist_flex position
                self.update_wrist_flex_with_pitch()
                
                # Execute P control
                self.execute_p_control_step()
                
                step_counter += 1
                time.sleep(control_period)
                
            except KeyboardInterrupt:
                print("User interrupted program")
                break
            except Exception as e:
                print(f"VR control loop error: {e}")
                traceback.print_exc()
                break

    def initialize(self):
        """Initialize controller"""
        # Read start joint angles
        print("Reading start joint angles...")
        start_obs = self.robot.get_observation()
        for key, value in start_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                self.start_positions[motor_name] = int(value)  # Don't apply calibration coefficients
        
        print("Start joint angles:")
        for joint_name, position in self.start_positions.items():
            print(f"  {joint_name}: {position}°")


def main():
    """Main function"""
    print("LeRobot SO101 VR Control (P Control)")
    print("="*50)
    
    # Initialize Rerun
    _init_rerun(session_name="SO101VRControl")
    
    try:
        # Get port
        port = input("Please enter SO101 robot USB port (e.g., /dev/ttyACM0): ").strip()
        # If directly press Enter, use default port
        if not port:
            port = "/dev/ttyACM0"
            print(f"Using default port: {port}")
        else:
            print(f"Connecting to port: {port}")

        # Get camera port
        camera_index = input("Please enter SO101 robot CAMERA port (e.g., 2): ").strip()
        # If directly press Enter, use default port
        if not camera_index:
            camera_index = "/dev/video2"
            print(f"Using Camera: {camera_index}")
        else:
            camera_index = f"/dev/video{camera_index}"
            print(f"Connecting to Camera: {camera_index}")
        
        # Configure robot
        robot = create_real_robot(port=port, camera_index=camera_index)
        
        # Create kinematics and controller instances
        kinematics = SO101Kinematics()
        controller = SO101VRTeleopController(robot, kinematics)
        
        # Initialize VR monitor
        vr_monitor = VRMonitor()
        
        # Start VR monitoring in separate thread
        vr_thread = threading.Thread(target=lambda: asyncio.run(vr_monitor.start_monitoring()), daemon=True)
        vr_thread.start()
        
        # Wait for VR monitor to initialize
        time.sleep(2)
        
        # Connect devices
        robot.connect()
        
        print("Devices connected successfully!")
        
        # Ask whether to recalibrate
        while True:
            calibrate_choice = input("Recalibrate robot? (y/n): ").strip().lower()
            if calibrate_choice in ['y', 'yes']:
                print("Starting recalibration...")
                robot.calibrate()
                print("Calibration complete!")
                break
            elif calibrate_choice in ['n', 'no']:
                print("Using previous calibration file")
                break
            else:
                print("Please enter y or n")
        
        # Initialize controller
        controller.initialize()
        
        # Move to zero position
        controller.move_to_zero_position(duration=3.0)
        
        print(f"Initial end effector position: x={controller.current_x:.4f}, y={controller.current_y:.4f}")
        
        # Start control loop
        controller.control_loop(vr_monitor)
        
        # Return to start position before exit
        controller.return_to_start_position()
        
        # Disconnect
        robot.disconnect()
        
        # Stop VR monitoring
        if vr_monitor.is_running:
            asyncio.run(vr_monitor.stop_monitoring())
            
        print("Program ended")
        
    except Exception as e:
        print(f"Program execution failed: {e}")
        traceback.print_exc()
        print("Please check:")
        print("1. Robot is correctly connected")
        print("2. USB port is correct")
        print("3. CAMERA port is correct")
        print("4. Have sufficient permissions to access USB device")
        print("5. Robot is correctly configured")
        print("6. VR system is properly set up")


if __name__ == "__main__":
    main()