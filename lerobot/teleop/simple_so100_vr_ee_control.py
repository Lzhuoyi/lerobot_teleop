#!/usr/bin/env python3
"""
简化的键盘控制SO100/SO101机器人
修复了动作格式转换问题
使用P控制，键盘只改变目标关节角度
"""

import time
import logging
import traceback
import math
import threading
import asyncio
import sys
import os
from lerobot.teleop.vr_monitor import VRMonitor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 关节校准系数 - 手动编辑
# 格式: [关节名称, 零位置偏移(度), 缩放系数]
JOINT_CALIBRATION = [
    ['shoulder_pan', 6.0, 1.0],      # 关节1: 零位置偏移, 缩放系数
    ['shoulder_lift', 2.0, 0.97],     # 关节2: 零位置偏移, 缩放系数
    ['elbow_flex', 0.0, 1.05],        # 关节3: 零位置偏移, 缩放系数
    ['wrist_flex', 0.0, 0.94],        # 关节4: 零位置偏移, 缩放系数
    ['wrist_roll', 0.0, 0.5],        # 关节5: 零位置偏移, 缩放系数
    ['gripper', 0.0, 1.0],           # 关节6: 零位置偏移, 缩放系数
]

def apply_joint_calibration(joint_name, raw_position):
    """
    应用关节校准系数
    
    Args:
        joint_name: 关节名称
        raw_position: 原始位置值
    
    Returns:
        calibrated_position: 校准后的位置值
    """
    for joint_cal in JOINT_CALIBRATION:
        if joint_cal[0] == joint_name:
            offset = joint_cal[1]  # 零位置偏移
            scale = joint_cal[2]   # 缩放系数
            calibrated_position = (raw_position - offset) * scale
            return calibrated_position
    return raw_position  # 如果没找到校准系数，返回原始值

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
    
    # Convert from radians to degrees
    joint2_deg = math.degrees(joint2)
    joint3_deg = math.degrees(joint3)

    joint2_deg = 90-joint2_deg
    joint3_deg = joint3_deg-90
    
    return joint2_deg, joint3_deg

def move_to_zero_position(robot, duration=3.0, kp=0.5):
    """
    使用P控制缓慢移动机器人到零位置
    
    Args:
        robot: 机器人实例
        duration: 移动到零位置所需时间(秒)
        kp: 比例增益
    """
    print("正在使用P控制缓慢移动机器人到零位置...")
    
    # 获取当前机器人状态
    current_obs = robot.get_observation()
    
    # 提取当前关节位置
    current_positions = {}
    for key, value in current_obs.items():
        if key.endswith('.pos'):
            motor_name = key.removesuffix('.pos')
            current_positions[motor_name] = value
    
    # 零位置目标
    zero_positions = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 0.0
    }
    
    # 计算控制步数
    control_freq = 50  # 50Hz控制频率
    total_steps = int(duration * control_freq)
    step_time = 1.0 / control_freq
    
    print(f"将在 {duration} 秒内使用P控制移动到零位置，控制频率: {control_freq}Hz，比例增益: {kp}")
    
    for step in range(total_steps):
        # 获取当前机器人状态
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                # 应用校准系数
                calibrated_value = apply_joint_calibration(motor_name, value)
                current_positions[motor_name] = calibrated_value
        
        # P控制计算
        robot_action = {}
        for joint_name, target_pos in zero_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                error = target_pos - current_pos
                
                # P控制: 输出 = Kp * 误差
                control_output = kp * error
                
                # 将控制输出转换为位置命令
                new_position = current_pos + control_output
                robot_action[f"{joint_name}.pos"] = new_position
        
        # 发送动作到机器人
        if robot_action:
            robot.send_action(robot_action)
        
        # 显示进度
        if step % (control_freq // 2) == 0:  # 每0.5秒显示一次进度
            progress = (step / total_steps) * 100
            print(f"移动到零位置进度: {progress:.1f}%")
        
        time.sleep(step_time)
    
    print("机器人已移动到零位置")

def return_to_start_position(robot, start_positions, kp=0.5, control_freq=50):
    """
    使用P控制返回到起始位置
    
    Args:
        robot: 机器人实例
        start_positions: 起始关节位置字典
        kp: 比例增益
        control_freq: 控制频率(Hz)
    """
    print("正在返回到起始位置...")
    
    control_period = 1.0 / control_freq
    max_steps = int(5.0 * control_freq)  # 最多5秒
    
    for step in range(max_steps):
        # 获取当前机器人状态
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                current_positions[motor_name] = value  # 不应用校准系数
        
        # P控制计算
        robot_action = {}
        total_error = 0
        for joint_name, target_pos in start_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                error = target_pos - current_pos
                total_error += abs(error)
                
                # P控制: 输出 = Kp * 误差
                control_output = kp * error
                
                # 将控制输出转换为位置命令
                new_position = current_pos + control_output
                robot_action[f"{joint_name}.pos"] = new_position
        
        # 发送动作到机器人
        if robot_action:
            robot.send_action(robot_action)
        
        # 检查是否到达起始位置
        if total_error < 2.0:  # 如果总误差小于2度，认为已到达
            print("已返回到起始位置")
            break
        
        time.sleep(control_period)
    
    print("返回起始位置完成")

def p_control_loop(robot, keyboard, target_positions, start_positions, current_x, current_y, kp=0.5, control_freq=50):
    """
    P控制循环
    
    Args:
        robot: 机器人实例
        keyboard: 键盘实例
        target_positions: 目标关节位置字典
        start_positions: 起始关节位置字典
        current_x: 当前x坐标
        current_y: 当前y坐标
        kp: 比例增益
        control_freq: 控制频率(Hz)
    """
    control_period = 1.0 / control_freq
    
    # 初始化pitch控制变量
    pitch = 0.0  # 初始pitch调整
    pitch_step = 1  # pitch调整步长
    
    print(f"开始P控制循环，控制频率: {control_freq}Hz，比例增益: {kp}")
    
    while True:
        try:
            # 获取键盘输入
            keyboard_action = keyboard.get_action()
            
            if keyboard_action:
                # 处理键盘输入，更新目标位置
                for key, value in keyboard_action.items():
                    if key == 'x':
                        # 退出程序，先回到起始位置
                        print("检测到退出命令，正在回到起始位置...")
                        return_to_start_position(robot, start_positions, 0.2, control_freq)
                        return
                    
                    # 关节控制映射
                    joint_controls = {
                        'q': ('shoulder_pan', -1),    # 关节1减少
                        'a': ('shoulder_pan', 1),     # 关节1增加
                        't': ('wrist_roll', -1),      # 关节5减少
                        'g': ('wrist_roll', 1),       # 关节5增加
                        'y': ('gripper', -1),         # 关节6减少
                        'h': ('gripper', 1),          # 关节6增加
                    }
                    
                    # x,y坐标控制
                    xy_controls = {
                        'w': ('x', -0.004),  # x减少
                        's': ('x', 0.004),   # x增加
                        'e': ('y', -0.004),  # y减少
                        'd': ('y', 0.004),   # y增加
                    }
                    
                    # pitch控制
                    if key == 'r':
                        pitch += pitch_step
                        print(f"增加pitch调整: {pitch:.3f}")
                    elif key == 'f':
                        pitch -= pitch_step
                        print(f"减少pitch调整: {pitch:.3f}")
                    
                    if key in joint_controls:
                        joint_name, delta = joint_controls[key]
                        if joint_name in target_positions:
                            current_target = target_positions[joint_name]
                            new_target = int(current_target + delta)
                            target_positions[joint_name] = new_target
                            print(f"更新目标位置 {joint_name}: {current_target} -> {new_target}")
                    
                    elif key in xy_controls:
                        coord, delta = xy_controls[key]
                        if coord == 'x':
                            current_x += delta
                            # 计算joint2和joint3的目标角度
                            joint2_target, joint3_target = inverse_kinematics(current_x, current_y)
                            target_positions['shoulder_lift'] = joint2_target
                            target_positions['elbow_flex'] = joint3_target
                            print(f"更新x坐标: {current_x:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")
                        elif coord == 'y':
                            current_y += delta
                            # 计算joint2和joint3的目标角度
                            joint2_target, joint3_target = inverse_kinematics(current_x, current_y)
                            target_positions['shoulder_lift'] = joint2_target
                            target_positions['elbow_flex'] = joint3_target
                            print(f"更新y坐标: {current_y:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")
            
            # 应用pitch调整到wrist_flex
            # 基于shoulder_lift和elbow_flex计算wrist_flex的目标位置
            if 'shoulder_lift' in target_positions and 'elbow_flex' in target_positions:
                target_positions['wrist_flex'] = - target_positions['shoulder_lift'] - target_positions['elbow_flex'] + pitch
                # 显示当前pitch值（每100步显示一次，避免刷屏）
                if hasattr(p_control_loop, 'step_counter'):
                    p_control_loop.step_counter += 1
                else:
                    p_control_loop.step_counter = 0
                
                if p_control_loop.step_counter % 100 == 0:
                    print(f"当前pitch调整: {pitch:.3f}, wrist_flex目标: {target_positions['wrist_flex']:.3f}")
            
            # 获取当前机器人状态
            current_obs = robot.get_observation()
            
            # 提取当前关节位置
            current_positions = {}
            for key, value in current_obs.items():
                if key.endswith('.pos'):
                    motor_name = key.removesuffix('.pos')
                    # 应用校准系数
                    calibrated_value = apply_joint_calibration(motor_name, value)
                    current_positions[motor_name] = calibrated_value
            
            # P控制计算
            robot_action = {}
            for joint_name, target_pos in target_positions.items():
                if joint_name in current_positions:
                    current_pos = current_positions[joint_name]
                    error = target_pos - current_pos
                    
                    # P控制: 输出 = Kp * 误差
                    control_output = kp * error
                    
                    # 将控制输出转换为位置命令
                    new_position = current_pos + control_output
                    robot_action[f"{joint_name}.pos"] = new_position
            
            # 发送动作到机器人
            if robot_action:
                robot.send_action(robot_action)
            
            time.sleep(control_period)
            
        except KeyboardInterrupt:
            print("用户中断程序")
            break
        except Exception as e:
            print(f"P控制循环出错: {e}")
            traceback.print_exc()
            break

def main():
    """主函数"""
    print("LeRobot VR控制示例 (P控制+IK)")
    print("="*50)
    
    try:
        from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
        # 获取端口
        port = input("请输入SO100机器人的USB端口 (例如: /dev/ttyACM0): ").strip()
        if not port:
            port = "/dev/ttyACM0"
            print(f"使用默认端口: {port}")
        else:
            print(f"连接到端口: {port}")
        robot_config = SO100FollowerConfig(port=port)
        robot = SO100Follower(robot_config)
        robot.connect()
        print("设备连接成功！")
        # 询问是否重新校准
        while True:
            calibrate_choice = input("是否重新校准机器人? (y/n): ").strip().lower()
            if calibrate_choice in ['y', 'yes', '是']:
                print("开始重新校准...")
                robot.calibrate()
                print("校准完成！")
                break
            elif calibrate_choice in ['n', 'no', '否']:
                print("使用之前的校准文件")
                break
            else:
                print("请输入 y 或 n")
        # 读取起始关节角度
        print("读取起始关节角度...")
        start_obs = robot.get_observation()
        start_positions = {}
        for key, value in start_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                start_positions[motor_name] = int(value)
        print("起始关节角度:")
        for joint_name, position in start_positions.items():
            print(f"  {joint_name}: {position}°")
        # 移动到零位置
        move_to_zero_position(robot, duration=3.0)
        # 初始化目标位置为零
        target_positions = {
            'shoulder_pan': 0.0,
            'shoulder_lift': 0.0,
            'elbow_flex': 0.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 0.0
        }
        # 初始化末端执行器xy
        x0, y0 = 0.1629, 0.1131
        current_x, current_y = x0, y0
        pitch = 0.0
        print(f"初始化末端执行器位置: x={current_x:.4f}, y={current_y:.4f}")
        # 启动VRMonitor
        vr_monitor = VRMonitor()
        vr_thread = threading.Thread(target=lambda: asyncio.run(vr_monitor.start_monitoring()), daemon=True)
        vr_thread.start()
        print("等待VR控制器连接...")
        # 等待VR控制器激活（双手trigger同时按下3秒）
        activation_time = 3.0
        print(f"请同时按住两个VR控制器的trigger按钮{activation_time}秒以激活VR控制...")
        trigger_start = None
        while True:
            dual_goals = vr_monitor.get_latest_goal_nowait()
            left_goal = dual_goals.get("left") if dual_goals else None
            right_goal = dual_goals.get("right") if dual_goals else None
            left_trigger = left_goal and left_goal.metadata and left_goal.metadata.get('trigger', 0) > 0.5
            right_trigger = right_goal and right_goal.metadata and right_goal.metadata.get('trigger', 0) > 0.5
            if left_trigger and right_trigger:
                if trigger_start is None:
                    trigger_start = time.time()
                elif time.time() - trigger_start >= activation_time:
                    print("VR控制已激活！")
                    break
            else:
                trigger_start = None
            time.sleep(0.05)
        print("进入VR控制循环。按Ctrl+C退出。")
        # 控制参数
        kp = 0.5
        control_freq = 50
        control_period = 1.0 / control_freq
        tip_length = 0.108
        pitch_step = 1.0
        try:
            while True:
                dual_goals = vr_monitor.get_latest_goal_nowait()
                left_goal = dual_goals.get("left") if dual_goals else None
                # 只用右手控制主臂
                right_goal = dual_goals.get("right") if dual_goals else None
                # VR控制信号转末端xy
                if right_goal and right_goal.target_position:
                    pos = right_goal.target_position
                    # 这里可根据实际VR坐标做缩放/偏移
                    x_vr = (pos[0] - 0.1) * 0.5
                    y_vr = (pos[1] - 0.96) * 1.0
                    # 限制工作空间
                    r = math.sqrt(x_vr**2 + y_vr**2)
                    if r > 0.24:
                        scale = 0.24 / r
                        x_vr *= scale
                        y_vr *= scale
                    current_x = x_vr
                    current_y = y_vr
                    # pitch控制
                    if right_goal.wrist_flex_deg is not None:
                        pitch = (right_goal.wrist_flex_deg + 60) * 0.02
                # IK
                try:
                    compensated_y = current_y + tip_length * math.sin(pitch)
                    joint2_target, joint3_target = inverse_kinematics(current_x, compensated_y)
                    target_positions['shoulder_lift'] = joint2_target
                    target_positions['elbow_flex'] = joint3_target
                    target_positions['wrist_flex'] = -joint2_target - joint3_target + pitch
                except Exception as e:
                    print(f"IK计算失败: {e}")
                # wrist_roll控制
                if right_goal and right_goal.wrist_roll_deg is not None:
                    target_positions['wrist_roll'] = -(right_goal.wrist_roll_deg-90) * 0.5
                # gripper控制
                if right_goal and right_goal.metadata:
                    if right_goal.metadata.get('trigger', 0) > 0.5:
                        target_positions['gripper'] = 0.0  # 关
                    else:
                        target_positions['gripper'] = 100.0  # 开
                # P控制
                current_obs = robot.get_observation()
                current_positions = {}
                for key, value in current_obs.items():
                    if key.endswith('.pos'):
                        motor_name = key.removesuffix('.pos')
                        calibrated_value = apply_joint_calibration(motor_name, value)
                        current_positions[motor_name] = calibrated_value
                robot_action = {}
                for joint_name, target_pos in target_positions.items():
                    if joint_name in current_positions:
                        current_pos = current_positions[joint_name]
                        error = target_pos - current_pos
                        control_output = kp * error
                        new_position = current_pos + control_output
                        robot_action[f"{joint_name}.pos"] = new_position
                if robot_action:
                    robot.send_action(robot_action)
                time.sleep(control_period)
        except KeyboardInterrupt:
            print("用户中断程序，正在退出...")
        finally:
            print("正在关闭VR监控...")
            if vr_monitor.is_running:
                asyncio.run(vr_monitor.stop_monitoring())
            robot.disconnect()
            print("程序结束")
    except Exception as e:
        print(f"程序执行失败: {e}")
        traceback.print_exc()
        print("请检查:")
        print("1. 机器人是否正确连接")
        print("2. USB端口是否正确")
        print("3. 是否有足够的权限访问USB设备")
        print("4. 机器人是否已正确配置")

if __name__ == "__main__":
    main() 