#!/usr/bin/env python3
"""
python -m lerobot.teleop.teleop_keyboard_v2
"""

import time
import logging
import traceback
from lerobot.teleop.SO101Robot import SO101Kinematics, create_real_robot
from lerobot.common.utils.visualization_utils import _init_rerun, log_rerun_data

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

class SO101TeleopController:
    """SO101机器人远程操作控制器"""
    
    def __init__(self, robot, kinematics=None):
        self.robot = robot
        self.kinematics = kinematics if kinematics else SO101Kinematics()
        
        # 控制参数
        self.kp = 0.5  # 比例增益
        self.control_freq = 50  # 控制频率(Hz)
        
        # 末端执行器初始位置
        self.current_x = 0.1629
        self.current_y = 0.1131
        
        # Pitch控制参数
        self.pitch = 0.0
        self.pitch_step = 1.0
        
        # 记录起始位置
        self.start_positions = {}
        
        # 目标位置
        self.target_positions = {
            'shoulder_pan': 0.0,
            'shoulder_lift': 0.0,
            'elbow_flex': 0.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 0.0
        }

    def apply_joint_calibration(self, joint_name, raw_position):
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

    def move_to_zero_position(self, duration=3.0):
        """
        使用P控制缓慢移动机器人到零位置
        
        Args:
            duration: 移动到零位置所需时间(秒)
        """
        print("正在使用P控制缓慢移动机器人到零位置...")
        
        # 获取当前机器人状态
        current_obs = self.robot.get_observation()
        
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
        total_steps = int(duration * self.control_freq)
        step_time = 1.0 / self.control_freq
        
        print(f"将在 {duration} 秒内使用P控制移动到零位置，控制频率: {self.control_freq}Hz，比例增益: {self.kp}")
        
        for step in range(total_steps):
            # 获取当前机器人状态
            current_obs = self.robot.get_observation()
            current_positions = {}
            for key, value in current_obs.items():
                if key.endswith('.pos'):
                    motor_name = key.removesuffix('.pos')
                    # 应用校准系数
                    calibrated_value = self.apply_joint_calibration(motor_name, value)
                    current_positions[motor_name] = calibrated_value
            
            # P控制计算
            robot_action = {}
            for joint_name, target_pos in zero_positions.items():
                if joint_name in current_positions:
                    current_pos = current_positions[joint_name]
                    error = target_pos - current_pos
                    
                    # P控制: 输出 = Kp * 误差
                    control_output = self.kp * error
                    
                    # 将控制输出转换为位置命令
                    new_position = current_pos + control_output
                    robot_action[f"{joint_name}.pos"] = new_position
            
            # 发送动作到机器人
            if robot_action:
                self.robot.send_action(robot_action)
            
            # 显示进度
            if step % (self.control_freq // 2) == 0:  # 每0.5秒显示一次进度
                progress = (step / total_steps) * 100
                print(f"移动到零位置进度: {progress:.1f}%")
            
            time.sleep(step_time)
        
        print("机器人已移动到零位置")

    def return_to_start_position(self):
        """
        使用P控制返回到起始位置
        """
        print("正在返回到起始位置...")
        
        control_period = 1.0 / self.control_freq
        max_steps = int(5.0 * self.control_freq)  # 最多5秒
        
        for step in range(max_steps):
            # 获取当前机器人状态
            current_obs = self.robot.get_observation()
            current_positions = {}
            for key, value in current_obs.items():
                if key.endswith('.pos'):
                    motor_name = key.removesuffix('.pos')
                    current_positions[motor_name] = value  # 不应用校准系数
            
            # P控制计算
            robot_action = {}
            total_error = 0
            for joint_name, target_pos in self.start_positions.items():
                if joint_name in current_positions:
                    current_pos = current_positions[joint_name]
                    error = target_pos - current_pos
                    total_error += abs(error)
                    
                    # P控制: 输出 = Kp * 误差
                    control_output = self.kp * error
                    
                    # 将控制输出转换为位置命令
                    new_position = current_pos + control_output
                    robot_action[f"{joint_name}.pos"] = new_position
            
            # 发送动作到机器人
            if robot_action:
                self.robot.send_action(robot_action)
            
            # 检查是否到达起始位置
            if total_error < 2.0:  # 如果总误差小于2度，认为已到达
                print("已返回到起始位置")
                break
            
            time.sleep(control_period)
        
        print("返回起始位置完成")

    def handle_keyboard_input(self, keyboard_action):
        """
        处理键盘输入，更新目标位置
        
        Args:
            keyboard_action: 键盘动作字典
            
        Returns:
            bool: 如果检测到退出命令返回False，否则返回True
        """
        if not keyboard_action:
            return True
            
        for key, value in keyboard_action.items():
            if key == 'c':
                # 退出程序，先回到起始位置
                print("检测到退出命令，正在回到起始位置...")
                self.return_to_start_position()
                return False
            
            # 关节控制映射
            joint_controls = {
                'r': ('shoulder_pan', 1),    # 关节1减少
                'f': ('shoulder_pan', -1),     # 关节1增加
                't': ('wrist_roll', 1),      # 关节5减少
                'g': ('wrist_roll', -1),       # 关节5增加
                'q': ('gripper', 1),         # 关节6减少
                'e': ('gripper', -1),          # 关节6增加
            }
            
            # x,y坐标控制
            xy_controls = {
                'w': ('x', 0.0021),  # x减少
                's': ('x', -0.0021),   # x增加
                'a': ('y', 0.0021),  # y减少
                'd': ('y', -0.0021),   # y增加
            }
            
            # pitch控制
            if key == 'z':
                self.pitch += self.pitch_step
                print(f"增加pitch调整: {self.pitch:.3f}")
            elif key == 'x':
                self.pitch -= self.pitch_step
                print(f"减少pitch调整: {self.pitch:.3f}")
            
            if key in joint_controls:
                joint_name, delta = joint_controls[key]
                if joint_name in self.target_positions:
                    current_target = self.target_positions[joint_name]
                    new_target = int(current_target + delta)
                    self.target_positions[joint_name] = new_target
                    print(f"更新目标位置 {joint_name}: {current_target} -> {new_target}")
            
            elif key in xy_controls:
                coord, delta = xy_controls[key]
                if coord == 'x':
                    self.current_x += delta
                    # 计算joint2和joint3的目标角度
                elif coord == 'y':
                    self.current_y += delta
                    # 计算joint2和joint3的目标角度
                joint2_target, joint3_target = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
                self.target_positions['shoulder_lift'] = joint2_target
                self.target_positions['elbow_flex'] = joint3_target
                print(f"更新坐标: ({self.current_x:.4f}, {self.current_y:.4f}), joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")
        
        return True

    def update_wrist_flex_with_pitch(self):
        """根据shoulder_lift和elbow_flex计算wrist_flex的目标位置，加上pitch调整"""
        if 'shoulder_lift' in self.target_positions and 'elbow_flex' in self.target_positions:
            self.target_positions['wrist_flex'] = (
                -self.target_positions['shoulder_lift'] 
                - self.target_positions['elbow_flex'] 
                + self.pitch
            )

    def execute_p_control_step(self):
        """执行一步P控制"""
        # 获取当前机器人状态
        current_obs = self.robot.get_observation()
        
        # 提取当前关节位置
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                # 应用校准系数
                calibrated_value = self.apply_joint_calibration(motor_name, value)
                current_positions[motor_name] = calibrated_value
        
        # P控制计算
        robot_action = {}
        for joint_name, target_pos in self.target_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                error = target_pos - current_pos
                
                # P控制: 输出 = Kp * 误差
                control_output = self.kp * error
                
                # 将控制输出转换为位置命令
                new_position = current_pos + control_output
                robot_action[f"{joint_name}.pos"] = new_position
        
        # 发送动作到机器人
        if robot_action:
            self.robot.send_action(robot_action)

        # 记录数据到Rerun
        log_rerun_data(current_obs, robot_action)

    def control_loop(self, keyboard):
        """
        主要的控制循环
        
        Args:
            keyboard: 键盘实例
        """
        control_period = 1.0 / self.control_freq
        step_counter = 0
        
        print(f"开始P控制循环，控制频率: {self.control_freq}Hz，比例增益: {self.kp}")
        
        while True:
            try:
                # 获取键盘输入
                keyboard_action = keyboard.get_action()
                
                # 处理键盘输入
                if not self.handle_keyboard_input(keyboard_action):
                    break  # 退出循环
                
                # 更新wrist_flex位置
                self.update_wrist_flex_with_pitch()
                
                # 显示pitch值（每100步显示一次，避免刷屏）
                step_counter += 1
                if step_counter % 100 == 0:
                    print(f"当前pitch调整: {self.pitch:.3f}, wrist_flex目标: {self.target_positions['wrist_flex']:.3f}")
                
                # 执行P控制
                self.execute_p_control_step()

                
                time.sleep(control_period)
                
            except KeyboardInterrupt:
                print("用户中断程序")
                break
            except Exception as e:
                print(f"P控制循环出错: {e}")
                traceback.print_exc()
                break

    def initialize(self):
        """初始化控制器"""
        # 读取起始关节角度
        print("读取起始关节角度...")
        start_obs = self.robot.get_observation()
        for key, value in start_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                self.start_positions[motor_name] = int(value)  # 不应用校准系数
        
        print("起始关节角度:")
        for joint_name, position in self.start_positions.items():
            print(f"  {joint_name}: {position}°")

def main():
    """主函数"""
    print("LeRobot SO101 键盘控制示例 (P控制)")
    print("="*50)
    # 初始化Rerun
    _init_rerun(session_name="SO101KeyboardControl")
    
    try:
        # 导入必要的模块
        from lerobot.common.robots.so100_follower import SO100Follower, SO100FollowerConfig
        from lerobot.common.robots.so101_follower import SO101Follower, SO101FollowerConfig
        from lerobot.common.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
        
        # 获取端口
        port = input("请输入SO101机器人的USB端口 (例如: /dev/ttyACM0): ").strip()
        # 如果直接按回车，使用默认端口
        if not port:
            port = "/dev/ttyACM0"
            print(f"使用默认端口: {port}")
        else:
            print(f"连接到端口: {port}")

        # 获取端口
        camera_index = input("请输入SO101机器人的CAMERA端口 (例如: 2): ").strip()
        # 如果直接按回车，使用默认端口
        if not camera_index:
            camera_index = "/dev/video2"
            print(f"使用Camera: {camera_index}")
        else:
            print(f"连接到Camera: {camera_index}")
        
        # 配置机器人
        robot = create_real_robot(port = port, camera_index = camera_index)
        
        # 配置键盘
        keyboard_config = KeyboardTeleopConfig()
        keyboard = KeyboardTeleop(keyboard_config)
        
        # 创建运动学和控制器实例
        kinematics = SO101Kinematics()
        controller = SO101TeleopController(robot, kinematics)
        
        # 连接设备
        robot.connect()
        keyboard.connect()
        
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
        
        # 初始化控制器
        controller.initialize()
        
        # 移动到零位置
        controller.move_to_zero_position(duration=3.0)
        
        print(f"初始化末端执行器位置: x={controller.current_x:.4f}, y={controller.current_y:.4f}")
        
        print("键盘控制说明:")
        print("- Q/A: 关节1 (shoulder_pan) 减少/增加")
        print("- W/S: 控制末端执行器x坐标 (joint2+3)")
        print("- E/D: 控制末端执行器y坐标 (joint2+3)")
        print("- R/F: pitch调整 增加/减少 (影响wrist_flex)")
        print("- T/G: 关节5 (wrist_roll) 减少/增加")
        print("- Y/H: 关节6 (gripper) 减少/增加")
        print("- X: 退出程序（先回到起始位置）")
        print("- ESC: 退出程序")
        print("="*50)
        print("注意: 机器人会持续移动到目标位置")
        
        # 开始控制循环
        controller.control_loop(keyboard)
        
        # 断开连接
        robot.disconnect()
        keyboard.disconnect()
        print("程序结束")
        
    except Exception as e:
        print(f"程序执行失败: {e}")
        traceback.print_exc()
        print("请检查:")
        print("1. 机器人是否正确连接")
        print("2. USB端口是否正确")
        print("3. CAMERA端口是否正确")
        print("4. 是否有足够的权限访问USB设备")
        print("5. 机器人是否已正确配置")

if __name__ == "__main__":
    main()