# To Run the teleop:
'''python
PYTHONPATH=src python -m examples.xlerobot.test_xbox
'''

import pygame
import time

class XboxControllerRaw:
    def __init__(self):
        # 初始化pygame
        pygame.init()
        pygame.joystick.init()
        
        # 检查手柄连接
        if pygame.joystick.get_count() == 0:
            print("没有检测到手柄，请确保Xbox手柄已连接")
            return
        
        # 获取第一个手柄
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"已连接手柄: {self.joystick.get_name()}")
        print(f"按钮数量: {self.joystick.get_numbuttons()}")
        print(f"轴数量: {self.joystick.get_numaxes()}")
        print(f"方向键数量: {self.joystick.get_numhats()}")
        
        # Xbox手柄按钮名称映射
        self.button_names = {
            0: "A",
            1: "B", 
            2: "X",
            3: "Y",
            4: "LB",
            5: "RB",
            6: "Back/View",
            7: "Start/Menu",
            8: "左摇杆按下",
            9: "右摇杆按下",
            10: "Xbox按钮",
            11: "左摇杆按下",
            12: "右摇杆按下"
        }
        
        # Xbox手柄轴名称映射
        self.axis_names = {
            0: "右摇杆X轴",
            1: "右摇杆Y轴", 
            2: "右扳机(RT)",
            3: "左摇杆X轴",
            4: "左摇杆Y轴",
            5: "左扳机(LT)"
        }
        
        # 方向键名称映射
        self.hat_names = {
            (0, -1): "上",
            (0, 1): "下",
            (-1, 0): "左", 
            (1, 0): "右",
            (-1, -1): "左上",
            (1, -1): "右上",
            (-1, 1): "左下",
            (1, 1): "右下"
        }
        
        self.running = False
        
    def start_listening(self):
        """开始监听手柄输入"""
        self.running = True
        print("开始监听Xbox手柄原始输入... 按Ctrl+C退出")
        print("=" * 50)
        
        try:
            while self.running:
                pygame.event.pump()
                
                # 打印所有按钮状态
                self.print_buttons()
                
                # 打印所有轴状态
                self.print_axes()
                
                # 打印方向键状态
                self.print_hats()
                
                time.sleep(0.1)  # 每0.1秒更新一次
                
        except KeyboardInterrupt:
            print("\n正在退出...")
            self.stop_listening()
    
    def print_buttons(self):
        """打印所有按钮状态"""
        button_states = []
        for i in range(self.joystick.get_numbuttons()):
            pressed = self.joystick.get_button(i)
            button_name = self.button_names.get(i, f"按钮{i}")
            if pressed:
                button_states.append(f"{button_name}:按下")
        
        if button_states:
            print(f"按钮: {' '.join(button_states)}")
    
    def print_axes(self):
        """打印所有轴状态"""
        axis_values = []
        for i in range(self.joystick.get_numaxes()):
            value = self.joystick.get_axis(i)
            # 只打印有变化的轴（值不为0）
            if abs(value) > 0.01:
                axis_name = self.axis_names.get(i, f"轴{i}")
                axis_values.append(f"{axis_name}:{value:.3f}")
        
        if axis_values:
            print(f"轴: {' '.join(axis_values)}")
    
    def print_hats(self):
        """打印方向键状态"""
        for i in range(self.joystick.get_numhats()):
            hat = self.joystick.get_hat(i)
            if hat != (0, 0):  # 只打印有按下的方向键
                hat_name = self.hat_names.get(hat, f"方向键{hat}")
                print(f"方向键: {hat_name}")
    
    def stop_listening(self):
        """停止监听"""
        self.running = False
        pygame.quit()
        print("已停止监听")

def main():
    """主函数"""
    controller = XboxControllerRaw()
    
    if hasattr(controller, 'joystick'):
        try:
            controller.start_listening()
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            controller.stop_listening()
    else:
        print("无法初始化手柄")

if __name__ == "__main__":
    main()