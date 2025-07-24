#!/usr/bin/env python3
"""
VR Monitor - 独立的VR控制信息监控脚本
可以在其他文件夹中调用xlevr的VR功能并读取打印VR控制信息
"""

import os
import sys
import asyncio
import json
import logging
import threading
import http.server
import ssl
import socket
from pathlib import Path
from typing import Optional



# 设置xlevr文件夹的绝对路径
XLEVR_PATH = "/home/jellyfish/lerobot_ws/XLeRobot/XLeVR"

def setup_xlevr_environment():
    """设置xlevr环境"""
    # 添加xlevr路径到Python路径
    if XLEVR_PATH not in sys.path:
        sys.path.insert(0, XLEVR_PATH)
    
    # 设置工作目录
    os.chdir(XLEVR_PATH)
    
    # 设置环境变量
    os.environ['PYTHONPATH'] = f"{XLEVR_PATH}:{os.environ.get('PYTHONPATH', '')}"

def get_local_ip():
    """Get the local IP address of this machine."""
    try:
        # Connect to a remote address to determine the local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        try:
            # Fallback: get hostname IP
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            # Final fallback
            return "localhost"

def import_xlevr_modules():
    """导入xlevr模块"""
    try:
        from xlevr.config import XLeVRConfig
        from xlevr.inputs.vr_ws_server import VRWebSocketServer
        from xlevr.inputs.base import ControlGoal, ControlMode
        return XLeVRConfig, VRWebSocketServer, ControlGoal, ControlMode
    except ImportError as e:
        print(f"Error importing xlevr modules: {e}")
        print(f"Make sure XLEVR_PATH is correct: {XLEVR_PATH}")
        return None, None, None, None

class SimpleAPIHandler(http.server.BaseHTTPRequestHandler):
    """简化的HTTP请求处理器，只提供基本的网页服务"""
    
    def end_headers(self):
        """Add CORS headers to all responses."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        try:
            super().end_headers()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, ssl.SSLError):
            pass
    
    def do_OPTIONS(self):
        """Handle preflight CORS requests."""
        self.send_response(200)
        self.end_headers()
    
    def log_message(self, format, *args):
        """Override to reduce HTTP request logging noise."""
        pass  # Disable default HTTP logging
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/' or self.path == '/index.html':
            # Serve main page from web-ui directory
            self.serve_file('web-ui/index.html', 'text/html')
        elif self.path.endswith('.css'):
            # Serve CSS files from web-ui directory
            self.serve_file(f'web-ui{self.path}', 'text/css')
        elif self.path.endswith('.js'):
            # Serve JS files from web-ui directory
            self.serve_file(f'web-ui{self.path}', 'application/javascript')
        elif self.path.endswith('.ico'):
            self.serve_file(self.path[1:], 'image/x-icon')
        elif self.path.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            # Serve image files from web-ui directory
            content_type = 'image/jpeg' if self.path.endswith(('.jpg', '.jpeg')) else 'image/png' if self.path.endswith('.png') else 'image/gif'
            self.serve_file(f'web-ui{self.path}', content_type)
        else:
            self.send_error(404, "Not found")
    
    def serve_file(self, filename, content_type):
        """Serve a file with the given content type."""
        try:
            # Get the web root path from the server
            web_root = getattr(self.server, 'web_root_path', XLEVR_PATH)
            file_path = os.path.join(web_root, filename)
            
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, f"File not found: {filename}")
        except Exception as e:
            print(f"Error serving file {filename}: {e}")
            self.send_error(500, "Internal server error")

class SimpleHTTPSServer:
    """简化的HTTPS服务器，用于提供网页界面"""
    
    def __init__(self, config):
        self.config = config
        self.httpd = None
        self.server_thread = None
        self.web_root_path = XLEVR_PATH
    
    async def start(self):
        """Start the HTTPS server."""
        try:
            # Create server
            self.httpd = http.server.HTTPServer((self.config.host_ip, self.config.https_port), SimpleAPIHandler)
            
            # Set web root path for file serving
            self.httpd.web_root_path = self.web_root_path
            
            # Setup SSL
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain('cert.pem', 'key.pem')
            self.httpd.socket = context.wrap_socket(self.httpd.socket, server_side=True)
            
            # Start server in a separate thread
            self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self.server_thread.start()
            
            print(f"🌐 HTTPS server started on {self.config.host_ip}:{self.config.https_port}")
            
        except Exception as e:
            print(f"❌ Failed to start HTTPS server: {e}")
            raise
    
    async def stop(self):
        """Stop the HTTPS server."""
        if self.httpd:
            self.httpd.shutdown()
            if self.server_thread:
                self.server_thread.join(timeout=5)
            print("🌐 HTTPS server stopped")

class VRMonitor:
    """VR控制信息监控器"""
    
    def __init__(self):
        self.config = None
        self.vr_server = None
        self.https_server = None
        self.is_running = False
        self.latest_goal = None  # Store the latest goal
        self.left_goal = None    # 分别保存左右控制器的goal
        self.right_goal = None
        self.headset_goal = None  # 添加头部goal
        self._goal_lock = threading.Lock()  # 添加线程锁
    
    def initialize(self):
        """初始化VR监控器"""
        print("🔧 Initializing XLeVR Monitor...")
        
        # 设置环境
        setup_xlevr_environment()
        
        # 导入模块
        XLeVRConfig, VRWebSocketServer, ControlGoal, ControlMode = import_xlevr_modules()
        if XLeVRConfig is None:
            print("❌ Failed to import xlevr modules")
            return False
        
        # 创建配置
        self.config = XLeVRConfig()
        self.config.enable_vr = True
        self.config.enable_keyboard = False
        self.config.enable_https = True  # 启用HTTPS服务器，VR需要网页界面
        
        # 创建命令队列
        self.command_queue = asyncio.Queue()
        
        # 创建VR服务器（print-only模式）
        try:
            self.vr_server = VRWebSocketServer(
                command_queue=self.command_queue,
                config=self.config,
                print_only=False  # 改为False，让数据发送到队列
            )
        except Exception as e:
            print(f"❌ Failed to create VR WebSocket server: {e}")
            return False
        
        # 创建HTTPS服务器
        try:
            self.https_server = SimpleHTTPSServer(self.config)
        except Exception as e:
            print(f"❌ Failed to create HTTPS server: {e}")
            return False
        
        print("✅ XLeVR Monitor initialized successfully")
        
        return True
    
    async def start_monitoring(self):
        """开始监控VR控制信息"""
        print("🚀 Starting VR Monitor...")
        
        if not self.initialize():
            print("❌ Failed to initialize VR monitor")
            return
        
        try:
            # 启动HTTPS服务器
            await self.https_server.start()
            
            # 启动VR服务器
            await self.vr_server.start()
            
            self.is_running = True
            print("✅ VR Monitor is now running")
            
            # 显示连接信息
            host_display = get_local_ip() if self.config.host_ip == "0.0.0.0" else self.config.host_ip
            print(f"📱 Open your VR headset browser and navigate to:")
            print(f"   https://{host_display}:{self.config.https_port}")
            print("🎯 Press Ctrl+C to stop monitoring")
            print()
            
            # 监控命令队列
            await self.monitor_commands()
            
        except KeyboardInterrupt:
            print("\n⏹️  Stopping VR monitor...")
        except Exception as e:
            print(f"❌ Error in VR monitor: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
        finally:
            await self.stop_monitoring()
    
    async def monitor_commands(self):
        """监控来自VR控制器的命令"""
        print("📊 Monitoring VR control commands...")
        
        while self.is_running:
            try:
                # 等待命令，超时1秒
                goal = await asyncio.wait_for(self.command_queue.get(), timeout=1.0)
                
                # 根据arm分别保存goal
                with self._goal_lock:
                    if goal.arm == "left":
                        self.left_goal = goal
                    elif goal.arm == "right":
                        self.right_goal = goal
                    elif goal.arm == "headset":  # 添加头部数据处理
                        self.headset_goal = goal
                    
                    # 保持向后兼容，保存最新的goal
                    self.latest_goal = goal
                
            except asyncio.TimeoutError:
                # 超时，继续循环
                continue
            except Exception as e:
                print(f"❌ Error processing command: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
    
    def print_control_goal(self, goal):
        """打印控制目标信息"""
        print(f"\n🎮 Control Goal Received:")
        print(f"   Timestamp: {asyncio.get_event_loop().time():.3f}")
        print(f"   Arm: {goal.arm}")
        print(f"   Mode: {goal.mode}")
        
        if goal.target_position is not None:
            if isinstance(goal.target_position, (list, tuple)):
                pos = goal.target_position
                print(f"   Target Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            else:
                print(f"   Target Position: {goal.target_position}")
        
        if goal.wrist_roll_deg is not None:
            print(f"   Wrist Roll: {goal.wrist_roll_deg:.1f}°")
        
        if goal.wrist_flex_deg is not None:
            print(f"   Wrist Flex: {goal.wrist_flex_deg:.1f}°")
        
        if goal.gripper_closed is not None:
            print(f"   Gripper: {'CLOSED' if goal.gripper_closed else 'OPEN'}")
        
        if goal.metadata:
            print(f"   Metadata: {goal.metadata}")
        
        print("-" * 30)
    
    def get_latest_goal_nowait(self, arm=None):
        """Return the latest VR control goal if available, else None.
        
        Args:
            arm: If specified ("left" or "right"), return that arm's goal.
                 If None, return a dict containing both left and right goals.
        """
        with self._goal_lock:
            if arm == "left":
                return self.left_goal
            elif arm == "right":
                return self.right_goal
            elif arm == "headset":  # 添加头部数据获取
                return self.headset_goal
            else:
                # 返回包含左右两个控制器的字典
                dual_goals = {
                    "left": self.left_goal,
                    "right": self.right_goal,
                    "headset": self.headset_goal,  # 添加头部数据
                    "has_left": self.left_goal is not None,
                    "has_right": self.right_goal is not None,
                    "has_headset": self.headset_goal is not None  # 添加头部状态
                }
                
                return dual_goals
    
    def get_left_goal_nowait(self):
        """Return the latest left arm goal if available, else None."""
        return self.get_latest_goal_nowait("left")
    
    def get_right_goal_nowait(self):
        """Return the latest right arm goal if available, else None."""
        return self.get_latest_goal_nowait("right")
    
    async def stop_monitoring(self):
        """停止监控"""
        self.is_running = False
        
        if self.vr_server:
            await self.vr_server.stop()
        
        if self.https_server:
            await self.https_server.stop()
        
        print("✅ VR Monitor stopped")

def main():
    """主函数"""
    print("🎮 XLeVR Monitor - XLeVR VR Control Information Monitor")
    print("=" * 60)
    
    # 检查telegrip路径
    if not os.path.exists(XLEVR_PATH):
        print(f"❌ XLeVR path does not exist: {XLEVR_PATH}")
        print("Please update XLEVR_PATH in the script")
        return
    
    # 创建监控器
    monitor = VRMonitor()
    
    # 运行监控
    try:
        asyncio.run(monitor.start_monitoring())
    except KeyboardInterrupt:
        print("\n👋 XLeVR Monitor stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 