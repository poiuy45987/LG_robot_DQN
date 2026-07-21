import os
import numpy as np

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import String, UInt8MultiArray, MultiArrayDimension
from std_srvs.srv import Trigger
from ament_index_python.packages import get_package_share_directory

from .mujoco_map_converter import MujocoGridConverter

class MujocoMapConverterNode(Node):
    def __init__(self):
        super().__init__('mujoco_map_converter_node')
        
        package_dir = get_package_share_directory('robot_vacuum_redqn')
        mujoco_env_folder = os.path.join(package_dir, 'mujoco_environments', 'environment', 'world')
        
        self.declare_parameter('mujoco_env_folder', mujoco_env_folder)
        self.declare_parameter('grid_size_cm', 2.0)
        self.declare_parameter('robot_h', 9.34)
        self.declare_parameter('obs_geom_gp', 0)

        self.timer = None
        self.srv = None
        
        # Subscriber와 Publisher 생성
        self.mujoco_map_sub = self.create_subscription(String, '/mujoco_map', self.mujoco_map_callback, 10)
        self.map_pub = self.create_publisher(UInt8MultiArray, '/converted_map', 10)
        
        self.planner_client = self.create_client(Trigger, '/plan_path')
        
        self.get_logger().info(f"Mujoco Map Converter Node activated.")

    async def mujoco_map_callback(self, msg):
        command = msg.data.strip()
        self.get_logger().info(f"Converting mujoco map: {command}")
        
        # 파라미터 값 읽기
        env_folder = self.get_parameter('mujoco_env_folder').value
        
        # 폴더 내부의 xml 파일 정렬 리스트 확보 (all 명령용)
        all_maps = [f for f in os.listdir(env_folder) if f.endswith('.xml')]
        all_maps = sorted(all_maps)

        # --- 모든 map을 처리하는 경우 ---
        if command.lower() == 'all':
            for i, map_name in enumerate(all_maps):
                self.get_logger().info(f"Converting map {i+1}: {map_name}...")
                xml_path = os.path.join(env_folder, map_name)
                
                # 맵 변환 및 토픽 발행
                self.convert_and_publish(xml_path)
                
                # Path planner가 이 맵으로 path planning을 다 끝낼 때까지 일시정지하고 대기
                success = await self.wait_for_planner_complete()
                
                if success:
                    self.get_logger().info(f"Completed path planning for {map_name}.\n")
                else:
                    self.get_logger().error(f"Failed to complete path planning for {map_name}.\n")
                    
            self.get_logger().info("All map path planning completed.")

        # --- 하나의 단일 .xml 맵만 처리하는 경우 ---
        elif command.endswith('.xml'):
            xml_path = os.path.join(env_folder, command)
            if os.path.exists(xml_path):
                self.get_logger().info(f"Converting the map: {command}...")
                self.convert_and_publish(xml_path)
                
                await self.wait_for_planner_complete()
            else:
                self.get_logger().error(f"The map {xml_path} does not exist.")
                
        else:
            self.get_logger().warning(f"Invalid command format: '{command}'. Please use 'all' or 'xxx.xml'.")
            return

    def convert_and_publish(self, xml_path):
        """순수 파이썬 알고리즘을 태우고 결과를 UInt8MultiArray 규격으로 쏘는 핵심 함수"""
        try:
            grid_size_cm = self.get_parameter('grid_size_cm').value
            robot_h = self.get_parameter('robot_h').value
            obs_geom_gp = self.get_parameter('obs_geom_gp').value
            
            converter = MujocoGridConverter(
                xml_path=xml_path,
                grid_size=grid_size_cm,
                robot_height=robot_h,
                obstacle_geom_group=obs_geom_gp
            )
            converter.convert_mjc2grid()
            numpy_grid = converter.obs_grid_map
              
            if numpy_grid is None:
                self.get_logger().error(f"Failed to convert map: {xml_path}")
                return

            numpy_grid = numpy_grid.astype(np.uint8)
            height, width = numpy_grid.shape

            # UInt8MultiArray message 규격으로 맞춤 생성
            msg = UInt8MultiArray()
            dim_width = MultiArrayDimension(label="width", size=width, stride=width)
            dim_height = MultiArrayDimension(label="height", size=height, stride=width * height)
            msg.layout.dim = [dim_height, dim_width]
            msg.data = numpy_grid.flatten().tolist()

            self.map_pub.publish(msg)
            self.get_logger().info(f"Published the converted map: {os.path.basename(xml_path)}")

        except Exception as e:
            self.get_logger().error(f"Failed to publish converted map: {str(e)}")

    async def wait_for_planner_complete(self):
        """패스 플래너에게 전화를 걸어 작업이 완전히 끝날 때까지 비동기로 대기하는 헬퍼 함수"""
        # 플래너 서비스가 살아있는지 검사
        if not self.planner_client.service_is_ready():
            self.get_logger().warn("[WAIT] Path planner service is not ready. Waiting for connection...")
            if not self.planner_client.wait_for_service(timeout_sec=2.0):
                self.get_logger().error("[ERROR] Failed to connect to path planner. Skipping wait.")
                return False

        req = Trigger.Request()
        future = self.planner_client.call_async(req)
        
        try:
            response = await future
            return response.success
        except Exception as e:
            self.get_logger().error(f"[ERROR] Path planner completion signal reception error: {e}")
            return False

def main(args=None):
    rclpy.init(args=args)
    node = MujocoMapConverterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()