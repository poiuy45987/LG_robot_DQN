import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray
from std_srvs.srv import Trigger
import time # 시뮬레이션을 위한 임시 임포트

class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')
        
        # 1. 맵 컨버터가 쏴주는 맵 토픽을 받아두기 (필요한 경우)
        self.map_sub = self.create_subscription(UInt8MultiArray, '/converted_map', self.map_callback, 10)
        
        # 🌟 2. 맵 컨버터가 전화를 걸어올 서비스 서버 오픈! (이름: /plan_path)
        self.plan_srv = self.create_service(Trigger, '/plan_path', self.plan_path_callback)
        
        self.latest_grid_map = None
        self.get_logger().info("Path Planner Node activated and waiting for map/service...")

    def map_callback(self, msg):
        """맵 컨버터가 변환해서 쏴준 2D 그리드 맵을 보관하는 콜백"""
        # 여기서 msg.layout과 msg.data를 파싱해서 사용하면 됩니다.
        self.latest_grid_map = msg
        self.get_logger().info("New map received from converter.")

    def plan_path_callback(self, request, response):
        """🌟 맵 컨버터가 await로 답장을 기다리게 만드는 핵심 서비스 콜백"""
        self.get_logger().info("Received path planning request from map converter...")
        
        try:
            # -------------------------------------------------------------
            # 🧠 [여기에 질문자님의 실제 패스 플래닝 알고리즘이 들어갑니다]
            # 예시로 1.5초 동안 무거운 계산(A*, Dijkstra, RL 등)을 한다고 가정해봅시다.
            self.get_logger().info("Calculating optimal path...")
            time.sleep(1.5) 
            # -------------------------------------------------------------
            
            # 계산이 성공적으로 끝났다면 영수증에 도장을 쾅 찍어줍니다.
            response.success = True
            response.message = "Path planning successfully completed!"
            self.get_logger().info("Path planning completed. Sending response to converter.")
            
        except Exception as e:
            response.success = False
            response.message = f"Failed to plan path: {str(e)}"
            self.get_logger().error(response.message)
            
        # 이 response가 리턴되는 순간 랜선을 타고 맵 컨버터의 await future가 풀립니다!
        return response

def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()