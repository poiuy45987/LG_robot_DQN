import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
from collections import deque
import random

class CoverageEnv(gym.Env):
    def __init__(self, **kwargs):
        super(CoverageEnv, self).__init__()
        
        # Parameter 기본값 설정
        self.robot_size = kwargs.get('robot_size', 3) # 로봇 크기 (robot_size x robot_size)
        
        # Map 생성을 위한 parameters
        self.map_row = kwargs.get('row_size', 100) # Map의 행 크기
        self.map_col = kwargs.get('col_size', 100) # Map의 열 크기
        self.large_obs_num = kwargs.get('large_obs_num', 5) # 큰 장애물 개수
        self.large_obs_max_size = kwargs.get('large_obs_max_size', 15) # 큰 장애물 최대 크기
        self.large_obs_min_size = kwargs.get('large_obs_min_size', 6) # 큰 장애물 최소 크기
        self.small_obs_num = kwargs.get('small_obs_num', 20) # 작은 장애물 개수
        self.small_obs_max_size = kwargs.get('small_obs_max_size', 2) # 작은 장애물 최대 크기
        self.small_obs_min_size = kwargs.get('small_obs_min_size', 1) # 작은 장애물 최소 크기
        print(kwargs['row_size'], kwargs['col_size'])
        
        # Robot 초기 위치
        self.start_row = 1; self.start_col = 1

        # Action Space: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = spaces.Discrete(4)
        
        # Observation Space: (3, grid_size, grid_size)
        # Channel 0: 로봇 위치, Channel 1: 방문 기록, Channel 2: 장애물
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3, self.map_row, self.map_col), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Map의 장애물 위치를 표시하는 layer
        self.map = self.random_map_generate()
        
        # 어느 grid를 다녀갔는지 표시하는 layer
        self.visited = self.map.copy()
        self.visited[self.start_row, self.start_col] = 1.0
        
        # Agent의 위치를 표시하는 layer
        self.agent_pos = [self.start_row, self.start_col]  # Agent 위치
        agent_layer = np.zeros((self.map_row, self.map_col), dtype=np.float32)
        agent_layer[tuple(self.agent_pos)] = 1.0        
        
        # Path를 저장하기 위한 list
        self.path = []
        self.path.append(tuple(self.agent_pos)) # 시작 위치를 path에 추가
        
        obs = np.stack([agent_layer, self.visited, self.map], axis=0)
        
        return obs, {}
    
    def clean_unreachable_area(self, map_data, start_pos):
        """
        start_pos: (x, y) 튜플
        반환값: 시작점으로부터 연결된 빈 공간(0.0)의 총 개수
        """
        start_x, start_y = start_pos
        
        # 시작점이 이미 장애물인 경우 0 반환: map을 다시 생성하도록 함.
        if map_data[start_y, start_x] == 1.0:
            raise ValueError("Starting position is on an obstacle.")

        rows, cols = map_data.shape
        queue = deque([(start_x, start_y)])
        visited = np.zeros_like(map_data, dtype=bool)
        count = 0

        while queue:
            curr_x, curr_y = queue.popleft()
            count += 1

            # 상하좌우 인접한 칸 확인
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = curr_x + dx, curr_y + dy

                # 맵 범위 안이고, 방문하지 않았으며, 장애물(1.0)이 아닌 경우
                if 0 <= nx < cols and 0 <= ny < rows:
                    if not visited[ny, nx] and map_data[ny, nx] == 0.0:
                        visited[ny, nx] = True
                        queue.append((nx, ny))
                        
        # 2. 도달 불가능한 영역을 장애물(1.0)로 설정
        cleaned_map_data = map_data.copy()
        cleaned_map_data[~visited] = 1.0  # 도달 불가능한 영역을 장애물로 설정
        
        return count, cleaned_map_data
    
    def random_map_generate(self):
        
        map_data = np.zeros((self.map_row, self.map_col), dtype=bool)
        
        # 벽 생성
        map_data[0, :] = True  # 상단 벽
        map_data[-1, :] = True  # 하단 벽
        map_data[:, 0] = True  # 좌측 벽
        map_data[:, -1] = True  # 우측 벽

        # 적절한 map이 완성될 때까지 큰 장애물 배치를 반복
        print("Generating the map...")
        max_trial = 10
        for trial in range(1, max_trial+1):
            
            # 큰 장애물 생성(가구 등 묘사) 배치
            for _ in range(self.large_obs_num):
                
                # 장애물 크기(w, h)와 위치(x, y) 랜덤 결정
                w = random.randint(self.large_obs_min_size, self.large_obs_max_size+1)
                h = random.randint(self.large_obs_min_size, self.large_obs_max_size+1)
                x = random.randint(1, self.map_col - w - 1)
                y = random.randint(1, self.map_row - h - 1)
                map_data[y:y+h, x:x+w] = True
            
            map_data[self.start_col, self.start_row] = False # 시작 위치는 항상 비워둠
            
            # 도달할 수 없는 map 영역을 정리
            count, cleaned_map_data = self.clean_unreachable_area(map_data, (self.start_row, self.start_col))
            
            # 도달 가능 영역이 너무 적으면 장애물을 다시 배치
            if count >= (self.map_row - 2) * (self.map_col - 2) * 0.5:
                map_data = cleaned_map_data
                print("Complete the organizing large obstacles!")
                break
            else:
                if trial < max_trial:
                    print(f"Trial {trial}: Failed the organinzing large obstacles. Regenerating the map...")
                    map_data[1:-1, 1:-1] = False # 내부를 다시 비워둠
                else:
                    print(f"Trial {trial}: Failed the organinzing large obstacles. Stop generating the map...")         
                
        
        # 고밀도 장애물 환경을 다루기 위한 작은 장애물 생성
        # 작은 장애물은 벽과 다른 장애물과 맞닿지 않도록 설정
        max_trial = 100
        for _ in range(self.small_obs_num):
            
            for trial in range(1, max_trial+1):
                # 장애물 크기(w, h)와 위치(x, y) 랜덤 결정
                w = random.randint(self.small_obs_min_size, self.small_obs_max_size+1)
                h = random.randint(self.small_obs_min_size, self.small_obs_max_size+1)
                x = random.randint(1, self.map_col - w - 1)
                y = random.randint(1, self.map_row - h - 1)
                
                # 장애물을 배치하려는 곳과 이웃한 픽셀에 장애물이 없어야 함.
                if not np.any(map_data[y-1:y+h+1, x-1:x+w+1]):
                    map_data[y:y+h, x:x+w] = True
                    break
                elif trial == max_trial:
                    print(f"Failed to place small obstacle. Skipping this obstacle.")
                    
        print("Complete the organizing small obstacles!")

        return map_data
    
    def show_map(self):
        
        fig_row_size = 8; fig_col_size = fig_row_size*(self.map_col/self.map_row)
        plt.figure(figsize=(fig_col_size, fig_row_size))
        
        # cmap='gray_r'은 0(빈공간)을 하얗게, 1(장애물)을 검게 보여줍니다.
        plt.imshow(self.map, cmap='gray_r', origin='lower')
        
        # 시작 위치 표시 (빨간 별)
        plt.plot(self.start_row, self.start_col, 'r*', markersize=15, label='Start Pos')
        
        plt.title("Grid Map Generation Check")
        plt.grid(True, which='both', color='lightgray', linewidth=0.5)
        plt.legend()
        plt.show()

    def _get_obs(self):
        # 3채널 상태 행렬 생성
        agent_layer = np.zeros((self.map_row, self.map_col), dtype=bool)
        agent_layer[tuple(self.agent_pos)] = True
        
        return np.stack([agent_layer, self.visited, self.map], axis=0)

    def step(self, action):
        
        # 이동 로직
        prev_pos = self.agent_pos.copy()
        if action == 0:   # Up
            next_agent_pos = [max(0, self.agent_pos[0] - 1), self.agent_pos[1]]
        elif action == 1: # Down
            next_agent_pos = [min(self.map_row - 1, self.agent_pos[0] + 1), self.agent_pos[1]]
        elif action == 2: # Left
            next_agent_pos = [self.agent_pos[0], max(0, self.agent_pos[1] - 1)]
        elif action == 3: # Right
            next_agent_pos = [self.agent_pos[0], min(self.map_col - 1, self.agent_pos[1] + 1)]
        
        # 다음 목적지에 장애물이 없는 경우에만 이동
        if not self.map[tuple(next_agent_pos)]:
            self.agent_pos = next_agent_pos

        # 보상 설계
        reward = 0
        terminated = False
        
        # 1. 벽에 부딪혔는지 확인
        if np.array_equal(prev_pos, self.agent_pos):
            reward = -0.5  # 패널티
        else:
            # 2. 새로운 곳을 방문했는지 확인
            if not self.visited[tuple(self.agent_pos)]:
                reward = 1.0  # 긍정 보상
                self.visited[tuple(self.agent_pos)] = True
            else:
                reward = -0.1 # 이미 가본 곳 재방문 패널티
        
        self.path.append(tuple(self.agent_pos))

        # 종료 조건: 모든 셀 방문 완료 시
        if np.all(self.visited):
            reward = 2.0
            terminated = True
        
        # 너무 오래 걸릴 경우 대비 (학습 효율)
        truncated = len(self.path) > self.map_col * self.map_row * 2
        #truncated = len(self.path) > 1000

        return self._get_obs(), reward, terminated, truncated, {"path": self.path}
    
def test_code():
    # 환경 설정 값
    config = {
        'row_size': 100,
        'col_size': 100,
        'large_obs_num': 20,
        'large_obs_max_size': 20,
        'large_obs_min_size': 10,
        'small_obs_num': 30,
        'small_obs_max_size': 3,
        'small_obs_min_size': 1,
        'robot_size': 3
    }
    
    # 환경 인스턴스 생성
    env = GridCoverageEnv(**config)
    
    # 맵 생성 (reset 호출 시 random_map_generate가 실행됨)
    # 현재 클래스 구조상 reset()에서 맵을 생성하므로 reset을 먼저 호출합니다.
    # 단, 현재 코드에 변수명 오타가 있을 수 있어 직접 함수를 호출해 보겠습니다.
    
    env.reset()
    env.show_map()

if __name__ == '__main__':
    test_code()