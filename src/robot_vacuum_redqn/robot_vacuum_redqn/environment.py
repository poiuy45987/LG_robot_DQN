import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_agg import FigureCanvasAgg
from IPython.display import display, clear_output
import cv2
import io
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import argparse

from config import EnvConfig, DEFAULT_SEED
from map_generator import generate_house_like_obstacles


def get_args():
    
    parser = argparse.ArgumentParser(
        description="""
        Environment에서 구현된 요소 중에 확인하고 싶은 것을 선택
        --robot: 로봇이 모델링된 모습을 보고싶은 경우
        --see_map 또는 --see_obs를 입력하여 시각화 대상을 선택
        --debug_reset 또는 --debug_step을 입력하여 테스트할 method를 선택
        ex) --robot --see_map --debug_reset
    """)
    
    parser.add_argument("--robot", action="store_true", help="로봇이 모델링된 모습을 시각화")
    parser.add_argument("--see_map", action="store_true", help="Map layer를 시각화")
    parser.add_argument("--see_obs", action="store_true", help="Observation을 시각화")
    parser.add_argument("--debug_reset", action="store_true", help="reset() method를 테스트")
    parser.add_argument("--debug_step", action="store_true", help="step() method를 테스트")

    return parser.parse_args()

def make_robot_mask(robot_size: int) -> np.ndarray:
    # 로봇 청소기 모양을 원형으로 만듦
    # robot_size: 로봇 청소기의 지름
    
    center = robot_size / 2.0; radius = robot_size / 2.0
    grid = np.arange(robot_size) + 0.5 - center
    x, y = np.meshgrid(grid, grid, sparse=True)
    mask = (x**2 + y**2 <= radius**2).astype(np.uint8)
    
    return mask

def visualize_mask(robot_size: int):
    mask = make_robot_mask(robot_size)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray_r') # 0은 검정, 1은 흰색으로 표시
    plt.title(f"Robot Mask (Size: {robot_size}x{robot_size})")
    plt.colorbar(label='Mask Value')
    plt.show()
    
class DQNCoverageEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig, seed: int | None = None):
        super().__init__()
        
        # ---- 1. 고정 설정 및 난수 생성기 ----
        self.cfg = cfg
        self._seed = seed if seed is not None else DEFAULT_SEED
        self.env_rng = np.random.default_rng(self._seed)
        self.map_rng = np.random.default_rng(self._seed)
        # ----------------------------------

        # ---- 2. 로봇 모양 정의 ----
        self.robot_size = cfg.robot_size
        self.robot_half_size = int(self.robot_size // 2)
        self.robot_mask = make_robot_mask(self.robot_size)
        
        y_idx, x_idx = np.nonzero(self.robot_mask)
        self.robot_mask_offsets = np.column_stack([
            x_idx - self.robot_half_size, 
            y_idx - self.robot_half_size
        ]).astype(np.int32)
        
        self.robot_area = int(self.robot_mask.sum())
        # -------------------------
        
        # ---- 3. Observation_space, Action_space ----
        self.local_view = self.cfg.local_view
        self.observation_space = spaces.Dict({
            "map": spaces.Box(low=0, high=1, shape=(3, self.local_view, self.local_view), dtype=np.uint8),
            "vec": spaces.Box(low=-1.0, high=1.0, shape=(2+4+4+1,), dtype=np.float32), # 2+4+4+1
        })
        self.action_space = spaces.Discrete(4, seed=self._seed)
        # --------------------------------------------
        
        # ---- 4. Map 관련 변수 ----
        self.H = cfg.H; self.W = cfg.W
        self.obstacles = None     # 장애물의 위치를 표시하는 layer: uint8 [H,W] (장애물: 1, 빈 공간: 0)
        self.cleaned = None       # 로봇이 청소한 grid를 표시하는 layer: uint8 [H,W] (Cleaned: 1, Uncleaned: 0)
        self.collision_map = None # Obstacle dilated map: uint8 [H,W] (Dilated obstacles: 1, 빈 공간: 0)
        self.reachable = None     # Reachable robot centers: uint8 [H,W] (Reachable center: 1, Unreachable center: 0)
        self.coverable = None     # Coverable cells: uint8 [H,W] (Coverable grid: 1, Uncoverable grid: 0)
        # ------------------------
        
        # ---- 5. Coverage 관련 변수 ----
        self.total_coverable_area = 0   # Coverable 영역을 계산
        self.last_coverage = 0.0        # 이전 step에서의 coverage 비율(Step 시도 시 coverate가 증가하는지 여부를 관찰하기 위함)
        self.no_progress_cnt = 0        # 이전 step에 비해 coverage가 증가하지 않은 횟수를 세는 변수. 그 횟수가 일정 이상이 되면 truncated
        # ------------------------------
        
        # ---- 6. 로봇의 현재 위치와 방향 ----
        self.pos = None        # 로봇 중심의 현재 위치 (x, y)
        self.dir = None        # 로봇이 바라보는 방향(이전에 진행한 방향) (0: E, 1: N, 2: W, 3: S)
        self.dir_vecs = [
            ( 1,  0),  # E
            ( 0,  1),  # N
            (-1,  0),  # W
            ( 0, -1),  # S
        ]
        self.traj = []         # 로봇 중심이 한 episode에서 이동한 경로
        # --------------------------------
        
        # ---- 7. 기타 사용하기 좋은 지표들 ----
        self.steps = 0                      # 로봇의 step 수
        self.max_steps = cfg.max_steps      # Episode 당 max_step 수
        self.collision_count = 0            # 한 Episode 당 충돌한 횟수
        self.warmup = False                 # Warmup 여부
        self.warmup_steps = 0               # Warmup 시 episode 당 max_step 수
        self.max_forward = cfg.max_forward  # 각 방향으로의 여유 grid 수의 최댓값
        # -----------------------------------
        
        # ---- 8. Visualization을 위한 figure과 canvas 설정 ----
        self.fig = Figure()
        self.canvas = FigureCanvasAgg(self.fig)
        # ----------------------------------------------------

    def set_env_mode(self, warmup: bool = False, ep_steps: int = 2000):
        self.warmup = warmup
        self.warmup_steps = ep_steps
    
    def _in_bounds_center(self, cx: int, cy: int) -> bool:
        return (self.robot_half_size <= cx < self.W - self.robot_half_size) and (self.robot_half_size <= cy < self.H - self.robot_half_size)

    # 중심 좌표를 받아서 로봇이 차지하는 영역(footprint) 좌표를 반환
    def _get_footprint_coords(self, cx: int, cy: int) -> tuple[np.ndarray, np.ndarray]:
        xs = cx + self.robot_mask_offsets[:, 0]
        ys = cy + self.robot_mask_offsets[:, 1]
        return xs, ys
    
    def _collides(self, *args) -> bool:
        """
        Case 1: _collides(cx, cy) -> 중심 좌표로 경계 및 장애물 체크
        Case 2: _collides(xs, ys) -> 이미 계산된 마스크 좌표 배열들로 장애물 체크
        """
        assert len(args) == 2
        
        arg1, arg2 = args
        if np.isscalar(arg1) and np.isscalar(arg2):
            cx, cy = int(arg1), int(arg2)
            if self.collision_map is not None:
                if not (0 <= cx < self.W and 0 <= cy < self.H):
                    return True
                return bool(self.collision_map[cy, cx])
            else:
                if not self._in_bounds_center(cx, cy):
                    return True
                xs, ys = self._get_footprint_coords(cx, cy)
        elif isinstance(arg1, np.ndarray) and isinstance(arg2, np.ndarray):
            xs, ys = arg1, arg2
        else:
            raise TypeError("Input must be tuple of scalar or tuple of np.ndarray")
            
        return bool((self.obstacles[ys, xs] == 1).any())


    def _compute_reachable_centers(self, start_cx: int, start_cy: int) -> np.ndarray:

        reachable = np.zeros((self.H, self.W), dtype=np.uint8)

        q = deque()
        reachable[start_cy, start_cx] = 1
        q.append((start_cx, start_cy))

        while q:
            x, y = q.popleft()
            for dx, dy in self.dir_vecs:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.W and 0 <= ny < self.H):
                    continue
                if reachable[ny, nx]:
                    continue
                if self._collides(nx, ny):
                    continue
                reachable[ny, nx] = 1
                q.append((nx, ny))

        return reachable


    def _compute_coverable_cells_from_reachable(self, reachable: np.ndarray) -> np.ndarray:

        coverable = np.zeros((self.H, self.W), dtype=np.uint8)

        # OR-shift reachable by footprint offsets -> union of all swept cells
        for dx, dy in self.robot_mask_offsets:
            dx = int(dx); dy = int(dy)

            src_y0 = max(0, -dy)
            src_y1 = self.H - max(0, dy)
            src_x0 = max(0, -dx)
            src_x1 = self.W - max(0, dx)

            dst_y0 = max(0, dy)
            dst_y1 = self.H - max(0, -dy)
            dst_x0 = max(0, dx)
            dst_x1 = self.W - max(0, -dx)

            coverable[dst_y0:dst_y1, dst_x0:dst_x1] |= reachable[src_y0:src_y1, src_x0:src_x1]

        # never count obstacle cells
        coverable &= (self.obstacles == 0).astype(np.uint8)
        return coverable

    # FIXME: Reward 구조 수정
    def _apply_footprint_rewards(self, cx: int, cy: int):
        
        if not self._in_bounds_center(cx, cy):
            return float(self.cfg.obstacle_penalty), True
        
        xs, ys = self._get_footprint_coords(cx, cy)
        if self._collides(xs, ys):
            return float(self.cfg.obstacle_penalty), True
        else:
            self.cleaned[ys, xs] = 1
            if self.cleaned[cy, cx]:
                return float(self.cfg.cleaned_penalty), False
            else:
                return float(self.cfg.uncleaned_reward), False

        # if (self.obstacles[ys, xs] == 1).any():
        #     return float(self.cfg.obstacle_penalty), True

        # cleaned_vals = self.cleaned[ys, xs]

        # # reward only for coverable cells
        # if self.coverable is None:
        #     mask = np.ones_like(cleaned_vals, dtype=np.uint8)
        # else:
        #     mask = (self.coverable[ys, xs] == 1)

        # newly = ((cleaned_vals == 0) & mask).astype(bool)
        # already = ((cleaned_vals == 1) & mask).astype(bool)

        # reward = newly.sum() * self.cfg.uncleaned_reward + already.sum() * self.cfg.cleaned_penalty
        # self.cleaned[ys[newly], xs[newly]] = 1
        
        # return float(reward), False


    def _coverage(self) -> float:
        if self.coverable is None:
            free = (self.obstacles == 0)
            total = int(free.sum())
            if total == 0:
                return 0.0
            cleaned_free = int((self.cleaned & free).sum())
            return cleaned_free / total

        if self.total_coverable_area == 0:
            return 0.0
        
        cleaned_coverable = int((self.cleaned & self.coverable).sum())
        return cleaned_coverable / self.total_coverable_area


    def _ray_distance_forward(self, cx: int, cy: int, d: int) -> int:
        dx, dy = self.dir_vecs[d]
        ex = cx + dx*self.max_forward; ex = int(max(0, min(self.W-1, ex)))
        ey = cy + dy*self.max_forward; ey = int(max(0, min(self.H-1, ey)))
        
        if dx != 0: # 가로 방향으로 이동하는 경우
            step = 1 if dx > 0 else -1
            line = self.collision_map[cy, cx+step:ex+step:step]
        else: # 세로 방향으로 이동하는 경우
            step = 1 if dy > 0 else -1
            line = self.collision_map[cy+step:ey+step:step, cx]
            
        obstacles = np.where(line == 1)[0]
        if len(obstacles) > 0:
            return int(obstacles[0]) # 첫 번째 장애물까지의 거리
        else:
            return len(line) # 장애물이 없으면 최대 거리 반환

    def _crop_patch(self, arr: np.ndarray, cx: int, cy: int, value: int | list | tuple = 0) -> np.ndarray:
        
        r = self.local_view//2
        px, py = cx+r, cy+r
        
        if arr.ndim == 2:
            assert isinstance(value, int)
            padded = np.pad(arr, pad_width=r, mode="constant", constant_values=value)
            patch = padded[py-r:py+r+1, px-r:px+r+1]
            
        elif arr.ndim == 3:
            
            num_channels = arr.shape[0]
            if isinstance(value, (list, tuple)):
                if len(value) != num_channels:
                    raise ValueError(f"Length of value ({len(value)}) must match num_channels ({num_channels})")
                pad_vals = value
            else:
                pad_vals = [value] * num_channels
                
            padded_channels = [
                np.pad(arr[i], pad_width=r, mode="constant", constant_values=pad_vals[i])
                for i in range(num_channels)
            ]
            final_padded = np.stack(padded_channels, axis=0)
            patch = final_padded[:, py-r:py+r+1, px-r:px+r+1]
            
        else:
            raise ValueError(f"Invalid observation dimension: {arr.ndim}. Expected 2 or 3.")
        
        return patch

    
    def _get_agent_layer(self, cx: int, cy: int):
        
        assert not self._collides(cx, cy)
        
        self.H, self.W = self.obstacles.shape
        agent_layer = np.zeros((self.H, self.W), dtype=np.uint8)
        
        x0, x1 = int(cx-self.robot_half_size), int(cx+self.robot_half_size+1)
        y0, y1 = int(cy-self.robot_half_size), int(cy+self.robot_half_size+1)
        
        agent_layer[y0:y1, x0:x1] = self.robot_mask
        
        return agent_layer
        
    
    def _get_obs(self):
        
        cx, cy = self.pos
        # 로봇의 위치가 표시된 agent_layer 생성 후 local_map 정보를 담음
        agent_layer = self._get_agent_layer(cx, cy)
        full_layer = np.stack([agent_layer, self.cleaned, self.obstacles], axis=0)
        patch = self._crop_patch(full_layer, cx, cy, value=[0, 0, 1])
        
        # 학습에 도움이 될 만한 수치적인 정보를 생성
        # 로봇의 위치 정보: [-1, 1]의 범위로 정규화
        x_norm = (cx/(self.W-1))*2-1
        y_norm = (cy/(self.H-1))*2-1

        # dir_onehot
        dir_onehot = np.zeros(4, dtype=np.float32)
        dir_onehot[self.dir] = 1.0

        # 4방향에서 바라본 여유 공간: [-1, 1]이 범위로 정규화
        ray_norm = np.zeros(4, dtype=np.float32)
        for d in range(4):
            ray_norm[d] = self._ray_distance_forward(cx, cy, d)
        ray_norm = (ray_norm / max(1, self.max_forward))*2-1

        # Coverage 비율: [0, 1] 범위의 숫자를 [-1, 1] 범위로 정규화
        cov = self._coverage()
        cov_norm = cov*2-1

        obs_vec = np.concatenate([
            np.array([x_norm, y_norm], dtype=np.float32),
            dir_onehot,
            ray_norm,
            np.array([cov_norm], dtype=np.float32),
        ], axis=0).astype(np.float32)
        
        return {"map": patch, "vec": obs_vec}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed) # 난수 생성기 self.np_random가 생성됨. 첫 episode에는 seed 초기화가 일어남. 다음 episode부터는 seed 초기화를 수행하지 않음.(seed=None)

        # Map 생성
        # 시작 지점만 reset하고 map은 그대로 사용하는 경우에는 map을 다시 생성하지 않음
        if self.obstacles is None or options is None or not options['reset_only_start_pos']: # 아직 장애물이 아예 생성되지 않은 경우 or start_pos만 reset하는 경우
            self.obstacles = generate_house_like_obstacles(self.cfg, self.map_rng) # 장애물의 위치를 표시하는 layer
            # 장애물을 dilation해서 collision_map 생성
            self.collision_map = cv2.dilate(
                self.obstacles, 
                self.robot_mask, 
                borderType=cv2.BORDER_CONSTANT, 
                borderValue=1
            )

        # 청소된 grid를 표시하는 layer
        self.cleaned = np.zeros((self.H, self.W), dtype=np.uint8) # 장애물도 청소한 grid로 간주

        # 출발 위치와 방향 선정: 동시에 self.cleaned에서 로봇이 차지하는 영역을 변환
        for _ in range(20000):
            cx = int(self.env_rng.integers(self.robot_half_size, self.W - self.robot_half_size))
            cy = int(self.env_rng.integers(self.robot_half_size, self.H - self.robot_half_size))
            _, collided = self._apply_footprint_rewards(cx, cy)
            if not collided:
                self.pos = (cx, cy)
                break
        else:
            print("Force the robot position to the center.")
            xs, ys = self._get_footprint_coords(self.W//2, self.H//2)
            self.obstacles[ys, xs] = 0
            _, collided = self._apply_footprint_rewards(cx, cy)
            if collided:
                raise ValueError("Obstacles were not removed before forced robot placement.")
        self.dir = int(self.env_rng.integers(0, 4))
        
        # Coverable grid를 계산 (Training 시에도 이용)
        self.reachable = self._compute_reachable_centers(*self.pos)
        self.coverable = self._compute_coverable_cells_from_reachable(self.reachable)
        self.total_coverable_area = int(self.coverable.sum())
        self.last_coverage = self._coverage()
        self.no_progress_cnt = 0
        
        self.steps = 0
        self.traj = [self.pos]
        self.collision_count = 0

        return self._get_obs(), {"Start_pos": self.pos}

    def step(self, action):

        self.steps += 1
        self.dir = action

        # 초기 설정 값
        reward = 0.0
        terminated = False
        truncated = False
        collision = False
        success = False

        # ---- 다음 위치로 이동: Collision이 일어나면 더 나아가지 않음 ----
        # Collision이 일어나면, collision 사실을 info에 넣어 알리기만 하고 더 나아가지 않음
        # 이 작업은 training 시와 test 시에 동일하게 동작함.
        dx, dy = self.dir_vecs[self.dir]
        cx, cy = self.pos
        nx, ny = int(cx + dx), int(cy + dy)
        
        reward, collided = self._apply_footprint_rewards(nx, ny) # 다음 위치로 이동할 때의 reward와 충돌 여부를 얻음
        
        if collided:
            collision = True
            self.collision_count += 1
            reward = self.cfg.obstacle_penalty
            nx, ny = cx, cy # 충돌이 일어나면 로봇을 움직이지 않음
        cx, cy = nx, ny
        self.pos = (cx, cy)
        self.traj.append(self.pos) # 충돌이 일어났더라도 self.traj에 좌표를 저장
        # ---------------------------------------------------------
        
        # ---- Terminate과 Truncated 조건 ----
        # Terminate: Coverage를 성공한 경우 & Collision이 일어난 경우
        # Truncate: 전체 step 수가 max_steps를 넘은 경우 & Coverage가 증가하지 않는 상태로 일정 이상의 step을 수행한 경우
        cur_coverage = self._coverage()
        # Coverage가 더 이상 증가하지 않은 step 수를 셈
        if cur_coverage > self.last_coverage:
            self.no_progress_cnt = 0
        else:
            self.no_progress_cnt += 1
        
        if collision:
            terminated = True
        elif cur_coverage >= self.cfg.target_coverage:
            terminated = True
            success = True    
        elif cur_coverage >= self.cfg.final_coverage_thres and self.no_progress_cnt >= self.cfg.max_no_progress_steps_final:
            truncated = True
        elif self.no_progress_cnt >= self.cfg.max_no_progress_steps:
            truncated = True
                
        self.last_coverage = cur_coverage

        # Truncated 조건: max_step을 넘으면 종료 (Step 수는 warmup 여부에 따라 달라짐)
        if self.warmup:
            max_steps = self.warmup_steps
        else:
            max_steps = self.max_steps
            
        if self.steps >= max_steps:
            truncated = True

        info = {
            "Coverage": cur_coverage, 
            "Steps": self.steps, 
            "Collision": collision,
            "Success": success,
            "Episode_collision": self.collision_count,
        }
        # ----------------------------------

        return self._get_obs(), reward, terminated, truncated, info
    
    def _draw_layer(self):
        
        self.fig.clear() # 이전 그림 지우기
        self.fig.set_size_inches(8, 8)
        agent_layer = self._get_agent_layer(*self.pos)

        axes = self.fig.subplots(2, 3)

        # obstacles
        axes[0, 0].imshow(self.obstacles, cmap='gray_r', origin='lower')
        axes[0, 0].set_title("Original Obstacles")

        # collision_map (Dilation 결과)
        axes[1, 0].imshow(self.collision_map, cmap='gray_r', origin='lower')
        axes[1, 0].set_title("Collision Map (Dilation)")

        # cleaned
        # 현재 로봇 위치를 점으로 찍음
        axes[0, 1].imshow(self.cleaned, cmap='gray_r', origin='lower')
        cx, cy = self.pos
        axes[0, 1].plot(cx, cy, 'r.') # 로봇 위치를 빨간 점으로 표시
        axes[0, 1].set_title(f"Cleaned Area (Coverage: {self.last_coverage:.2%})")
        
        # agent_layer
        # 현재 로봇 위치를 점으로 찍음
        axes[1, 1].imshow(agent_layer, cmap='gray_r', origin='lower')
        cx, cy = self.pos
        axes[1, 1].plot(cx, cy, 'r.') # 로봇 위치를 빨간 점으로 표시
        axes[1, 1].set_title(f"Agent layer")
        
        # reachable
        axes[0, 2].imshow(self.reachable, cmap='gray_r', origin='lower')
        axes[0, 2].set_title(f"Reachable grid")
        
        # coverable
        axes[1, 2].imshow(self.coverable, cmap='gray', origin='lower')
        axes[1, 2].set_title(f"Coverable map")
        
        self.fig.tight_layout()
    
    def _draw_traj(self):
        
        self.fig.clear() # 이전 그림 지우기
        self.fig.set_size_inches(8, 8)
        agent_layer = self._get_agent_layer(*self.pos)
        
        ax = self.fig.add_subplot(1, 1, 1)
        
        # Map에서 시각화
        total_map = np.zeros_like(self.obstacles)
        total_map[self.coverable == 0] = 4 # Cover 불가능한 영역을 칠함
        total_map[self.obstacles == 1] = 1 # Obstacle 표시
        total_map[self.cleaned == 1] = 2   # Cleaned 영역 표시
        total_map[agent_layer == 1] = 3    # 현재 로봇 위치 표시
        custom_cmap = ListedColormap(['white', 'black', 'yellow', 'red', 'blue'])
        ax.imshow(total_map, cmap=custom_cmap, origin='lower', vmin=0, vmax=4)
        # 로봇이 움직인 궤적 표시
        if len(self.traj) > 1:
            traj_arr = np.array(self.traj)
            ax.plot(traj_arr[:, 0], traj_arr[:, 1], color='lime', linewidth=1.5, alpha=0.8, zorder=5)
            ax.plot(traj_arr[0, 0], traj_arr[0, 1], color='purple', marker='o', zorder=6)
        legend_elements = [
            Patch(facecolor='black', edgecolor='black', label='Obstacles'),
            Patch(facecolor='yellow', edgecolor='yellow', label='Cleaned region'),
            Patch(facecolor='red', edgecolor='red', label='Robot'),
            Patch(facecolor='blue', edgecolor='blue', label='Uncoverable region'),
            Line2D([0], [0], color='lime', lw=1.5, label='Trajectory'),
            Line2D([0], [0], marker='o', color='purple', label='Start_pos', markerfacecolor='purple', linestyle='None'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
        ax.set_title(f"Coverage: {self.last_coverage:.2f}")
        self.fig.tight_layout()
        
    def _draw_obs(self):
        
        self.fig.clear() # 이전 그림 지우기
        self.fig.set_size_inches(10, 8)
        
        # 그림을 그릴 창을 2행 3열로 나눔. 첫 번째 행의 높이는 두 번째 행보다 2배 높게 설정.
        gs = GridSpec(2, 3, figure=self.fig, height_ratios=[4, 1])
        
        # obs data를 얻음
        obs = self._get_obs()
        H, W = obs["map"][2].shape
        
        # ---- obs의 map data 그리기 ----
        
        # gs의 첫 번째 행을 그래프를 그리는 데 사용
        ax1 = self.fig.add_subplot(gs[0, 0])
        ax2 = self.fig.add_subplot(gs[0, 1])
        ax3 = self.fig.add_subplot(gs[0, 2])
        axes = [ax1, ax2, ax3]
            
        # obstacles
        axes[0].imshow(obs["map"][2], cmap='gray_r', origin='lower')
        axes[0].plot(H//2, W//2, 'r.')
        axes[0].set_title("Obstacles local view")
        legend_elements0 = [
            Patch(facecolor='black', edgecolor='black', label='Obstacles'),
            Patch(facecolor='white', edgecolor='black', label='Free space'),
        ]
        axes[0].legend(handles=legend_elements0, loc='upper left', bbox_to_anchor=(0, -0.1))

        # cleaned
        # 현재 로봇 위치를 점으로 찍음
        axes[1].imshow(obs["map"][1], cmap='gray_r', origin='lower')
        axes[1].plot(H//2, W//2, 'r.') # 로봇 위치를 빨간 점으로 표시
        axes[1].set_title("Cleaned map local view")
        legend_elements1 = [
            Patch(facecolor='black', edgecolor='black', label='Covered space'),
            Patch(facecolor='white', edgecolor='black', label='Uncovered space'),
        ]
        axes[1].legend(handles=legend_elements1, loc='upper left', bbox_to_anchor=(0, -0.1))
        
        # agent_layer
        # 현재 로봇 위치를 점으로 찍음
        axes[2].imshow(obs["map"][0], cmap='gray_r', origin='lower')
        axes[2].plot(H//2, W//2, 'r.') # 로봇 위치를 빨간 점으로 표시
        axes[2].set_title(f"Agent layer local view")
        legend_elements2 = [
            Patch(facecolor='black', edgecolor='black', label='Robot'),
            Patch(facecolor='white', edgecolor='black', label='Non-robot'),
        ]
        axes[2].legend(handles=legend_elements2, loc='upper left', bbox_to_anchor=(0, -0.1))
        # -----------------------------
        
        # ---- obs의 vec data를 text 형식으로 출력하기 ----
        
        # gs의 두 번째 행을 text 출력에 사용
        ax_txt = self.fig.add_subplot(gs[1, :])
        ax_txt.axis('off')
        
        # text 출력
        vec_info_text = (f"[Observation Status]\n"
        f"- Normlized position: ({obs["vec"][0]:.2f}, {obs["vec"][1]:.2f})\n"
        f"- Normalized ray: East: {obs["vec"][6]:.2f}, North: {obs["vec"][7]:.2f}, West: {obs["vec"][8]:.2f}, South: {obs["vec"][9]:.2f}\n"
        f"- Normalized coverage: {obs["vec"][10]:.2f}\n")
        ax_txt.text(0, 0.5, vec_info_text, transform=ax_txt.transAxes, fontsize=14,
                    va='top', ha='left', family='monospace')
        # ---------------------------------------------
        
        self.fig.tight_layout()
    
    
    def get_visualized_img(self, img_choice: str = 'traj') -> np.ndarray:

        if img_choice not in ['layer', 'traj', 'obs']:
            raise ValueError(f"Invalid img_choice: {img_choice}. Must be one of ['layer', 'traj', 'obs']")
           
        if img_choice == 'layer':
            self._draw_layer()
        elif img_choice == 'traj':
            self._draw_traj()
        else:
            self._draw_obs()
            
        self.canvas.draw()
        return np.array(self.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3] # [H, W, C]

    def show_visualized_img(self, img_choice: str = 'traj'):

        if img_choice not in ['layer', 'traj', 'obs']:
            raise ValueError(f"Invalid img_choice: {img_choice}. Must be one of ['layer', 'traj', 'obs']")
        
        img_array = self.get_visualized_img(img_choice)

        from PIL import Image
        img_pil = Image.fromarray(img_array)
        
        try:
            # 주피터 셀 환경일 때: 여러 번 호출하면 셀 아래에 그림이 순서대로 쭉 나열됩니다.
            from IPython.display import display
            display(img_pil)
        except ImportError:
            # 터미널 환경일 때: 시스템 이미지 뷰어로 창을 띄웁니다. 
            # 여러 번 호출하면 창이 여러 개 뜹니다.
            img_pil.show()
            

# def get_img_from_fig(fig, dpi=100):
#     """Figure를 렌더링하여 RGB 픽셀 배열(numpy)로 변환 (메모리 캡처)"""
#     buf = io.BytesIO()
#     fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
#     buf.seek(0)
    
#     # 메모리 버퍼를 넘파이 배열로 변환
#     img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
#     # OpenCV 등을 이용해 디코딩 (이미지 버퍼 -> RGB 행렬)
#     img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     plt.close(fig) # 캡처 후 즉시 메모리 해제
    
#     return img.transpose(2, 0, 1) # [H, W, C] -> [C, H, W] (텐서보드 규격)

        
# def visualize_obs(fig: Figure, obs: np.ndarray):
        
#     # 3개의 레이어를 합쳐서 시각화하거나 각각 따로 띄울 수 있습니다.
#     axes = fig.subplots(1, 3)
#     H, W = obs["map"][2].shape

#     # obstacles
#     axes[0].imshow(obs["map"][2], cmap='gray_r', origin='lower')
#     axes[0].plot(H//2, W//2, 'r.')
#     axes[0].set_title("Obstacles local view")
#     legend_elements0 = [
#         Patch(facecolor='black', edgecolor='black', label='Obstacles'),
#         Patch(facecolor='white', edgecolor='black', label='Free space'),
#     ]
#     axes[0].legend(handles=legend_elements0, loc='upper left', bbox_to_anchor=(0, -0.1))

#     # cleaned
#     # 현재 로봇 위치를 점으로 찍음
#     axes[1].imshow(obs["map"][1], cmap='gray_r', origin='lower')
#     axes[1].plot(H//2, W//2, 'r.') # 로봇 위치를 빨간 점으로 표시
#     axes[1].set_title("Cleaned map local view")
#     legend_elements1 = [
#         Patch(facecolor='black', edgecolor='black', label='Covered space'),
#         Patch(facecolor='white', edgecolor='black', label='Uncovered space'),
#     ]
#     axes[1].legend(handles=legend_elements1, loc='upper left', bbox_to_anchor=(0, -0.1))
    
#     # agent_layer
#     # 현재 로봇 위치를 점으로 찍음
#     axes[2].imshow(obs["map"][0], cmap='gray_r', origin='lower')
#     axes[2].plot(H//2, W//2, 'r.') # 로봇 위치를 빨간 점으로 표시
#     axes[2].set_title(f"Agent layer local view")
#     legend_elements2 = [
#         Patch(facecolor='black', edgecolor='black', label='Robot'),
#         Patch(facecolor='white', edgecolor='black', label='Non-robot'),
#     ]
#     axes[2].legend(handles=legend_elements2, loc='upper left', bbox_to_anchor=(0, -0.1))
    
#     print(f"""
#     [Observation Status]
#     - Normlized position: ({obs["vec"][0]:.2f}, {obs["vec"][1]:.2f})
#     - Normalized ray: East: {obs["vec"][6]:.2f}, North: {obs["vec"][7]:.2f}, West: {obs["vec"][8]:.2f}, South: {obs["vec"][9]:.2f}
#     - Normalized coverage: {obs["vec"][10]:.2f}
#     """)
    

if __name__ == "__main__":
    # environment가 잘 생성되었는지 테스트하는 코드
    args = get_args()
    cfg = EnvConfig()
    env = DQNCoverageEnv(cfg, seed=42)

    # 0. Robot visualize
    if args.robot:
        visualize_mask(robot_size=env.robot_size)
    
    # Debug reset() method
    obs, info = env.reset()
    if args.debug_reset:
        if args.see_map:
            # 1. Map visualize: Agent layer, Cleaned layer, Obstacle layer, Collision map, 
            env.show_visualized_img('layer')
            env.show_visualized_img('traj')
        if args.see_obs:
            # 2. observation visualization
            env.show_visualized_img('obs')

    
    # Debug step() method
    if args.debug_step:
        # action_seq = [0]*150 + [1]*3 + [3]*150 + [2]*250 + [3]*100 + [1]*200
        action_seq = [0]*1000 + [3]*1000 + [2]*1000 + [1] * 1000
        for action in action_seq:
            obs, reward, terminated, truncated, info = env.step(action)
        if args.see_map:
            # 1. Map visualize: Agent layer, Cleaned layer, Obstacle layer, Collision map, 
            env.show_visualized_img('layer')
            env.show_visualized_img('traj')
        if args.see_obs:
            # 2. observation visualization
            env.show_visualized_img('obs')