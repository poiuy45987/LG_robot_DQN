import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
import argparse
import math
from dataclasses import dataclass

from path_planner.config import EnvConfig, DEFAULT_SEED, ANG_SEG_NUM, CLEANED_MAP_MAX
from path_planner.map_layer import MapLayers, MapConfigSchema, BoundingBox
from path_planner.utils.visualizer import display_image, visualize_mask, draw_layer, draw_obs, draw_traj
from path_planner.utils.map_utils import float_to_int_coord, make_robot_mask
from path_planner.utils.trajectory_metrics import cleaning_time_minutes, overlap_percent

# Action 및 진행 가능 각도 종류 설정
# Action: 현재 로봇이 바라보는 방향과의 각도 차이를 의미. 0~90, -90~0를 각각 ACTION_SEG_NUM개의 구간으로 나눠서 얻은 2*ACTION_SEG_NUM+1가지의 각도 + 유턴
# Total_ang: 진행 가능한 각도 종류
ACTION_NUM = 2 * ANG_SEG_NUM + 2 # 전체 action 종류: 현재 로봇이 바라보는 방향과의 각도 차이로 action 구성.
ALL_DIR_NUM = 4 * ANG_SEG_NUM # 진행 가능한 모든 각도 종류

@dataclass
class EnvHistory:
    steps: int
    pos: tuple[float, float]
    collision_cnt: int
    direc: int
    no_progress_cnt: int
    last_coverage: int


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


class CoverageEnv(gym.Env):
    metadata = {"render_modes": []}
    
    def __init__(self, 
                 cfg: EnvConfig = EnvConfig(), 
                 env_rng: np.random.Generator = np.random.default_rng(DEFAULT_SEED), 
                 map_rng: np.random.Generator = np.random.default_rng(DEFAULT_SEED)):
        super().__init__()
        
        # ---- 1. Map 및 로봇 관련 설정 설정 및 난수 생성기 ----
        self.cfg = cfg  # 로봇 크기, map의 설정 등에 관한 정보가 담겨 있음. 
        self.env_rng = env_rng
        self.map_rng = map_rng
        self.np_random = self.env_rng
        # -------------------------------------------------
        
        # ---- 2. Observation_space, Action_space ----
        self.action_space: spaces.Discrete = spaces.Discrete(ACTION_NUM, seed=self.env_rng)
        self.observation_space: spaces.Dict = spaces.Dict({
            "map": spaces.Box(low=0.0, high=1.0, shape=(3*self.cfg.stack_steps, self.cfg.local_view, self.cfg.local_view), dtype=np.float32),
            "loc_vec": spaces.Box(low=-1.0, high=1.0, shape=(ACTION_NUM*2,), dtype=np.float32),
            "glob_vec": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(ACTION_NUM,), dtype=np.float32)
        })
        # --------------------------------------------

        # ---- 3. Map 관련 변수 ----
        self.H = self.cfg.init_H; self.W = self.cfg.init_W  # Map이 새로 reset될 때마다 초기화됨.
        self.map_layers = MapLayers(self.cfg.map_cfg, self.map_rng, self.cfg.robot_size)
        # ------------------------
        
        # ---- 4. 고정된 변수들 ----
        self.step_len: int = 1 # 한 번의 step으로 나아가는 길이
        # ------------------------
        
        # ---- 5. 매 step마다 변하는 변수들 ----
        self.steps: int = 0
        self.pos: tuple[float, float] | None = None
        self.dir: int | None = None
        self.collision_count: int = 0
        self.last_coverage: float = 0.0 # 이전 step에서의 coverage 비율(Step 시도 시 coverate가 증가하는지 여부를 관찰하기 위함)
        self.no_progress_cnt: int = 0  
        
        self.warmup: bool = False
        self.warmup_steps: int = 0
        self.done: bool = False
        
        self.env_history: list[EnvHistory] = []
        # ------------------------------
        
        # ---- 6. 방향 관련 설정 ----
        all_angles = np.linspace(0, 2*np.pi, ALL_DIR_NUM, endpoint=False) # 진행 가능한 각도
        self.all_dir_vecs = np.column_stack([np.cos(all_angles), np.sin(all_angles)]) # self.dir에 대응하는 방향 벡터 배열. Shape: (N, 2)
        
        diridx_diff = []
        for i in range(1, ANG_SEG_NUM+1):
            diridx_diff.extend([i, -i]) 
        self.action_diridx = np.array([0] + diridx_diff + [ANG_SEG_NUM*2]) # action index에 대응하는 방향 벡터의 인덱스 변화량. Index가 클수록 각도 변화량이 큼.
        self.ang_diff_diridx = self.action_diridx * (np.pi/2) / ANG_SEG_NUM # action index에 대응하는 각도 변화량. 단위: rad
        self.base_dir_vecs = [(1, 0), (-1, 0), (0, 1), (0, -1)] # Reachable grid 계산용
        # --------------------------------
        
        # ---- 7. Visualization을 위한 figure과 canvas 설정 ----
        self.fig = Figure()
        self.canvas = FigureCanvasAgg(self.fig)
        # ----------------------------------------------------

        # self.cleaned_segment: np.ndarray | None = None
    
    def reset(self, *, seed: int=None, mode: str='train', map_config: MapConfigSchema | None = None) -> tuple[dict, dict] | None:
        """

        Args:
            seed (int, optional): Random generator seed. Defaults to None.
            map_config (MapConfigSchema, optional): None이면 map을 새로 생성하지 않고 시작점만 초기화. Defaults to None.

        Returns:
            tuple[dict, dict] | None: tuple[dict, dict]인 경우, (obs, info)의 tuple이 출력. None인 경우는 reset에 실패한 경우. Map을 바꿔야 함.
        """
        super().reset(seed=seed) # 난수 생성기 self.np_random이 해당 seed로 초기화됨.

        # 새로운 map 생성 또는 시작점만 바꾸고 map 초기화
        for _ in range(100):
            self.pos = self.map_layers.reset(map_config=map_config, mode=mode, env_rng=self.env_rng)
            
            # 시작 지점이 정해지면 이후 step 진행
            if self.pos is not None:
                break
            
            # map이 지정된 경우에 시작 지점을 설정하지 못한 경우, for문을 더 수행하지 않고 method를 종료. 다음 map으로 넘어가기 위해 None을 return.
            if map_config and map_config.file_path and self.pos is None:
                # print(f"Failed to set start pos at map file {os.path.basename(map_config.file_path)}")
                return None
        
        # map 스펙이 지정되고 random하게 map을 생성하는 경우, 100회의 시도 이후에도 시작점을 설정하지 못하면 reset을 종료. 다른 map_config을 받기 위해 None을 return.    
        else:
            print("Failed to reset environment after 100 attempts.")
            print(f"Map config: {map_config}")
            return None
            
        # 초기 로봇이 바라보는 방향: 벽면의 반대 방향
        self.H = self.map_layers.map_info.H
        self.W = self.map_layers.map_info.W
        eff_size = self.map_layers.map_info.eff_size
        self.dir = self._get_init_dir(self.pos, eff_size, mode)

        # self.cleaned_segment = np.zeros((self.H, self.W), dtype=np.uint8)
        
        # 로봇이 (cx, cy)로 이동했을 때 로봇이 cover한 영역, coverage 수치 등을 update
        self.map_layers.update_map_layers(*self.pos)
        self.last_coverage = self.coverage
        
        # 변수 초기화
        self.steps = 0
        self.collision_count = 0
        self.last_coverage = 0.0
        self.no_progress_cnt = 0
        self.env_history = [self._get_env_history_data()]
        self.done = False
        
        # self.patch_stack reset
        self.patch_stack = deque(maxlen=self.cfg.stack_steps) # Observation 정보를 얻는 local map step 수
        curr_patch = self._get_processed_patch(*self.pos)
        self._update_patch_stack(curr_patch)
            
        return self._get_obs(), {"Start_pos": self.pos}
        
    
    def step(self, action=None, global_dir=None):
        
        if action is None and global_dir is None:
            raise ValueError("Either action or global_dir must be provided.")
        
        if self.done:
            return self._get_obs(), 0.0, True, True, {}
            
        self.steps += 1

        # 초기 설정 값
        reward = 0.0
        terminated = False
        truncated = False
        collision = False
        success = False
        
        cx, cy = self.pos
        
        ############### 다음 위치로 이동: Collision이 일어나면 더 나아가지 않음 ###############
        # Collision이 일어나면, collision 사실을 info에 넣어 알리기만 하고 더 나아가지 않음
        # 이 작업은 training 시와 test 시에 동일하게 동작함.
        
        if global_dir is not None: # North, East, West, South 방향으로 바로 이동하는 경우
            self.dir = global_dir * ANG_SEG_NUM
            nx, ny = self.get_next_pos(self.dir, action=0) # self.dir를 바꾼 뒤 action 0 수행하여 해당 방향으로 이동
        else: # DQN network로 얻은 action으로 이동하는 경우
            nx, ny = self.get_next_pos(self.dir, action)
            self.dir = self._get_all_dir_indices(self.dir)[action] # 방향 update
        
        collided, new_cleaned_degree, revisit_degree = self.map_layers.update_map_layers(nx, ny)
            
        # ------------------------------------------------------------
        # [REWARD FUNCTION]
        # ------------------------------------------------------------
        # 1. Step penalty
        reward -= self.cfg.step_penalty
        
        if collided:
            # 2. Obstacle penalty
            reward -= self.cfg.obstacle_penalty            
        else:
            # 3. Cover reward and Cleaned grid penalty
            if new_cleaned_degree > 0:
                reward += self.cfg.uncleaned_reward * new_cleaned_degree
                reward += self.coverage * self.cfg.step_penalty # 새로운 grid를 cover하면 step_penalty를 완화
            else:
                reward -= revisit_degree * self.cfg.cleaned_penalty
            
        # 4. Turn penalty
        if global_dir is not None:
            ang_diff_coeff = 0.0  # global_dir 이동(zigzag 등)은 action 기반 회전 패널티 대상이 아님
        else:
            ang_diff_coeff = abs(self.action_diridx[action]) / ANG_SEG_NUM    # 90도 회전: 1.0, 180도 회전: 2.0
        reward -= ang_diff_coeff * self.cfg.turn_penalty
        
        # 5. Complete reward
        if self.coverage >= self.cfg.target_coverage:
            reward += self.cfg.complete_reward
        # ------------------------------------------------------------
            
        # 상태 update
        if collided:
            collision = True
            self.collision_count += 1
            nx, ny = cx, cy # 충돌이 일어나면 로봇을 움직이지 않음
        
        self.pos = (nx, ny)
        ############################################################################
        
        # ---- Terminate과 Truncated 조건 ----
        # Terminate: Coverage를 성공한 경우 & Collision이 일어난 경우
        # Truncate: 전체 step 수가 max_steps를 넘은 경우 & Coverage가 증가하지 않는 상태로 일정 이상의 step을 수행한 경우
        cur_coverage = self.coverage
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
            max_steps = self.cfg.max_steps
            
        if self.steps >= max_steps:
            truncated = True

        if terminated or truncated:
            self.done = True
            
        info = {
            "Coverage": cur_coverage, 
            "Steps": self.steps, 
            "Collision": collision,
            "Success": success,
            "Episode_collision": self.collision_count,
        }
        # ----------------------------------
        
        # Trajectory data를 저장
        self.env_history.append(self._get_env_history_data())
        
        # self.patch_stack update
        curr_patch = self._get_processed_patch(cx, cy)
        self._update_patch_stack(curr_patch)

        return self._get_obs(), reward, terminated, truncated, info
    
    
    def one_step_back(self, remain_collision_count: bool = False):
        """_summary_

        Args:
            remain_collision_count (bool, optional): 가상의 step을 수행한 뒤 다시 back step을 하는 경우, collision count를 유지해야 함. Defaults to False.
        """        
        self.env_history.pop()
        last_instance_data = self.env_history[-1] # 가장 마지막 position, direction, step 수 등에 관한 데이터
        
        # Instance variable 변경
        self.steps = last_instance_data.steps
        self.pos = last_instance_data.pos
        self.dir = last_instance_data.direc
        self.no_progress_cnt = last_instance_data.no_progress_cnt
        self.last_coverage = last_instance_data.last_coverage
        
        if remain_collision_count:
            self.collision_count = last_instance_data.collision_cnt
        
        # Map layer 변경 및 self.patch_stack 변경
        crop_r = self.raw_patch_crop_r
        raw_past_patch = self.map_layers.one_step_back(crop_radius=crop_r, stack_steps_num=self.cfg.stack_steps)
        processed_past_patch = self._get_processed_patch(raw_patch=raw_past_patch)
        self.patch_stack.appendleft(processed_past_patch)
        
        
    def backstep(self, step_num: int = 1):
        """
        step_num만큼 뒤로 돌아감.

        Args:
            step_num (int, optional): _description_. Defaults to 1.
        """
        
        if len(self.env_history) <= step_num + 1:
            print(f"Cannnot backstep {step_num} steps. Only {len(self.env_history)-1} steps are available.")
            return self._get_obs()
        
        while step_num > 0:
            self.one_step_back(remain_collision_count=True)
            step_num -= 1
        
        return self._get_obs()
    

    def set_env_mode(self, warmup: bool = False, ep_steps: int = 2000):
        self.warmup = warmup
        self.warmup_steps = ep_steps
        
    
    def is_collide(self, cx: float, cy: float) -> bool:
        return self.map_layers.collides(cx, cy)
    

    @property
    def coverage(self) -> float:
        
        if self.map_layers.coverable is None:
            free = (self.map_layers.obstacles == 0)
            total = int(free.sum(dtype=np.int64))
            if total == 0:
                return 0.0
            cleaned_mask = (self.map_layers.cleaned > 0)
            cleaned_free = int((cleaned_mask & free).sum(dtype=np.int64))
            return cleaned_free / total

        if self.map_layers.total_coverable_area == 0:
            return 0.0
        
        return float(self.map_layers.coveraged_area / self.map_layers.total_coverable_area)
    
    @property
    def raw_patch_crop_r(self) -> float:
        
        r = int(self.cfg.local_view//2) # patch 중심으로부터 변까지의 pixel 수
        crop_r = int(math.ceil(r*1.5))  # 회전 변환을 위해 넉넉하게 crop(sqrt(2)보다 큰 배수로 crop)
        
        return crop_r
    
    @property
    def overlap_percent(self) -> float:
        
        waypoints_grid = np.array([data.pos for data in self.env_history])
        waypoints_m = waypoints_grid * self.cfg.grid_size * 0.01
        return overlap_percent(waypoints_m)
    
    @property
    def cleaning_time(self) -> float:
        
        waypoints_grid = np.array([data.pos for data in self.env_history])
        waypoints_m = waypoints_grid * self.cfg.grid_size * 0.01
        return cleaning_time_minutes(waypoints_m)
    
    @property
    def traj_arr(self) -> float | None:
        if len(self.env_history) <= 0:
            return np.empty((0, 2))
        return np.array([data.pos for data in self.env_history])
    
    
    def _ray_distance_forward(self, cx: float, cy: float, d: int) -> float:
        
        all_dir_indices = self._get_all_dir_indices(self.dir)
        dir_vecs = self.all_dir_vecs[all_dir_indices]   # Shape: (ACTION_NUM, 2)
        
        dx = dir_vecs[d][0]; dy = dir_vecs[d][1] # dx, dy는 실수
        
        # 1. 샘플링할 거리 배열 생성 (0부터 max_forward까지 1씩 증가)
        steps = np.arange(1, self.cfg.max_forward + 1)
        
        # 2. 각 step에서의 실제 물리적 위치(실수) 계산
        line_x = cx + steps * dx
        line_y = cy + steps * dy
        
        # 3. 맵 경계 내에 있는 좌표만 필터링
        valid = (line_x >= -0.5) & (line_x < self.W - 0.5) & \
                (line_y >= -0.5) & (line_y < self.H - 0.5)
        
        if not np.any(valid): # 해당 방향으로 한칸만 이동해도 로봇의 중심이 map을 벗어나는 경우
            return 0.0, self.cfg.max_forward
        
        # 4. 실수 좌표를 가장 가까운 정수 격자 인덱스로 변환 (반올림)
        line_x_idx, line_y_idx = float_to_int_coord(line_x[valid], line_y[valid])
        
        # 5. 충돌 검사 (장애물 탐색)
        line_collisions = self.map_layers.collision_map[line_y_idx, line_x_idx]
        hit_indices = np.where(line_collisions == 1)[0]

        if len(hit_indices) > 0:
            first_hit_idx = hit_indices[0] # 첫 장애물 인덱스
            dist_to_obstacle = float(first_hit_idx) # 장애물 직전까지의 이동 가능 거리 (격자 단위)
            
            # 💡 [핵심 1] 장애물 부딪히기 "직전"까지만 경로를 자름!
            line_x_idx = line_x_idx[:first_hit_idx]
            line_y_idx = line_y_idx[:first_hit_idx]
        else:
            dist_to_obstacle = float(len(line_x_idx))

        # 장애물에 바로 막혀서 전진 불가능한 경우
        if len(line_x_idx) == 0:
            return dist_to_obstacle, self.cfg.max_forward

        # 7. 새로운 grid를 cover할 수 있는 로봇 중심 위치까지의 거리 측정
        robot_off_x, robot_off_y = self.map_layers.robot_mask_offsets[:, 0], self.map_layers.robot_mask_offsets[:, 1]
        cover_x = (line_x_idx[:, None] + robot_off_x[None, :])
        cover_y = (line_y_idx[:, None] + robot_off_y[None, :])

        valid_coords = (cover_x >= 0) & (cover_x < self.W) & (cover_y >= 0) & (cover_y < self.H)
        safe_x = np.clip(cover_x, 0, self.W - 1)
        safe_y = np.clip(cover_y, 0, self.H - 1)

        is_uncleaned_cell = (self.map_layers.uncleaned[safe_y, safe_x] == 1) & valid_coords
        line_uncovered = np.any(is_uncleaned_cell, axis=1)
        new_cover_indices = np.where(line_uncovered)[0]

        if len(new_cover_indices) > 0:
            first_cover_idx = new_cover_indices[0] + 1
            dist_to_new_cover = float(first_cover_idx)
        else:
            dist_to_new_cover = self.cfg.max_forward

        return dist_to_obstacle, dist_to_new_cover
    
    
    def _get_processed_patch(self, cx: float = None, cy: float = None, raw_patch: np.ndarray | None = None) -> np.ndarray:
        """
        중심 (cx, cy)에서 local_view 크기의 local patch를 뽑음.
        Local patch의 processing은 여기에 구현

        Args:
            cx, cy (float): Crop 영역의 중심
            local_view (int): Local view의 가로, 세로 grid 개수

        Returns:
            np.ndarray: Processing까지 완료된 최종 local patch
        """
        
        cx, cy = self.pos
        r =  self.cfg.local_view//2
        crop_r = self.raw_patch_crop_r  # 회전 변환을 위해 넉넉하게 crop(sqrt(2)보다 큰 배수로 crop)
        
        # Raw patch가 없는 경우, 직접 crop하여 생성
        if raw_patch is None:
            raw_patch = self.map_layers.get_raw_patch(cx, cy, crop_r)
        
        H, W = raw_patch.shape[:2]
        
        # 회전 변환을 위한 affine matrix 생성
        dir_vecs = self.all_dir_vecs[self.dir]
        c = dir_vecs[0]; s = -dir_vecs[1] # 로봇이 바라보는 방향이 오른쪽 방향인 경우
        M = np.array([
            [ c, -s, (1-c)*crop_r +  s*crop_r],
            [ s,  c, -s*crop_r + (1-c)*crop_r]
        ])
        
        # 회전 실행 (INTER_LINEAR: 부드러운 보간, INTER_NEAREST: 픽셀값 보존)
        # 장애물 마스크 등이 포함되어 있다면 INTER_LINEAR 후 임계값 처리를 권장합니다.
        rotated_patch = cv2.warpAffine(raw_patch, M, (W, H), flags=cv2.INTER_LINEAR)

        # Polar coordinate으로 변환
        # final_patch = rotated_patch[crop_r-r:crop_r+r+1, crop_r-r:crop_r+r+1]
        final_patch = cv2.warpPolar(rotated_patch, 
                                    dsize=(self.cfg.local_view, self.cfg.local_view), 
                                    center=(crop_r, crop_r),
                                    maxRadius=r,
                                    flags=cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR)

        # 후처리 (정규화 및 차원 복구)
        if final_patch.ndim == 3:
            
            final_patch = final_patch.transpose(2, 0, 1).astype(np.float32) # (H, W, C) -> (C, H, W) 복구
            
            # Cleaned_map에서 robot이 차지하고 있는 부분은 0으로 만듦
            robot_r = int(self.cfg.robot_size // 2)
            robot_r_idx = math.ceil((robot_r / r) * self.cfg.local_view)
            final_patch[1] /= CLEANED_MAP_MAX
            final_patch[1, :, :robot_r_idx+1] = 0
        else:
            final_patch = final_patch.astype(np.float32)

        return final_patch
    
    
    def _update_patch_stack(self, current_patch: np.ndarray):
        """
        self.patch_stack에 current_patch를 추가

        Args:
            current_patch (np.ndarray): self._get_processed_patch로 얻은 현재 local view patch
        """
        if len(self.patch_stack) == 0:
            self.patch_stack.extend([current_patch] * self.cfg.stack_steps)
        else:
            self.patch_stack.append(current_patch)
    
    
    def _get_obs(self) -> dict:
        
        cx, cy = self.pos
        
        # -------------- Map data 생성 --------------
        # self.patch_stack에 저장된 patch들을 합쳐서 observation으로 사용
        total_patch = np.concatenate(list(self.patch_stack), axis=0).astype(np.float32)
        # ------------------------------------------
        
        # ---------------- Additional state vector ----------------
        
        # Local information
        num_action = self.action_space.n
        
        # 각 방향에서 바라본 여유 공간: [-1, 1]이 범위로 정규화
        obs_dist = np.zeros(num_action, dtype=np.float32)
        new_uncover_dist = np.zeros(num_action, dtype=np.float32)
        for d in range(num_action):
            obs_dist[d], new_uncover_dist[d] = self._ray_distance_forward(cx, cy, d)
        # 여유 공간을 보고 충돌할 수 있는 action을 제외하기 위한 action_mask를 생성
        action_mask = (obs_dist != 0).astype(np.float32)
        uncover_score = float(self.cfg.max_forward) - new_uncover_dist
        ray_raw = np.concatenate([obs_dist, uncover_score])
        ray_norm = (ray_raw / max(1, self.cfg.max_forward))*2-1
        
        loc_vec = ray_norm
        
        # Global information
        
        # 로봇의 위치 정보: [-1, 1]의 범위로 정규화
        x_norm = (cx/(self.W-1))*2-1
        y_norm = (cy/(self.H-1))*2-1

        # 로봇이 바라보는 방향
        dir_vec = self.all_dir_vecs[self.dir]

        # Coverage 비율: [0, 1] 범위의 숫자를 [-1, 1] 범위로 정규화
        cov = self.coverage
        cov_norm = cov*2-1
        
        glob_vec = np.concatenate([
            np.array([x_norm, y_norm], dtype=np.float32),
            dir_vec,
            np.array([cov_norm], dtype=np.float32),
        ], axis=0).astype(np.float32)
        # ----------------------------------------------------
        
        return {"map": total_patch, "loc_vec": loc_vec, "glob_vec": glob_vec, "action_mask": action_mask}
    
    
    def _get_env_history_data(self) -> EnvHistory:
        """
        self.env_history에 저장할 단일 data를 생성하는 method
        Action이 수행된 후의 data를 저장

        Returns:
            EnvHistory: Environment의 history를 저장하는 양식
        """
        
        env_history = EnvHistory(steps=self.steps,
                                 pos=self.pos,
                                 collision_cnt=self.collision_count,
                                 direc=self.dir,
                                 no_progress_cnt=self.no_progress_cnt,
                                 last_coverage=self.last_coverage)
        
        return env_history


    def _get_init_dir(self, init_pos: tuple[int, int], eff_size: BoundingBox, mode: str='train'):
        """
        초기 위치에서 로봇이 바라보는 방향 설정: 로봇이 붙어 있는 벽과 반대 방향을 바라보도록 설정
        """
        wall_dists = np.array([
            (eff_size.y_max-1) - init_pos[1],    # 0: North wall distance
            init_pos[1] - eff_size.y_min,       # 1: South wall distance
            init_pos[0] - eff_size.x_min,       # 2: West wall distance
            (eff_size.x_max-1) - init_pos[0],   # 3: East wall distance
        ])
        
        # 각 벽(North, South, West, East)에 가까울 때 '바라봐야 할 반대 방향' 세그먼트 인덱스
        # 예시 (0: East, 1: North, 2: West, 3: South 기준):
        # - North 벽(0번)에 붙음 -> South(3번) 바라봄
        # - South 벽(1번)에 붙음 -> North(1번) 바라봄
        # - West 벽(2번)에 붙음  -> East(0번) 바라봄
        # - East 벽(3번)에 붙음  -> West(2번) 바라봄
        opposite_dir_map = np.array([3, 1, 0, 2]) # 사용하는 환경 각도 정의에 맞춰 수치 조정
        
        min_dist = wall_dists.min()
        closest_wall_indices = np.where(wall_dists == min_dist)[0]
        
        # 가장 가까운 벽들의 '반대 방향' 세그먼트들 추출
        first_candidate_dir = opposite_dir_map[closest_wall_indices] * ANG_SEG_NUM

        if mode == 'train':
            return int(self.env_rng.choice(first_candidate_dir))
        else:
            return int(first_candidate_dir[0])


    def _get_all_dir_indices(self, dir: int = None) -> np.ndarray:
        """
        로봇이 바라본븐 방향이 dir일 때, 각 action이 가리키는 방향 벡터를 action index 순서대로 출력하는 method
        
        Args:
            dir (int, optional): 로봇이 바라보는 방향. Defaults to None.

        Returns:
            np.ndarray: 각 action이 가리키는 방향 벡터
        """

        assert dir is not None, "Direction (self.dir) is not set."
        return (dir + self.action_diridx) % ALL_DIR_NUM    
        

    def get_next_pos(self, dir: int = None, action: int = None) -> tuple[float, float]:
        
        assert self.pos is not None, "Current position (self.pos) is not set."
        assert action is not None, "Action is not provided."
        if dir is None:
            dir = self.dir

        dir_idx = self._get_all_dir_indices(dir)[action]
        dx = self.all_dir_vecs[dir_idx][0]; dy = self.all_dir_vecs[dir_idx][1]
        cx, cy = self.pos
        return float(cx + dx), float(cy + dy)
    
    
    def get_dir_from_next_pos(self, next_pos: tuple[float, float], curr_pos: tuple[float, float] = None) -> int:
        eps = 1e-7
        if curr_pos is None:
            curr_pos = self.pos
        cx, cy = curr_pos; nx, ny = next_pos
            
        for dir_idx, dir_vec in enumerate(self.all_dir_vecs):
            dx = dir_vec[0]; dy = dir_vec[1]
            if abs(nx - (cx+dx)) < eps and abs(ny - (cy+dy)) < eps:
                return dir_idx
        else:
            raise ValueError(f"We can't arrived at next_pos {next_pos} from start_pos {self.pos} by one action!")

    
    def get_visualized_img(self, img_choice: str = 'traj', preprocessor = None) -> np.ndarray:

        if img_choice not in ['layer', 'traj', 'obs']:
            raise ValueError(f"Invalid img_choice: {img_choice}. Must be one of ['layer', 'traj', 'obs']")
           
        if img_choice == 'layer':
            draw_layer(map_layers=self.map_layers, 
                       fig=self.fig, 
                       pos=self.pos, 
                       last_coverage=self.last_coverage)
        elif img_choice == 'traj':
            draw_traj(map_layers=self.map_layers, 
                      fig=self.fig, 
                      pos=self.pos, 
                      traj_arr=self.traj_arr, 
                      coverage=self.coverage, 
                      overlap_percent=self.overlap_percent, 
                      cleaning_time=self.cleaning_time, 
                      cleaned_map_max=CLEANED_MAP_MAX)
        else:
            draw_obs(obs=self._get_obs(), 
                     fig=self.fig, 
                     stack_steps=self.cfg.stack_steps, 
                     local_view_dim=self.cfg.local_view, 
                     cleaned_map_max=CLEANED_MAP_MAX,
                     preprocessor=preprocessor)
            
        self.canvas.draw()
        return np.array(self.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3] # [H, W, C]

    def show_visualized_img(self, img_choice: str = 'traj'):

        if img_choice not in ['layer', 'traj', 'obs']:
            raise ValueError(f"Invalid img_choice: {img_choice}. Must be one of ['layer', 'traj', 'obs']")
        
        img_array = self.get_visualized_img(img_choice)
        display_image(img_array)
    
    # def is_zigzag_zone(self, cx: float = None, cy: float = None):
    #     if cx is None or cy is None:
    #         cx, cy = self.pos
    #     cx, cy = float_to_int_coord(cx, cy)
    #     return self.map_layers.mode_map[cy, cx] == 0
            

if __name__ == "__main__":
    
    # environment가 잘 생성되었는지 테스트하는 코드
    args = get_args()
    cfg = EnvConfig()
    env = CoverageEnv(cfg)

    # 0. Robot visualize
    if args.robot:
        visualize_mask(make_robot_mask(robot_size=cfg.robot_size))
    
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
        action_seq = [0]*1000 + [3]*1000 + [2]*1000 + [1]*1000
        for action in action_seq:
            obs, reward, terminated, truncated, info = env.step(action)
        if args.see_map:
            # 1. Map visualize: Agent layer, Cleaned layer, Obstacle layer, Collision map, 
            env.show_visualized_img('layer')
            env.show_visualized_img('traj')
        if args.see_obs:
            # 2. observation visualization
            env.show_visualized_img('obs')