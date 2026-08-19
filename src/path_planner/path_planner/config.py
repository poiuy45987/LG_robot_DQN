from dataclasses import dataclass, field
import os
import warnings
from enum import IntEnum
from frozendict import frozendict

DEFAULT_SEED = 42

ANG_SEG_NUM = 7 # 0~90도 사이 각도를 나누는 간격 수.
CLEANED_MAP_MAX = 10
LOCAL_VIEW_DIM = 51
# Fixed resolution for the full-map observation.  Map dimensions vary by the
# curriculum, so the global state is downsampled before entering the network.
GLOBAL_MAP_DIM = 17

MAP_SAVE_DIR = os.path.normpath('src/path_planner/path_planner/maps')
MODEL_SAVE_DIR = os.path.normpath('src/path_planner/path_planner/models')
TB_SAVE_DIR = os.path.normpath('src/path_planner/path_planner/models')
RESULT_SAVE_DIR = os.path.normpath('src/path_planner/path_planner/result')

TRAIN_MAP_FILE_FORMAT = "{mode}_map_level{level}_{map_id:04d}_{H}x{W}.npy"
TRAIN_MAP_FILE_REGEX = r"(?P<mode>\w+)_map_level(?P<level>\d+)_(?P<map_id>\d+)_(?P<H>\d+)x(?P<W>\d+)"
TEST_MAP_FILE_REGEX = r"(?P<mode>\w+)_map_L(?P<level>\d+)_(?P<map_id>\d+)"

# Map을 시각화할 때 table, chair을 구별하기 위해 설정한 값
# 실제로 훈련 또는 validation 과정에서 map을 생성할 때는 장애물을 전부 1로 바꿈
class GridCell(IntEnum):
    BLANK = 0
    WALL = 1
    TABLE = 2
    CHAIR = 3
    MORE_OBS = 4

@dataclass
class TrainConfig: # Training과 관련된 설정들
    
    # --- Training 설정 및 Warmup ---
    max_episodes: int = 30000
    batch_size: int = 32
    use_train_maps: bool = False
    max_step_per_eps: int = 10
    max_eps_per_map_size: int = 100
    path_reward_thres: float = 0.85
    
    # --- Optimizer 및 Update 주기 ---
    optimizer: str = 'adam' # 'sgd' or 'adam'
    lr: float = 3e-4
    momentum: float = 0.9
    min_lr: float = 1e-6
    detach_period: int = 20
    scheduler_max_step: int = 500
    
    # --- Validation 및 Checkpoint ---
    valid_freq: int = 1
    ckp_freq: int = 20
    valid_map_num: int = 5
    valid_start_point_num: int = 1
    
    def _validate_configs(self):

        if self.max_episodes < self.max_eps_per_map_size:
            warnings.warn(f"It is recommended that max_episodes({self.max_episodes}) is bigger than max_eps_per_map_size({self.max_eps_per_map_size}).")
        if self.path_reward_thres < 0.5:
            warnings.warn(f"It is recommended that path_reward_thres({self.path_reward_thres}) is bigger than 0.5.")
        if self.path_reward_thres > 1.0:
            warnings.warn(f"It is recommended that path_reward_thres({self.path_reward_thres}) is smaller than 1.0.")
        
        # Optimizer 관련 설정
        if self.lr <= 0:
            raise ValueError(f"lr({self.lr}) is negative. It should be posivie value.")
        if self.min_lr > self.lr:
            raise ValueError(f"Initial lr({self.lr}) is smaller than min_lr({self.min_lr}).")
        
    def __post_init__(self):
        
        self._validate_configs() # 정상적인 config가 들어왔는지 검사


@dataclass
class MapConfig:
    
    # Size와 관련된 인자들을 cm 단위로 받고 grid 단위로 변환
    
    # ---- house map params ----
    
    grid_size: float = 2.5 # cm 단위
    init_map_height: float = 400.0 # cm 단위
    init_map_width: float = 400.0 # cm 단위
    
    # Map 크기를 grid 단위로 변환
    init_H: int = field(init=False)
    init_W: int = field(init=False)

    # 모든 size는 cm 단위
    num_tables: int = 1
    max_table_size: float = 200.0
    min_table_size: float = 100.0
    table_leg_size: float = 10.0

    num_chairs_per_level: frozendict = frozendict({
        1: 3,
        2: 4,
        3: 6,
        4: 6
    })
    
    num_small_obs_per_level: frozendict = frozendict({
        1: 2,
        2: 3,
        3: 4,
        4: 5
    })
    
    chairs_per_table_min: int = 2
    chairs_per_table_max: int = 4
    max_chair_size: float = 55.0
    min_chair_size: float = 45.0
    chair_leg_size: float = 5.0
    chair_spread: float = 20.0
    
    small_obs_size_max: float = 15.0
    small_obs_size_min: float = 10.0
    window_size: float = 100
    small_obs_num_per_window_max: int = 0
    small_obs_num_per_window_min: int = 0
    
    def _to_grid(self, value: float) -> int:
        """cm 단위를 grid 단위로 변환 (최소 1 그리드 보장)"""
        return int(max(1, value // self.grid_size))
    
    def _make_odd(self, value: int) -> int:
        """값이 짝수일 경우, 하나 더 큰 홀수로 변환"""
        return value + 1 if value % 2 == 0 else value
    
    def _validate_configs(self):
        
        # 1. max값과 min값 비교
        if self.max_table_size < self.min_table_size:
            raise ValueError(f"max_table_size({self.max_table_size}) is smaller than min_table_size({self.min_table_size}).")
        if self.max_chair_size < self.min_chair_size:
            raise ValueError(f"max_chair_size({self.max_chair_size}) is smaller than min_chair_size({self.min_chair_size}).")
        if self.chairs_per_table_max < self.chairs_per_table_min:
            raise ValueError(f"chairs_per_table_max({self.chairs_per_table_max}) is smaller than chairs_per_table_min({self.chairs_per_table_min}).")
        if self.small_obs_size_max < self.small_obs_size_min:
            raise ValueError(f"small_obs_size_max({self.small_obs_size_max}) is smaller than small_obs_size_min({self.small_obs_size_min}).")
        if self.small_obs_num_per_window_max < self.small_obs_num_per_window_min:
            raise ValueError(f"small_obs_num_per_window_max({self.small_obs_num_per_window_max}) is smaller than small_obs_num_per_window_min({self.small_obs_num_per_window_min}).")
        
        # 2. 의자 및 책상 다리 두께 검사
        if self.table_leg_size*2 >= self.min_table_size:
            raise ValueError(f"table_leg_size({self.table_leg_size}) is too big. It should be smaller than {self.min_table_size/2.0}.")
        if self.chair_leg_size*2 >= self.max_chair_size:
            raise ValueError(f"chair_leg_size({self.chair_leg_size}) is too big. It should be smaller than {self.max_chair_size/2.0}.")
        if self.chair_leg_size*2 >= self.min_chair_size:
            raise ValueError(f"chair_leg_size({self.chair_leg_size}) is too big. It should be smaller than {self.min_chair_size/2.0}.")
        
        # 3. 맵 크기 대비 grid, window 등의 size가 적절한지 검사
        min_map_dim = min(self.init_map_height, self.init_map_width)
        if self.grid_size > min_map_dim:
            raise ValueError(f"grid_size({self.grid_size}) is too big. It should be smaller than {min(self.init_map_height, self.init_map_width)}.")
        if self.window_size > min_map_dim:
            raise ValueError(f"window_size({self.window_size}) is too big. It should be smaller than {min(self.init_map_height, self.init_map_width)}.")
    
    def __post_init__(self):
        
        self._validate_configs() # 정상적인 config가 들어왔는지 검사
        
        # ---- cm 단위로 들어온 size 값들을 grid 단위로 변환 ----
        self.init_H = self._to_grid(self.init_map_height)
        self.init_W = self._to_grid(self.init_map_width)
        
        self.max_table_size = self._to_grid(self.max_table_size)
        self.min_table_size = self._to_grid(self.min_table_size)
        self.table_leg_size = self._to_grid(self.table_leg_size)
        
        self.max_chair_size = self._to_grid(self.max_chair_size)
        self.min_chair_size = self._to_grid(self.min_chair_size)
        self.chair_leg_size = self._to_grid(self.chair_leg_size)
        self.chair_spread = self._to_grid(self.chair_spread)
        
        self.window_size = self._to_grid(self.window_size)
        self.small_obs_size_max = self._to_grid(self.small_obs_size_max)
        self.small_obs_size_min = self._to_grid(self.small_obs_size_min)
    

@dataclass
class EnvConfig:
    
    map_cfg: MapConfig = field(default_factory=MapConfig)
    
    # ---- environment params ----
    
    # Size 관련
    robot_size: float = 36.0 # cm 단위
    local_view: int = 200  # 단위: cm
    max_forward: int = 50 # 단위: cm
    
    # Step, Termination 관련
    max_steps: int = 3000
    max_no_progress_steps: int = 60
    max_no_progress_steps_final: int = 30
    target_coverage: float = 0.95
    final_coverage_thres: float = 0.90
    stack_steps: int = 1 # Map의 observation data의 step 수
    
    # Reward function parameter (footprint-based)
    uncleaned_reward: float = 1.0
    cleaned_penalty: float = 0.1
    obstacle_penalty: float = 10.0
    turn_penalty: float = 0.1
    step_penalty: float = 0.01
    complete_reward: float = 1.0
    intrinsic_reward: float = 1.0
    
    @property
    def grid_size(self) -> float: return self.map_cfg.grid_size
    @property
    def init_map_height(self) -> float: return self.map_cfg.init_map_height
    @property
    def init_map_width(self) -> float: return self.map_cfg.init_map_width
    @property
    def init_H(self) -> int: return self.map_cfg.init_H
    @property
    def init_W(self) -> int: return self.map_cfg.init_W
    
    def _to_grid(self, value: float) -> int: return self.map_cfg._to_grid(value)
    def _make_odd(self, value: int) -> int: return self.map_cfg._make_odd(value)
    
    def _validate_configs(self):
        
        # 1. max값과 min값 비교
        if self.target_coverage < self.final_coverage_thres:
            raise ValueError(f"target_coverage({self.target_coverage}) is smaller than final_coverage_thres({self.final_coverage_thres}).")
        
        # 2. 설정된 step 수 비교
        if self.max_steps < self.max_no_progress_steps:
            warnings.warn(f"It is recommended that max_steps({self.max_steps}) is bigger than max_no_progress_steps({self.max_no_progress_steps}).")
        if self.max_steps < self.max_no_progress_steps_final:
            warnings.warn(f"It is recommended that max_steps({self.max_steps}) is bigger than max_no_progress_steps_final({self.max_no_progress_steps_final}).")
        if self.max_no_progress_steps_final < self.max_no_progress_steps:
            warnings.warn(f"It is recommended that max_no_progress_steps_final({self.max_no_progress_steps_final}) is bigger than max_no_progress_steps({self.max_no_progress_steps}).")
        
        # 3. 맵 크기 대비 grid, window, robot 등의 size가 적절한지 검사
        min_map_dim = min(self.init_map_height, self.init_map_width)
        if self.robot_size > min_map_dim:
            raise ValueError(f"robot_size({self.robot_size}) is too big. It should be smaller than {min(self.init_map_height, self.init_map_width)}.")
        if self.local_view > min_map_dim:
            raise ValueError(f"local_view({self.local_view}) is too big. It should be smaller than {min(self.init_map_height, self.init_map_width)}.")
        
        # 4. Reward 수치 경고
        if self.uncleaned_reward < 0:
            warnings.warn(f"uncleaned_reward({self.uncleaned_reward}) is negative. All reward hyperparameters should be positive value.")
        if self.cleaned_penalty < 0:
            warnings.warn(f"cleaned_penalty({self.cleaned_penalty}) is negative. All reward hyperparameters should be positive value.")
        if self.obstacle_penalty < 0:
            warnings.warn(f"obstacle_penalty({self.obstacle_penalty}) is negative. All reward hyperparameters should be positive value.")
        if self.turn_penalty < 0:
            warnings.warn(f"turn_penalty({self.turn_penalty}) is negative. All reward hyperparameters should be positive value.")
        if self.step_penalty < 0:
            warnings.warn(f"step_penalty({self.step_penalty}) is negative. All reward hyperparameters should be positive value.")
        if self.complete_reward < 0:
            warnings.warn(f"complete_reward({self.complete_reward}) is negative. All reward hyperparameters should be positive value.")        
        if self.intrinsic_reward < 0:
            warnings.warn(f"intrinsic_reward({self.intrinsic_reward}) is negative. All reward hyperparameters should be positive value.")
    
    def __post_init__(self):
        
        self._validate_configs() # 정상적인 config가 들어왔는지 검사
        
        # self.robot_size = self._make_odd(self._to_grid(self.robot_size))
        self.robot_size = self._make_odd(self._to_grid(self.robot_size))
        self.local_view = self._make_odd(self._to_grid(self.local_view))
        self.max_forward = self._to_grid(self.max_forward) # 각 방향으로의 여유 grid 수의 최댓값
