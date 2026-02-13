from dataclasses import dataclass, field

DEFAULT_SEED = 42

@dataclass
class TrainConfig: # Training과 관련된 설정들
    
    # --- Training 설정 및 Warmup ---
    buffer_size: int = 500000
    warmup_episodes: int = 20
    warmup_ep_steps: int = 10000
    warmup_tot_steps: int = 200000
    max_episodes: int = 30000
    batch_size: int = 64
    
    # --- Optimizer 및 Update 주기 ---
    optimizer: str = 'sgd' # 'sgd' or 'adam'
    lr: float = 1e-4
    momentum: float = 0.9
    target_update: int = 1000
    policy_update: int = 20
    
    # --- Validation 및 Checkpoint ---
    valid_freq: int = 100
    ckp_freq: int = 20
    valid_map_num: int = 5
    valid_start_point_num: int = 3
    
    # --- Exploration 관련 ---
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: int = 2000000
    use_softmax: bool = False
    softmax_temp: float = 1.0
    use_noisy: bool = False
    target_with_noisy: bool = False
    
    # --- State pre-processing 및 Q-value ---
    grid_map_size: int = 51
    do_normalize: bool = False
    gamma: float = 0.99
    
    # --- Map 설정 ---
    reset_only_start_pos: bool = False
    

@dataclass
class EnvConfig:
    
    # Size와 관련된 인자들을 cm 단위로 받고 grid 단위로 변환
    
    # ---- house map params ----
    
    grid_size: float = 4.0 # cm 단위
    map_height: float = 800.0 # cm 단위
    map_width: float = 800.0 # cm 단위
    
    # Map 크기를 grid 단위로 변환
    H: int = field(init=False)
    W: int = field(init=False)

    boundary_thickness: int = 1 # Map 전체를 둘러싼 벽의 두께, Grid 단위
    
    not_use_house_map: bool = False
    use_house_map: bool = field(init=False)

    # 모든 size는 cm 단위
    num_tables: int = 10
    max_table_size: float = 100.0
    min_table_size: float = 80.0
    table_leg_size: float = 8.0

    chair_size: float = 50.0
    chairs_per_table_min: int = 2
    chairs_per_table_max: int = 4
    chair_leg_size: float = 4.0
    chair_spread: float = 15.0
    
    small_obs_size_max: float = 15.0
    small_obs_size_min: float = 10.0
    window_size: float = 100
    small_obs_num_per_window_max: int = 2
    small_obs_num_per_window_min: int = 1
    
    # ---- environment params ----
    
    max_steps: int = 100000
    max_no_progress_steps: int = 300
    max_no_progress_steps_final: int = 500
    target_coverage: float = 0.95
    final_coverage_thres: float = 0.90
    local_view: int = 200  # 단위: cm
    max_forward: int = 50 # 단위: cm
    robot_size: float = 36.0 # cm 단위
    
    # max_rotate_k: int = 4

    # Reward function parameter (footprint-based)
    uncleaned_reward: float = 1.0
    cleaned_penalty: float = -0.1
    obstacle_penalty: float = -10.0
    
    def _to_grid(self, value: float) -> int:
        """cm 단위를 grid 단위로 변환 (최소 1 그리드 보장)"""
        return int(max(1, value // self.grid_size))
    
    def _make_odd(self, value: int) -> int:
        """값이 짝수일 경우, 하나 더 큰 홀수로 변환"""
        return value + 1 if value % 2 == 0 else value
    
    def __post_init__(self):
        
        self.robot_size = self._make_odd(self._to_grid(self.robot_size))
        self.local_view = self._make_odd(self._to_grid(self.local_view))
        
        self.use_house_map = not self.not_use_house_map
        
        # ---- cm 단위로 들어온 size 값들을 grid 단위로 변환 ----
            
        self.max_forward = self._to_grid(self.max_forward) # 각 방향으로의 여유 grid 수의 최댓값
        
        # map_generator 변수
        self.H = self._to_grid(self.map_height)
        self.W = self._to_grid(self.map_width)
        
        self.max_table_size = self._to_grid(self.max_table_size)
        self.min_table_size = self._to_grid(self.min_table_size)
        self.table_leg_size = self._to_grid(self.table_leg_size)
        
        self.chair_size = self._to_grid(self.chair_size)
        self.chair_leg_size = self._to_grid(self.chair_leg_size)
        self.chair_spread = self._to_grid(self.chair_spread)
        
        self.window_size = self._to_grid(self.window_size)
        self.small_obs_size_max = self._to_grid(self.small_obs_size_max)
        self.small_obs_size_min = self._to_grid(self.small_obs_size_min)