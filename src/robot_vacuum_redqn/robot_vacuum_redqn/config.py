from dataclasses import dataclass, field
import warnings

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
    use_epsilon: bool = False
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
    
    def _validate_configs(self):
        
        # Training episode와 step 수, buffer 관련 설정
        if self.buffer_size <= self.batch_size:
            raise ValueError(f"buffer_size({self.buffer_size}) should be bigger than batch_size({self.batch_size}).")
        if self.buffer_size <= self.warmup_tot_steps:
            raise ValueError(f"buffer_size({self.buffer_size}) should be bigger than warmup_tot_steps({self.warmup_tot_steps}).")
        if self.max_episodes <= self.warmup_episodes:
            raise ValueError(f"max_episodes({self.max_episodes}) should be bigger than warmup_episodes({self.warmup_episodes}).")
        
        # Optimizer 관련 설정
        if self.lr <= 0:
            raise ValueError(f"lr({self.lr}) is negative. It should be posivie value.")
        if self.target_update <= self.policy_update:
            warnings.warn(f"target_update({self.target_update}) is shorter than policy_update({self.policy_update})."
                          f"This may lead to unstable training.")
            
        # Exploration 관련 설정
        if self.use_epsilon: # epsilon 관련 설정 검토
            if self.epsilon_start < self.epsilon_end:
                raise ValueError(f"epsilon_start({self.epsilon_start}) should be bigger than epsilon_end({self.epsilon_end}).")
            if self.epsilon_start > 1.0:
                raise ValueError(f"epsilon_start({self.epsilon_start}) should not be bigger than 1.")
            if self.epsilon_end >= 1.0:
                raise ValueError(f"epsilon_end({self.epsilon_end}) should be smaller than 1.")
        
        if self.use_noisy: # Noisy linear layer를 사용하는 경우, epsilon 기법와 softmax 기법을 끄는 것이 권장됨.
            if self.use_epsilon:
                warnings.warn(f"If noisy linear layer is used, it is recommend not to use epsilon strategy.")
            if self.use_softmax:
                warnings.warn(f"If noisy linear layer is used, it is recommend not to use softmax strategy.")
        else: # Noisy linear layer를 사용하지 않는 경우, target_with_noisy를 꺼야함.
            if self.target_with_noisy:
                raise ValueError(f"Noisy linear layer is not used.(use_noisy=False)"
                                 f"target_with_noisy should be False.")
            
        # gamma 설정
        if self.gamma >= 1.0:
            raise ValueError(f"gamma({self.gamma}) should be smaller than 1.")
        
        # 하나의 map에 대해서 훈련할 경우 설정 검토
        if self.reset_only_start_pos and self.valid_map_num > 1:
            warnings.warn(
                f"Training is configured for a single map (reset_only_start_pos=True), "
                f"but validation is set to {self.valid_map_num} maps. "
                f"Consider setting valid_map_num to 1 for consistency."
            )
        
    def __post_init__(self):
        
        self._validate_configs() # 정상적인 config가 들어왔는지 검사
    

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
    
    # Reward function parameter (footprint-based)
    uncleaned_reward: float = 1.0
    cleaned_penalty: float = -0.1
    obstacle_penalty: float = -10.0
    turn_penalty: float = -0.1
    step_penalty: float = -0.01
    
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
        if self.chairs_per_table_max < self.chairs_per_table_min:
            raise ValueError(f"chairs_per_table_max({self.chairs_per_table_max}) is smaller than chairs_per_table_min({self.chairs_per_table_min}).")
        if self.small_obs_size_max < self.small_obs_size_min:
            raise ValueError(f"small_obs_size_max({self.small_obs_size_max}) is smaller than small_obs_size_min({self.small_obs_size_min}).")
        if self.small_obs_num_per_window_max < self.small_obs_num_per_window_min:
            raise ValueError(f"small_obs_num_per_window_max({self.small_obs_num_per_window_max}) is smaller than small_obs_size_min({self.small_obs_num_per_window_min}).")
        if self.target_coverage < self.final_coverage_thres:
            raise ValueError(f"target_coverage({self.target_coverage}) is smaller than final_coverage_thres({self.final_coverage_thres}).")
        
        # 2. 설정된 step 수 비교
        if self.max_steps < self.max_no_progress_steps:
            warnings.warn(f"It is recommended that max_steps({self.max_steps}) is bigger than max_no_progress_steps({self.max_no_progress_steps}).")
        if self.max_steps < self.max_no_progress_steps_final:
            warnings.warn(f"It is recommended that max_steps({self.max_steps}) is bigger than max_no_progress_steps_final({self.max_no_progress_steps_final}).")
        if self.max_no_progress_steps_final < self.max_no_progress_steps:
            warnings.warn(f"It is recommended that max_no_progress_steps_final({self.max_no_progress_steps_final}) is bigger than max_no_progress_steps({self.max_no_progress_steps}).")
        
        # 3. 의자 및 책상 다리 두께 검사
        if self.table_leg_size*2 >= self.min_table_size:
            raise ValueError(f"table_leg_size({self.table_leg_size}) is too big. It should be smaller than {self.min_table_size/2.0}.")
        if self.chair_leg_size*2 >= self.chair_size:
            raise ValueError(f"chair_leg_size({self.chair_leg_size}) is too big. It should be smaller than {self.chair_size/2.0}.")
        
        # 4. 맵 크기 대비 grid, window, robot 등의 size가 적절한지 검사
        min_map_dim = min(self.map_height, self.map_width)
        if self.grid_size > min_map_dim:
            raise ValueError(f"grid_size({self.grid_size}) is too big. It should be smaller than {min(self.map_height, self.map_width)}.")
        if self.window_size > min_map_dim:
            raise ValueError(f"window_size({self.window_size}) is too big. It should be smaller than {min(self.map_height, self.map_width)}.")
        if self.robot_size > min_map_dim:
            raise ValueError(f"robot_size({self.robot_size}) is too big. It should be smaller than {min(self.map_height, self.map_width)}.")
        if self.local_view > min_map_dim:
            raise ValueError(f"local_view({self.local_view}) is too big. It should be smaller than {min(self.map_height, self.map_width)}.")
        
        # 5. Reward 수치 경고
        if self.uncleaned_reward <= 0:
            warnings.warn(f"uncleaned_reward({self.uncleaned_reward}) is negative. It should be positive value.")
        if self.cleaned_penalty >= 0:
            warnings.warn(f"cleaned_penalty({self.cleaned_penalty}) is positive. It should be negative value.")
        if self.obstacle_penalty >= 0:
            warnings.warn(f"obstacle_penalty({self.obstacle_penalty}) is positive. It should be negative value.")
        if self.turn_penalty >= 0:
            warnings.warn(f"turn_penalty({self.turn_penalty}) is positive. It should be negative value.")
        if self.step_penalty >= 0:
            warnings.warn(f"step_penalty({self.step_penalty}) is positive. It should be negative value.")          
    
    def __post_init__(self):
        
        self._validate_configs() # 정상적인 config가 들어왔는지 검사
        
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