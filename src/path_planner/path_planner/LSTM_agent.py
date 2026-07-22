import re
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import wandb
import vessl
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os
from dataclasses import asdict
from itertools import product, cycle
from gymnasium.vector import SyncVectorEnv
import math

import glob
from IPython.display import display

# 앞서 정의한 클래스들을 임포트한다고 가정 (또는 같은 파일에 위치)
from path_planner.config import EnvConfig, TrainConfig, LOCAL_VIEW_DIM, TRAIN_MAP_FILE_REGEX, MAP_SAVE_DIR
from path_planner.map_layer import MapConfigSchema
from path_planner.environment import CoverageEnv, ACTION_NUM
from path_planner.LSTM_network import LSTMPolicyNetwork
from path_planner.utils.utils import get_device_info, set_torch_seed
from path_planner.utils.visualizer import display_image
from path_planner.utils.trajectory_metrics import GRID_RESOLUTION_M, LINEAR_VELOCITY

class CustomGymVecEnv(SyncVectorEnv):
    """
    Gym 내장 병렬 클래스를 그대로 상속받되, 
    우리가 원했던 '특정 환경만 골라 리셋하는 기능' 딱 하나만 추가합니다.
    """
    
    def __init__(self, env_fns, **kwargs):
        # SyncVectorEnv는 multiprocessing 관련 옵션(worker, shared_memory 등)을 받지 않으므로
        # env_fns만 부모 클래스에 전달합니다.
        super().__init__(env_fns, **kwargs)

    # 특정 환경(env_id)에서만 메소드를 실행할 때 사용하는 함수
    def call_at(self, method_name: str = "reset", env_id: int = 0, *args, **kwargs):
        # 1. 대상 환경 객체에 직접 접근
        target_env = self.envs[env_id]

        # 2. 메소드/속성 가져오기
        attr = getattr(target_env, method_name)

        # 3. 실행 후 결과 반환 (호출 가능하면 함수 실행, 변수면 그대로 반환)
        if callable(attr):
            return attr(*args, **kwargs)
        return attr
    
    def step(self, actions):
        self._autoreset_envs.fill(False)
        return super().step(actions)


# 경로의 품질 비교 함수
def is_better_path(cand, ref):
    """
    cand(후보): (cov, overlap, time)
    ref(기준):  (cov, overlap, time)
    cand가 ref보다 우수한경우 True 반환
    """
    cov_c, ov_c, t_c = cand
    cov_r, ov_r, t_r = ref

    # Candidate과 reference의 coverage 90% 미만이면 coverage 우선으로 경로 비교
    if cov_c < 0.9 or cov_r < 0.9: 
        return cov_c > cov_r

    # 둘 다 coverage 90% 이상인 경우, overlap rate으로 경로 비교. 차이가 cleaning time으로 비교
    else:
        if abs(ov_c - ov_r) < 10.0: # 10%p 차이 미만이면 청소시간 비교
            return t_c < t_r
        return ov_c < ov_r

class LSTMAgent:
    
    def __init__(self, args):
        
        self.args = args
        self.seed = args.seed
        self.model_name = os.path.join(args.model_dir, args.model_name)
        self.map_save_dir = args.map_save_dir
        
        # Device 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        get_device_info(self.device) # Device 정보 출력
        
        # torch의 seed 설정
        set_torch_seed(args.seed)
        
        # Environment 조성
        env_args = {
            k: v for k, v in vars(args).items() 
            if k in EnvConfig.__dataclass_fields__ and v is not None
        }
        self.env_cfg = EnvConfig(**env_args)
        # 우선, validation 및 test를 위한 environment만 조성
        if args.mode == 'train':
            self.test_env = CoverageEnv(self.env_cfg, seed=args.seed+100000) # 훈련용 environment와 다른 환경을 구성하기 위해서
        else:
            self.test_env =  CoverageEnv(self.env_cfg, seed=args.seed+200000)

        # Policy network 조성
        # kwargs 처리
        ang_diff_diridx = self.test_env.ang_diff_diridx
        if ang_diff_diridx is None:
            raise ValueError("PolicyNetwork를 초기화하려면 'ang_diff_diridx' 인자가 반드시 필요합니다")
        network_config = {
            'action_enc_dim': args.action_enc_dim,
            'locel_view_dim': LOCAL_VIEW_DIM,
            'stack_steps': self.env_cfg.stack_steps,
            'map_feat_dim': 512,
            'loc_vec_in_dim': ACTION_NUM,
            'vec_feat_dim': 64,
            'lstm_in_dim': 5,
            'lstm_in_prj_dim': args.lstm_in_prj_dim,
            'lstm_hid_dim': args.lstm_hid_dim,
            'ang_diff_diridx': ang_diff_diridx,
        }
        self.policy_net = LSTMPolicyNetwork(**network_config).to(self.device)
        
        # Train과 관련된 instance 변수는 mode가 train일 때만 생성
        if args.mode == 'train':
            
            # Train hyperparameter를 받음
            train_args = {
                k: v for k, v in vars(args).items() 
                if k in TrainConfig.__dataclass_fields__ and v is not None
            }
            self.train_cfg = TrainConfig(**train_args)
            
            # Training용 environment 목록: Batch size 만큼의 environment를 조성하고 training data를 한번에 생성
            num_envs = self.train_cfg.batch_size
            env_fns = [lambda i=i: CoverageEnv(self.env_cfg, seed=args.seed + i) for i in range(num_envs)]

            self.train_envs = CustomGymVecEnv(env_fns, autoreset_mode="Disabled")
            self.valid_env = CoverageEnv(self.env_cfg, seed=args.seed + 100000)

            self.train_map_sizes = None # 이미 저장된 train map을 사용할 때와 train map을 그때그때 생성할 때 모두 사용하는 변수
            if args.use_train_maps:
                # Training에 필요한 map 얻기
                self.train_maps = {} # Key: Map size, Value: {Key=level: Value=[map file list]}
                self.train_map_sizes = ['120x120', 'Cropped']
                
                train_map_dir = os.path.join(self.map_save_dir, 'train')
                
                # 각 map의 path를 level, map size별로 나누어서 저장
                for folder in self.train_map_sizes:
                    folder_path = os.path.join(train_map_dir, folder)
                    self.train_maps[folder] = {}
                    if os.path.exists(folder_path):
                        # Image 폴더는 제외하고 .npy 파일들만 정확하게 수집
                        npy_file_names = sorted([
                            f for f in os.listdir(folder_path) 
                            if f.endswith('.npy') and os.path.isfile(os.path.join(folder_path, f))
                        ])
                        
                        for file_name in npy_file_names:
                            full_path = os.path.join(folder_path, file_name)
                            match = re.search(TRAIN_MAP_FILE_REGEX, file_name)
                            gd = match.groupdict()
                            level = int(gd['level'])
                            if level not in self.train_maps[folder]:
                                self.train_maps[folder][level] = []
                            self.train_maps[folder][level].append(full_path)
                        
                        for level in self.train_maps[folder].keys():
                            self.train_maps[folder][level].sort()
                            self.train_maps[folder][level] = cycle(self.train_maps[folder][level])
            else:
                self.train_map_sizes = []
                window_size = self.env_cfg.map_cfg.window_size
                for mult in range(1, 5):
                    map_size = (window_size*mult, window_size*mult)
                    self.train_map_sizes.append(map_size)
                self.train_map_sizes.append('None')
                
            self.train_maps_sizes_iterator = iter(self.train_map_sizes) # 다음에 훈련시킬 map size를 결정하기 위한 iterator
            self.train_maps_sizes_idx = 0 # Map size iterator 현황을 저장하기 위한 변수
            self.train_maps_level_iterator = {} # Map의 level을 고루 변화시킬 때 사용하는 iterator
            for folder in self.train_map_sizes:
                self.train_maps_level_iterator[folder] = cycle(range(1, 5)) # 각 map size마다 level iterator를 생성
                
            valid_map_sizes = []
            window_size = self.env_cfg.map_cfg.window_size
            for mult in range(1, 5):
                map_size = (window_size*mult, window_size*mult)
                valid_map_sizes.append(map_size)
            valid_map_sizes.append('None')
            valid_levels = range(1, 5)
            self.valid_map_cfgs = list(product(valid_levels, valid_map_sizes))
                    
            # Training에 필요한 변수 설정
            self.total_steps = 0            # Training을 진행한 steps 수 (Warmup 과정 포함)
            self.start_episode = 1
            self.eps_num_per_map_size = 0   # 주어진 map size에서 training한 episode 개수
            self.map_size_idx = 0           # 현재 훈련시키고 있는 map size의 index를 저장하는 변수
            
            # Validation에 필요한 변수 설정
            self.max_coverage_mean = 0.0                # Validation을 수행했을 때 가장 높았던 coverage
            self.min_overlap_percent_mean = float('inf')   # Validation을 수행했을 때 가장 낮았던 overlap rate
            self.min_cleaning_time_mean = float('inf')  # Validation을 수행했을 때 가장 낮았던 cleaning time    
            self.best_traj_img = None                   # Validation을 수행했을 때 가장 coverage를 잘 했던 trajectory를 img로 저장 (RGB image)
            # self.num_heuristic = 0                      # Heuristic action이 선택된 횟수
            
            # Training을 위한 난수 생성기의 seed 설정
            self.train_rng = np.random.default_rng(seed=args.seed)
            
            # Training process를 볼 logger 설정: tb, wandb, vessl
            self._setup_logging()
            
            # Optimizer 설정
            if self.train_cfg.optimizer == 'sgd':
                self.optimizer = optim.SGD(self.policy_net.parameters(), lr=self.train_cfg.lr, momentum=self.train_cfg.momentum)
            elif self.train_cfg.optimizer == 'adam':
                self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.train_cfg.lr)
            else:
                raise ValueError(f"Unsupported optimizer type: {self.train_cfg.optimizer}")
            
        # Loading model
        self._load_model(args)
        
    
    def _setup_logging(self):
        """
        Training process를 지켜 볼 tool 설정: tb, wandb, vessl
        """
        
        import pytz
        import datetime
        
        # 현재 시간 얻기
        kst = pytz.timezone('Asia/Seoul')
        current_time = datetime.datetime.now(kst).strftime("%y%m%d_%H%M")
        
        # FIXME
        # Training 조건을 구별하기 위해 표시할 hyperparmeter 설정
        params_list_for_log = [
            'batch_size', 'lr', 'optimizer', 'momentum', 'action_enc_dim', 'lstm_in_prj_dim', 'lstm_hid_dim'    
        ]
        train_cfg_dict = asdict(self.train_cfg)
        params_config = {k: v for k, v in train_cfg_dict.items() if k in params_list_for_log}
        
        # TensorBoard 설정
        self.tb_writer = None
        if self.args.use_tb:
            tb_save_dir = os.path.join(self.args.tb_save_dir, current_time)
            if not os.path.exists(tb_save_dir):
                os.makedirs(tb_save_dir)
            self.tb_writer = SummaryWriter(tb_save_dir)
            
        # wandb 설정
        self.wandb_run = None
        if self.args.use_wandb:
            self.wandb_run = wandb.init(
                entity="lg-robot-cleaner",
                project="Robot_vacuum_ReDQN",
                config=params_config,
                name=f"{current_time}_{self.args.model_name}_training"
            )
            
        # vessl 설정
        if self.args.use_vessl:
            vessl.init(
                organization="snu-eng-gtx1080", 
                project="lg-robot-ReDQN", 
                hp=params_config,
                # name=f"{current_time}_{args.model_name}_training"                  
            )
        # ---------------------------------------------------------
        

    def _load_model(self, args):
        """
        Train:     
        - model_name: 최종적으로 훈련시킨 모델의 이름
        - checkpoint: 훈련 도중 저장되는 중간 결과물들은 (모델 이름)_checkpoints/ 폴더에 저장됨.
        - Checkpoint 파일명: (모델 이름)_(에피소드 번호).pth
        - 모델 저장 경로 설정: 모델은 models/ 디렉토리에 저장됨.
        - 하위 폴더는 checkpoint를 모으는 폴더로 training 과정에 저장됨.
        - 훈련이 다 된 모델은 models/ 폴더 바로 아래에 있음.
        
        Test:
        - model_name: Test할 모델 이름
        """
        model_dir = args.model_dir # Model이 담긴 폴더: .../models
        model_name_base, _ = os.path.splitext(args.model_name) # 확장자 '.pth'를 제거한 model_name
        checkpoint_dir = os.path.join(model_dir, model_name_base + "_checkpoints") # Checkpoint 저장 폴더
        
        # 폴더 생성
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if args.mode == 'train':
            
            model_path = None
            is_checkpoint = False
            
            # 1. Loading할 model 선택
            if args.pre_model_name: # 사전 학습 완료된 모델부터 이어서 훈련하는 경우
                print("Use pre-trained model...")
                model_path = os.path.join(model_dir, args.pre_model_name)
                print(f"Loading pre-trained model: {model_path}")
            else: # 사전 학습 완료된 모델이 없는 경우(처음부터 훈련하는 경우 + checkpoint에서 이어서 훈련하는 경우)
                checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
                if checkpoints:
                    print("Checkpoint directory already exists. Continue training...")
                    model_path = max(checkpoints, key=os.path.getctime) # 가장 최근 checkpoint 파일부터 훈련 재개
                    is_checkpoint = True
                    print(f"Loading latest checkpoint: {model_path}")
                else:
                    print("No checkpoint files found!")
                    return
                
            # 2. Model loading
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            data = torch.load(model_path, map_location=self.device, weights_only=False)
            self.policy_net.load_state_dict(data['model_state_dict'])
            self.max_coverage_mean = data.get('max_coverage_mean', 0)
            self.min_overlap_percent_mean = data.get('min_overlap_percent_mean', float('inf'))
            self.min_cleaning_time_mean = data.get('min_cleaning_time_mean', float('inf'))
            
            if is_checkpoint: # Checkpoint를 사용하는 경우 optimizer와 step 수도 추가로 load
                # Optimizer loading
                try:
                    self.optimizer.load_state_dict(data['optimizer_state_dict']) # Optimizer도 checkpoint와 똑같이 유지
                except KeyError as e:
                    print(f"Optimizer state mismatch. Skipping: {e}")
              
                # 추가적인 변수 loading
                self.total_steps = data['total_steps']
                self.map_size_idx = data['map_size_idx']
                self.start_episode = data['episode'] + 1
                self.eps_num_per_map_size = data['eps_num_per_map_size']
                
                # map_size_iterator원래 형태로
                if self.map_size_idx > 0:
                    for _ in range(self.map_size_idx-1):
                        try:
                            next(self.train_maps_sizes_iterator)
                        except StopIteration: # 마지막 map size인 경우, 이전에 설정한 map size를 그대로 사용
                            pass
                
                print(f"Resumed training from episode {data['episode']}")
                
            # best_traj_img loading
            traj_img_dir = checkpoint_dir
            if args.pre_model_name:
                pre_model_name_base, _ = os.path.splitext(args.pre_model_name) # 확장자 '.pth'를 제거한 model_name
                traj_img_dir = os.path.join(model_dir, pre_model_name_base + "_checkpoints")
            
            best_traj_img_path = os.path.join(traj_img_dir, args.best_traj_img_name)
            if os.path.isfile(best_traj_img_path):
                img = cv2.imread(best_traj_img_path) # BGR image
                self.best_traj_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB 변환
                
        elif args.mode == 'test':
            
            test_model = os.path.join(model_dir, args.model_name)
            if not os.path.exists(test_model):
                raise FileNotFoundError(f"Test model file not found: {test_model}")
            checkpoint = torch.load(test_model, map_location=self.device, weights_only=False)
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            print(f"Test model loaded: {test_model}")
    

    def _save_model(self, mode='model', info=None):
        
        model_dir = self.args.model_dir # Model이 담긴 폴더명: .../models
        model_name_base, _ = os.path.splitext(self.args.model_name)
        checkpoint_dir = os.path.join(self.args.model_dir, model_name_base + "_checkpoints") # Checkpoint 저장 폴더
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        save_path = None
        save_data = {
            'model_state_dict': self.policy_net.state_dict(),
            'max_coverage_mean': self.max_coverage_mean,
            'min_overlap_percent_mean': self.min_overlap_percent_mean,
            'min_cleaning_time_mean': self.min_cleaning_time_mean,
        }

        if mode == 'model': # model을 /models에 저장
            save_path = os.path.join(self.args.model_dir, self.args.model_name) # model 저장 경로
            
            # Checkpoint 파일에 가장 coverage가 잘 수행된 trajectory 그림을 저장
            if self.best_traj_img is not None:
                img_file_path = os.path.join(checkpoint_dir, self.args.best_traj_img_name)
                img_bgr = cv2.cvtColor(self.best_traj_img, cv2.COLOR_RGB2BGR) # OpenCV 저장용: (RGB -> BGR)
                cv2.imwrite(img_file_path, img_bgr)
                
        elif mode == 'checkpoint': # checkpoint 폴더에 저장
            
            assert info is not None
            episode = info['episode']
            save_path = os.path.join(checkpoint_dir, model_name_base + f"_{episode}.pth")
            save_data.update({
                'episode': episode,
                'total_steps': self.total_steps,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'map_size_idx': self.map_size_idx,
                'eps_num_per_map_size': self.eps_num_per_map_size,
            })
            
        else:
            raise ValueError(f"Unsupported mode: {mode}. Expected 'model' or 'checkpoint'.")

        torch.save(save_data, save_path)    
        
    def _pre_process_obs(self, obs_list, local_view_dim=LOCAL_VIEW_DIM) -> dict:
        
        """
        여러 환경의 obs들이 담긴 obs_list를 받아 병렬적으로 전처리하여 tensor로 변환.
        local_view_map은 적절한 크기로 resize만 진행.
        """
        # 1. CPU 리스트 데이터를 빠르게 PyTorch GPU 텐서로 통합 (연산 그래프 분리 .detach())
        if isinstance(obs_list, dict):
            maps = torch.as_tensor(obs_list['map'], device=self.device).float().detach() # Shape: (B, C, H, W) or (C, H, W)
            loc_vecs = torch.as_tensor(obs_list['loc_vec'], device=self.device).float().detach() # Shape: (B, vector_dim) or (vector_dim,)
            glob_vecs = torch.as_tensor(obs_list['glob_vec'], device=self.device).float().detach() # Shape: (B, vector_dim) or (vector_dim,)
            action_masks = torch.as_tensor(obs_list['action_mask'], device=self.device).float().detach() # Shape: (B, action_num) or (action_num,)

            if maps.ndim == 3: # local map view가 (C, H, W)로 들어오는 경우
                maps = maps.unsqueeze(0)
                loc_vecs = loc_vecs.unsqueeze(0)
                glob_vecs = glob_vecs.unsqueeze(0)
                action_masks = action_masks.unsqueeze(0)

        elif isinstance(obs_list, (list, tuple)):
            maps = (torch.stack([torch.as_tensor(o["map"], device=self.device) for o in obs_list], dim=0).float().detach()) # Shape: (B, C, H, W)
            loc_vecs = (torch.stack([torch.as_tensor(o["loc_vec"], device=self.device) for o in obs_list], dim=0).float().detach()) # Shape: (B, vector_dim)
            glob_vecs = (torch.stack([torch.as_tensor(o["glob_vec"], device=self.device) for o in obs_list], dim=0).float().detach()) # Shape: (B, vector_dim)
            action_masks = (torch.stack([torch.as_tensor(o["action_mask"], device=self.device) for o in obs_list], dim=0).float().detach()) # Shape: (B, action_num)
        else:
            raise TypeError(f"Unexpected type for obs_list: {type(obs_list)}. Expected dict or list.")
        
        # 2. local_view_map resize
        B, C, H, W = maps.shape
        target_dim = local_view_dim
        if H != target_dim or W != target_dim:
            total_patch = F.interpolate(maps, size=(target_dim, target_dim), mode='area')
        else:
            total_patch = maps

        return {
            "map": total_patch,           # Shape: (B, C, local_view_dim, local_view_dim)
            "loc_vec": loc_vecs,          # Shape: (B, loc_vec_dim)
            "glob_vec": glob_vecs,        # Shape: (B, glob_vec_dim)
            "action_mask": action_masks   # Shape: (B, action_num)
        }
        
    
    def _validation(self, episode: int):
        
        self.policy_net.eval() # eval mode로 전환
        coverage = []; overlap_percent = []; cleaning_time = []
        max_coverage = 0.0; min_overlap_percent = float('inf'); min_cleaning_time = float('inf')
        coverage_threshold = 0.90; overlap_percent_gap = 0.10
        coverage_mean = 0.0; overlap_percent_mean = 0.0; cleaning_time_mean = 0.0
        best_traj_img = None # 가장 성능이 좋았던 map에서의 trajectory를 보여줌
        
        # Model의 성능 평가
        for level, map_sizes in self.valid_map_cfgs:
            
            if map_sizes != 'None':
                H, W = map_sizes
            else:
                H, W = None, None
            map_config = MapConfigSchema(level=level, H=H, W=W)
            
            for _ in range(self.train_cfg.valid_map_num):
                
                # Validation 환경을 초기화
                if self.train_cfg.reset_only_start_pos:
                    obs, _ = self.valid_env.reset()
                else:
                    reset_info = self.valid_env.reset(map_config=map_config)
                    if reset_info:
                        obs, _ = reset_info
                    else:
                        break # 다음 map_config를 이용
                    
                for start_num in range(self.train_cfg.valid_start_point_num):
                    
                    # start_num == 0인 경우 map을 초기화하면서 시작 지점도 초기화가 되었으므로 reset을 실행하지 않음.
                    if start_num != 0:
                        obs, _ = self.valid_env.reset() # 시작 지점 초기화
                    cur_coverage, cur_overlap_percent, cur_cleaning_time = self._test_one_map(self.valid_env, obs, debug=False) # Coverage 성능을 평가
                    
                    coverage.append(cur_coverage)
                    overlap_percent.append(cur_overlap_percent)
                    cleaning_time.append(cur_cleaning_time)

                    # Coverage를 어느 정도 달성한 경우, overlap_percent과 cleaning_time을 중심으로 best_traj의 후보로 고려
                    if cur_coverage >= coverage_threshold:
                        
                        # max_coverage 최신화
                        if max_coverage < cur_coverage:
                            max_coverage = cur_coverage
                        
                        # Overlap 비율이 감소했을 때: best_traj의 후보로 고려
                        if cur_overlap_percent < min_overlap_percent:
                            min_overlap_percent = cur_overlap_percent
                            min_cleaning_time = cur_cleaning_time
                            best_traj_img = self.valid_env.get_visualized_img()
                        
                        # Overlap 비율이 최소 overlap 비율과 크지 않을 경우, cleaning_time이 가장 빠른 trajectory를 선택
                        elif cur_overlap_percent <= min_overlap_percent + overlap_percent_gap:
                            if cur_cleaning_time < min_cleaning_time:
                                min_cleaning_time = cur_cleaning_time
                                best_traj_img = self.valid_env.get_visualized_img() # best trajectory의 이미지 저장

                    # Coverage 달성도가 미달인 경우, coverage가 높은 경로를 best_traj의 후보로 고려
                    elif max_coverage < cur_coverage:
                        max_coverage = cur_coverage
                        best_traj_img = self.valid_env.get_visualized_img()
                    
        coverage_mean = np.mean(coverage) if coverage else 0.0
        overlap_percent_mean = np.mean(overlap_percent) if overlap_percent else 0.0
        cleaning_time_mean = np.mean(cleaning_time) if cleaning_time else 0.0
        
        # 이전 model의 종합적 성능(Coverage, Overlap rate, Cleaning time 모두 고려)보다 더 좋으면 model을 저장
        if coverage_mean >= coverage_threshold: # Coverage가 일정 수준 이상인 경우: Overlap, Cleaning time을 고려
            
            # self.max_coverage_mean 최신화
            if coverage_mean > self.max_coverage_mean:
                self.max_coverage_mean = coverage_mean
            
            # Overlap 비율이 감소했을 때: best_model의 후보로 고려   
            if overlap_percent_mean < self.min_overlap_percent_mean:
                self.min_overlap_percent_mean = overlap_percent_mean
                self.min_cleaning_time_mean = cleaning_time_mean
                if best_traj_img is not None:
                    self.best_traj_img = best_traj_img
                self._save_model(mode='model') # 성능이 가장 좋았던 model을 저장
            
            elif overlap_percent_mean <= self.min_overlap_percent_mean + overlap_percent_gap:
                if cleaning_time_mean < self.min_cleaning_time_mean:
                    self.min_cleaning_time_mean = cleaning_time_mean
                    if best_traj_img is not None:
                        self.best_traj_img = best_traj_img
                    self._save_model(mode='model') # 성능이 가장 좋았던 model을 저장
        
        elif coverage_mean > self.max_coverage_mean:
            self.max_coverage_mean = coverage_mean
            if best_traj_img is not None:
                self.best_traj_img = best_traj_img
            self._save_model(mode='model') # 성능이 가장 좋았던 model을 저장
        
        # Validation 결과를 tensorboard와 wandb에 기록
        if self.tb_writer:
            self.tb_writer.add_scalar('Validation/Coverage_mean', coverage_mean, episode)
        if self.wandb_run:
            self.wandb_run.log({'Validation/Coverage_mean': coverage_mean,
                                'Validation/Overlap Rate Mean': overlap_percent_mean,
                                'Validation/Cleaning Time Mean': cleaning_time_mean,
                                'Validation/Best_path': wandb.Image(best_traj_img)}, step=self.total_steps)
        if self.args.use_vessl:
            vessl.log(step=episode, payload={'Validation/Coverage_mean': coverage_mean,
                                             'Validation/Overlap Rate Mean': overlap_percent_mean,
                                             'Validation/Cleaning Time Mean': cleaning_time_mean,
                                             'Validation/Best_path': vessl.Image(best_traj_img)})
        
        # Validation 결과 출력
        print(f"[Validation] Episode {episode}: \n"
              f"\tCoverage Mean = {coverage_mean*100:.2f}%\n"
              f"\tOverlap Rate Mean = {overlap_percent_mean:.2f}%\n"
              f"\tCleaning Time Mean = {cleaning_time_mean:.2f} min")
            
        self.policy_net.train() # train mode로 전환
    
    def _test_one_map(self, env: CoverageEnv, obs: dict, debug: bool = False) -> tuple[float, float, float]:
        
        # 2. Path 생성: LSTM network에 통과시켜서 각 action의 확률값을 얻고 map 상의 전체 경로 생성
        done = False
        debug_skip_count = 0
        
        h_t = torch.zeros(1, self.policy_net.lstm_hid_dim, device=self.device)
        c_t = torch.zeros(1, self.policy_net.lstm_hid_dim, device=self.device)
        
        processed_obs = self._pre_process_obs(obs, local_view_dim=LOCAL_VIEW_DIM)

        # LSTM의 hidden state와 cell state, processed_obs를 복원하기 위한 list
        state_history = []
        
        # [1] 디버그 모드일 때 사용할 도화지(fig)를 미리 딱 한 번만 만듦
        if debug:
            fig, axes = plt.subplots(2, 1, figsize=(15, 20))
            # 초기 이미지
            init_traj_img = env.get_visualized_img(img_choice='traj')
            init_obs_img = env.get_visualized_img(img_choice='obs')
            # init_pro_obs_img = env.get_visualized_img(img_choice='obs', preprocessor=self._pre_process_obs)
            
            # 초기 빈 이미지 설치
            im_traj = axes[0].imshow(init_traj_img)
            im_obs = axes[1].imshow(init_obs_img)
            # im_pro_obs = axes[2].imshow(init_pro_obs_img)
            text_traj = axes[0].text(0.5, 0, "", transform=axes[0].transAxes, ha="center", fontsize=15, color='black')
            axes[0].set_title("Trajectory", fontsize=20)
            axes[1].set_title("Observation", fontsize=20)
            # axes[2].set_title("Processed Observation", fontsize=20)
            for ax in axes: ax.axis('off')
            plt.tight_layout()
            
            # Jupyter용 디스플레이 핸들을 생성 (이걸 통해 이미지만 쏙 바꿉니다)
            display_handle = display(fig, display_id=True)
            plt.close(fig) # 별도의 정적 출력이 생기지 않도록 닫기
        
        # LSTM network에 통과시키면서 action을 얻으면서 경로 생성. Action 확률을 저장.
        while not done:
            
            # Network에 통과시켜 각 action이 선택될 확률을 출력
            loc_map_data = processed_obs["map"]
            loc_vec_data = processed_obs["loc_vec"]
            glob_vec_data = processed_obs["glob_vec"]
            action_mask = processed_obs["action_mask"]

            state_history.append({'h_t': h_t.clone(), 'c_t': c_t.clone()})

            with torch.no_grad():
                action_probs, h_t, c_t = self.policy_net(loc_map_data, loc_vec_data, glob_vec_data, h_t, c_t)
                
                # Action이 나올 확률을 가지고 action 선택. Action 선택 확률을 저장. 장애물에 부딪히는 action은 제외
                masked_probs = action_probs * action_mask # Shape: action_probs (B, action_num) / action_mask (B, action_num)
                masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-8)
            
            action = torch.argmax(masked_probs).item()
            
            # Action 수행: 다음 obs 얻기
            next_obs, reward, terminated, truncated, info = env.step(action)
            processed_obs = self._pre_process_obs(next_obs, local_view_dim=LOCAL_VIEW_DIM)
            
            # 각 환경별 종료 여부 판단
            done = terminated or truncated
            
            if debug:
                
                if debug_skip_count > 0:
                    debug_skip_count -= 1
                else:
                    
                    # 각 action이 선택될 확률 및 선택된 action 표시
                    prob_np = masked_probs.squeeze().cpu().numpy()
                    probs_str = ", ".join([f"{prob:.3f}" for prob in prob_np])
                    action_info = (f"[Selected action]: {action}\n"
                                   f"[Action probs] [{probs_str}]")
                    
                    # Map 시각화
                    traj_img = env.get_visualized_img(img_choice='traj')
                    obs_img = env.get_visualized_img(img_choice='obs')
                    # pro_obs_img = env.get_visualized_img(img_choice='obs', preprocessor=self._pre_process_obs)
                    
                    im_traj.set_data(traj_img)
                    im_obs.set_data(obs_img)
                    # im_pro_obs.set_data(pro_obs_img)
                    text_traj.set_text(action_info)
                    
                    # [4] 화면 갱신 (도화지 위치는 그대로, 내용물만 부드럽게 변경)
                    display_handle.update(fig)
                    
                    user_val = input("Next step: [Enter] | Auto: [Number: If want to backstep, input negative number] | Exit: [q] >> ")
                    user_val = user_val.strip()
                    
                    if user_val.lower() == 'q':
                        break
                    
                    try:
                        val = int(user_val)
                        if val < 0:
                            backstep_num = abs(val)
                            last_obs = env.backstep(backstep_num)
                            processed_obs = self._pre_process_obs(last_obs)

                            pop_count = min(backstep_num, len(state_history))
                            for _ in range(pop_count):
                                target_state = state_history.pop()
                            h_t = target_state['h_t']
                            c_t = target_state['c_t']
                            continue
                        elif val > 0:
                            debug_skip_count = int(user_val) - 1
                        else:
                            continue
                    except ValueError:
                        print("숫자 또는 'q'를 입력해주세요.")
        
        coverage = env.coverage
        overlap_percent = env.overlap_percent
        cleaning_time = env.cleaning_time
        
        return coverage, overlap_percent, cleaning_time
    
    def train(self):
        
        assert self.args.mode == 'train'

        torch.autograd.set_detect_anomaly(True)

        # Policy network를 train mode로 설정
        self.policy_net.train()        

        no_more_map_size = False # 더 바꿀 map size가 없는 경우: map size 변경 logic을 건너뜀.
        try:
            curr_map_size = next(self.train_maps_sizes_iterator)
        except StopIteration: # 마지막 map size인 경우, 이전에 설정한 map size를 그대로 사용
            no_more_map_size = True
            curr_map_size = self.train_map_sizes[-1]
        
        print(f"[Initial map size] {curr_map_size}.")
        
        # Episode 동안 map을 변경하지 않음. 고정된 map에서 경로 생성 수 policy update을 반복
        for episode in range(self.start_episode, self.train_cfg.max_episodes+1):

            print(f"[Episode {episode}] Starting training...")
            eps_path_rwd_mean = 0 # Episode의 훈련 상태를 표시할 reward
            
            # Seed: 처음 environment를 reset할 때만 
            if episode == self.start_episode:
                seed = self.seed
            else:
                seed = None
            
            obs_list = []
            info_list = []
            
            # Map 설정
            for env_id in range(self.train_envs.num_envs):
                reset_info = None
                while not reset_info:
                    level = next(self.train_maps_level_iterator[curr_map_size])
                    if self.train_cfg.use_train_maps:
                        map_file_path = next(self.train_maps[curr_map_size][level])
                        map_config = MapConfigSchema(file_path=map_file_path)
                    else:
                        H = None; W = None
                        if curr_map_size != 'None':
                            H, W = curr_map_size
                        map_config = MapConfigSchema(level=level, H=H, W=W)
                    reset_info = self.train_envs.call_at(method_name="reset", env_id=env_id, seed=seed, map_config=map_config)
                obs, info = reset_info
                obs_list.append(obs)
                info_list.append(info)
            processed_obs = self._pre_process_obs(obs_list, local_view_dim=LOCAL_VIEW_DIM) # Map data를 resize, Observation data를 얻음
            
            steps = 0 # Episode 내에서의 step 수
            new_maps = True

            # 고정된 map에 대해서 경로 생성 최대 횟수를 넘으면 다른 map으로 바꿈. (while문을 빠져나옴)
            while steps < self.train_cfg.max_step_per_eps:
                
                # 1. Map 상태 reset: 
                # Map을 방금 새로 생성한 경우, reset이 이미 일어났으므로 reset을 생략.
                # 기존 map을 사용하는 경우, map 상태를 reset.
                if not new_maps:
                    obs_list = []
                    info_list = []
                    for env_id in range(self.train_envs.num_envs):
                        obs, info = self.train_envs.call_at(method_name="reset", env_id=env_id, seed=seed, map_config=None)
                        obs_list.append(obs)
                        info_list.append(info)
                    processed_obs = self._pre_process_obs(obs_list, local_view_dim=LOCAL_VIEW_DIM)
                
                new_maps = False
                    
                # 2. Path 생성: LSTM network에 통과시켜서 각 action의 확률값을 얻고 map 상의 전체 경로 생성
                done = False
                
                # Hidden state와 cell state 초기화
                batch_size = len(obs_list)
                h_t = torch.zeros(batch_size, self.policy_net.lstm_hid_dim, device=self.device)
                c_t = torch.zeros(batch_size, self.policy_net.lstm_hid_dim, device=self.device)
                
                # log probability 저장ㅌ
                saved_log_probs = []
                prob_masks = [] # 종료된 environment에서 계산된 log probability가 gradient update을 하지 않도록 하기 위한 mask
                saved_entropies = []
                curr_act_prob = np.ones(batch_size, dtype=bool) # 현재 아직 완료되지 않은 environment

                # LSTM network에 통과시키면서 action을 얻으면서 경로 생성. Action 확률을 저장.
                while not done:

                    # Hidden state와 cell state를 detach하여 gradient가 LSTM step을 넘어서 전달되는 것을 방지.
                    h_t = h_t.detach()
                    c_t = c_t.detach()
                    
                    # Hidden state와 cell state를 detach하여 gradient가 LSTM step을 넘어서 전달되는 것을 방지.
                    h_t = h_t.detach()
                    c_t = c_t.detach()
                    
                    # Network에 통과시켜 각 action이 선택될 확률을 출력
                    loc_map_data = processed_obs["map"]
                    loc_vec_data = processed_obs["loc_vec"]
                    glob_vec_data = processed_obs["glob_vec"]
                    action_mask = processed_obs["action_mask"]
                    action_probs, h_t, c_t = self.policy_net(loc_map_data, loc_vec_data, glob_vec_data, h_t, c_t)
                    
                    # Action이 나올 확률을 가지고 action 선택. Action 선택 확률을 저장. 장애물에 부딪히는 action은 제외
                    masked_probs = action_probs * action_mask # Shape: action_probs (B, action_num) / action_mask (B, action_num)
                    masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-8)
                    dist = torch.distributions.Categorical(probs=masked_probs)
                    actions_tensor = dist.sample() # Action을 확률을 기반으로 선택. Shape: (B,)
                    actions = actions_tensor.cpu().numpy() # step()을 위해 numpy 형식으로 변환
                    
                    # Entropy 계산: wandb logging용
                    entropy = dist.entropy().detach()
                    saved_entropies.append(entropy)
                    
                    # Action 수행: 다음 obs 얻기
                    next_obs_list, rewards, terminateds, truncateds, info = self.train_envs.step(actions)
                    processed_obs = self._pre_process_obs(next_obs_list, local_view_dim=LOCAL_VIEW_DIM)
                    
                    # 확률 미분 계산을 위해 log probability를 저장
                    log_prob = dist.log_prob(actions_tensor) # 각 action이 선택될 확률의 log probability. Shape: (B,)
                    saved_log_probs.append(log_prob)
                    prob_masks.append(torch.as_tensor(curr_act_prob, device=self.device, dtype=torch.float32))
                    
                    # 각 환경별 종료 여부 판단
                    dones = terminateds | truncateds # Shape: (B,)
                    done = dones.all()
                    curr_act_prob = curr_act_prob & ~dones # 이미 종료된 environment는 0으로 표시 -> Gradient 계산 시 probability가 영향을 주지 않도록 함.

                ##############################################################
                # [PATH REWARD FUNCTION]
                ##############################################################
                # 각 environment에서 얻은 path의 coverage, overlap rate, cleaning time 계산
                coverages = np.array(self.train_envs.get_attr("coverage")) # Shape: (B,)
                overlap_rates = np.array(self.train_envs.get_attr("overlap_percent")) / 100 # Shape: (B,)
                cleaning_times = np.array(self.train_envs.get_attr("cleaning_time")) # Shape: (B,)
                
                # cleaning time의 기준 설정: Total step을 직진으로 이동했을 때 걸린 시간을 기준으로 삼음.
                total_steps = np.array(self.train_envs.get_attr("steps")) # Shape: (B,)
                straight_cleaning_times = total_steps * GRID_RESOLUTION_M / LINEAR_VELOCITY / 60.0
                cleaning_times_rel = cleaning_times / (straight_cleaning_times + 1e-8) # 경로 길이에 대한 cleaning time
                
                # Tensor 변환
                cov_tensor = torch.tensor(coverages, device=self.device, dtype=torch.float32).detach()
                overlap_tensor = torch.tensor(overlap_rates, device=self.device, dtype=torch.float32).detach()
                time_tensor = torch.tensor(cleaning_times_rel, device=self.device, dtype=torch.float32).detach()
                
                # Reward 계산
                cov_rwd_max = 0.5 # Coverage reward의 max 수치
                cov_thres = 0.9 # Coverage를 주로 고려하는 coverage 수치 임계값
                above_cov_thres_mask = (cov_tensor >= cov_thres).float()
                cov_reward = torch.clamp(cov_tensor, min=0.0, max=cov_thres) * (cov_rwd_max/cov_thres) # Coverage가 cov_thres 미만일 때는 coverage만 reward에 적용. 만점은 cov_rwd_max 
                overlap_reward = (1.0 - overlap_tensor) * (1.0 - cov_rwd_max) # Overlap은 coverage가 cov_thres 이상일 때만 고려. 만점은 1.0 - cov_rwd_max
                time_penalty = torch.clamp((time_tensor - 1.0) * 0.05, min=0.0, max=0.1) # 직진 경로 시간과 비교해서 얼마나 차이나는지에 따라 penalty 부여. 직진보다 세 배 더 걸릴 때 최대 감점.
                path_reward = cov_reward + (overlap_reward - time_penalty) * above_cov_thres_mask
                
                advantage = (path_reward - path_reward.mean()).detach() # Shape: (B,)
                ##############################################################
                
                # loss 계산 및 parameter update
                log_probs_tensor = torch.stack(saved_log_probs) # Shape: (T, B) (T: time steps)
                prob_masks_tensor = torch.stack(prob_masks) # Shape: (T, B)
                # log probability를 합하여 전체 경로의 probability 계산 
                # (완료된 environment에서 얻은 probability는 계수 0을 곱하여 gradient 전달이 안 되도록 함.)
                sum_log_probs = (log_probs_tensor * prob_masks_tensor).sum(dim=0) # Shape: (B,) 
                policy_loss = -(sum_log_probs * advantage) # 음수를 붙여서 path_reward가 증가하는 방향으로 update 되도록 함.
                
                loss = policy_loss.mean() # Batch 평균 loss
                self.optimizer.zero_grad() # Gradient 초기화
                loss.backward() # Gradient 계산
                grad_norm_before = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0) # Gradient clipping
                self.optimizer.step() # Parameter update
                
                # Entropy 통계
                entropies_tensor = torch.stack(saved_entropies) # Shape: (T, B)
                mean_entropy = (entropies_tensor * prob_masks_tensor).sum() / (prob_masks_tensor.sum() + 1e-8)
                
                # Path reward의 평균
                path_reward_mean = path_reward.mean().item()
                eps_path_rwd_mean = path_reward_mean # Episode path reward update (가장 최근 model의 path reward를 출력할 것임.)
                
                log_dict = {
                    "train/episode": episode,
                    "train/step": steps,
                    "train/loss": loss.item(),
                    "train/grad_norm_raw": grad_norm_before.item() if isinstance(grad_norm_before, torch.Tensor) else grad_norm_before,
                    "train/entropy_mean": mean_entropy.item(), # 💡 수집된 평균 엔트로피
                    "train/map_steps_mean": total_steps.mean(),
                    
                    # Reward 지표 (Mean, Max, Min)
                    "train/path_reward_mean": path_reward_mean,
                    "train/path_reward_max": path_reward.max().item(),
                    "train/path_reward_min": path_reward.min().item(),
                    
                    # Coverage 지표 (Mean, Max, Min)
                    "train/coverage_mean": cov_tensor.mean().item(),
                    "train/coverage_max": cov_tensor.max().item(),
                    "train/coverage_min": cov_tensor.min().item(),
                    
                    # Overlap 지표 (Mean, Max, Min)
                    "train/overlap_mean": overlap_tensor.mean().item(),
                    "train/overlap_max": overlap_tensor.max().item(),
                    "train/overlap_min": overlap_tensor.min().item(),
                    
                    # Time 지표
                    "train/time_mean": cleaning_times.mean(),
                    "train/time_ratio_mean": time_tensor.mean().item(),
                }
                
                # wandb가 초기화되어 있다면 기록
                if self.args.use_wandb: # 혹은 관련 조건문
                    wandb.log(log_dict)
                
                # 훈련되고 있음을 확인하기 위해 steps 현황을 출력
                # print(f"[Step {self.total_steps+1:04d}] Path reward (Mean): {log_dict['train/path_reward_mean']:.3f}")
                
                # Steps: Parameter update 횟수
                steps += 1
                self.total_steps += 1
                
            # 주어진 map size에서 training한 episode 횟수 증가
            self.eps_num_per_map_size += 1
            
            # 좋은 경로 생성에 성공했거나 episode 횟수가 한계치를 넘어서면 map size를 바꿈
            change_map_size = False
            if not no_more_map_size: # 변경할 map_size가 더 없는 경우에는 change_map_size를 무조건 False로 설정되도록 함.
                if path_reward_mean >= self.train_cfg.path_reward_thres:
                    change_map_size = True
                    print(f"[Map Scale Up] Achieved path reward {path_reward_mean:.3f} at map size {curr_map_size}. "
                          f"Exceeded the threshold {self.train_cfg.path_reward_thres}.")
                elif self.eps_num_per_map_size > self.train_cfg.max_eps_per_map_size:
                    change_map_size = True
                    print(f"[Map Scale Up] Reached maximum episodes ({self.eps_num_per_map_size}) for map size {curr_map_size}.")
                
            if change_map_size:
                try:
                    curr_map_size = next(self.train_maps_sizes_iterator)
                except StopIteration: # 마지막 map size인 경우, 이전에 설정한 map size를 그대로 사용
                    print("[Map Scale Up] No more map sizes available. Maintaining the current size.")
                    curr_map_size = self.train_map_sizes[-1]
                    no_more_map_size = True
                else:
                    print(f"[Map Scale Up] Successfully changed map size to {curr_map_size}.")
                    self.map_size_idx += 1
                    self.eps_num_per_map_size = 0
            
            # 한 map에 대해서 훈련이 끝나면 checkpoint 저장 및 validation 수행

            # Episode 훈련 결과 출력
            print(f"\tEpisode reward: {eps_path_rwd_mean}")

            # Checkpoint 저장
            if episode % self.train_cfg.ckp_freq == 0:
                checkpoint_info = {"episode": episode}
                self._save_model(mode='checkpoint', info=checkpoint_info)
                
            # Validation 수행
            if episode % self.train_cfg.valid_freq == 0:
                print("Validation...")
                self._validation(episode)
            
            # Map 기록 저장
            if episode % 5 == 0:
                map_img = self.train_envs.call_at(method_name="get_visualized_img", env_id=0, img_choice="traj")
                if self.tb_writer:
                    self.tb_writer.add_image("Visualization/Robot_path", map_img, episode, dataformats="HWC")
                if self.wandb_run:
                    self.wandb_run.log({"Visualization/Robot_path": wandb.Image(map_img)}, step=self.total_steps)
                if self.args.use_vessl:
                    vessl.log(step=episode, payload={"Visualization/Robot_path": vessl.Image(map_img)})

    def test(self, use_maps_folder: bool=True):

        self.policy_net.eval() # eval mode로 전환
        total_coverage = []; total_overlap_percent = []; total_cleaning_time = []
        computation_time = []
        reset_seed = self.seed
        
        condition_results = {} # Map condition 별로 test 결과를 저장하기 위한 dictionary
        
        # Level별 Best/Worst 경로 기록용 딕셔너리
        visualized_maps = {}
        
        maps_folder = os.path.join(self.map_save_dir, 'test')  # Map을 저장한 폴더
        maps = None                                            # Map file 이름
        map_num = self.args.test_map_num
        if use_maps_folder:
            if os.path.isdir(maps_folder):
                maps = sorted([f for f in os.listdir(maps_folder) if f.endswith('.npy')])
                if len(maps) > 0:
                    print(f"There is {len(maps)} map files for testing.")
                    map_num = len(maps)
                    use_maps_folder = True
                else:
                    print("There is no map file in the maps folder. Test will be conducted with random maps.")
                    use_maps_folder = False
            else:
                print("There is no maps folder. Test will be conducted with random maps.")
                use_maps_folder = False
        
        # 각 map에 대해서 test
        for map_idx in range(map_num):
             
            # FIXME: Map folder가 없을 경우도 구현 필요
            map_path = os.path.join(maps_folder, maps[map_idx]) if use_maps_folder else None
            map_config = MapConfigSchema(file_path = map_path)

            reset_info = self.test_env.reset(seed=reset_seed, map_config=map_config)
            if not reset_info:
                print(f"Can't reset map file: {maps[map_idx]}")
                continue
            obs, _ = reset_info
            level = self.test_env.map_layers.map_info.level
            
            # Map condition key: 현재를 level로만 분류
            condition_key = level
            if condition_key not in condition_results:
                condition_results[condition_key] = {
                    'coverage': [],
                    'overlap_percent': [],
                    'cleaning_time': []
                }
                
                # Level별 시각화 구조체 초기화
                visualized_maps[condition_key] = {
                    'best': None,   # (metrics, img, info_str)
                    'worst': None
                }
                
            coverage = []; overlap_percent = []; cleaning_time = []
            
            # 여러 starting point에서 model 성능을 test
            for start_idx in range(self.args.test_start_point_num):
            
                # start_idx == 0인 경우 map을 초기화하면서 시작 지점도 초기화가 되었으므로 reset을 실행하지 않음.
                if start_idx != 0:
                    obs, _ = self.test_env.reset() # 시작 지점 초기화
                start_time = time.time()
                cur_coverage, cur_overlap_percent, cur_cleaning_time = self._test_one_map(self.test_env, obs, debug=self.args.debug) # Coverage 성능을 평가
                end_time = time.time()
                computation_time.append(end_time-start_time)
                
                # Reachable grid의 수가 전체 grid 수의 절반을 넘는 경우에만 저장
                if self.test_env.map_layers.coverable.sum() >= self.test_env.H * self.test_env.W * 0.5:
                    condition_results[condition_key]['coverage'].append(cur_coverage)
                    condition_results[condition_key]['overlap_percent'].append(cur_overlap_percent)
                    condition_results[condition_key]['cleaning_time'].append(cur_cleaning_time)
                    
                    total_coverage.append(cur_coverage)
                    total_overlap_percent.append(cur_overlap_percent)
                    total_cleaning_time.append(cur_cleaning_time)
                    
                    cand_metrics = (cur_coverage, cur_overlap_percent, cur_cleaning_time)
                    cand_img = self.test_env.get_visualized_img(img_choice='traj')
                    cand_info = f"Map name: {os.path.basename(map_path)} | Cov: {cur_coverage*100:.1f}%, Overlap: {cur_overlap_percent:.1f}%, Time: {cur_cleaning_time:.1f}m"

                    best_rec = visualized_maps[condition_key]['best']
                    worst_rec = visualized_maps[condition_key]['worst']

                    # 1) Best 경로 갱신 확인
                    if best_rec is None or is_better_path(cand_metrics, best_rec[0]):
                        visualized_maps[condition_key]['best'] = (cand_metrics, cand_img, cand_info)

                    # 2) Worst 경로 갱신 확인 (is_better_path의 반대)
                    if worst_rec is None or is_better_path(worst_rec[0], cand_metrics):
                        visualized_maps[condition_key]['worst'] = (cand_metrics, cand_img, cand_info)
            # coverage_mean = np.mean(coverage) if coverage else 0.0
            # overlap_percent_mean = np.mean(overlap_percent) if overlap_percent else 0.0
            # cleaning_time_mean = np.mean(cleaning_time) if cleaning_time else 0.0
            
            # print(f"[Test result for {map_name}]\n"
            #     f"    Coverage mean: {coverage_mean*100:.2f}%\n"
            #     f"    Cleaning time mean: {cleaning_time_mean/60:.2f} min\n"
            #     f"    Overlap rate mean: {overlap_percent_mean*100:.2f}%")
            
            reset_seed = None
        
        # =========================================================================
        # 맵 조건(난이도)별 최종 평균 및 표준편차 출력부
        # =========================================================================
        print("\n" + "="*60)
        print("📊 [Test Results Breakdown by Map Conditions]")
        print("="*60)
        
        for cond_key in sorted(condition_results.keys()):
            level = cond_key
            res = condition_results[cond_key]
            
            # 데이터 개수 체크
            if not res['coverage']:
                print(f" ▶ Map Condition: Level {level} -> No Valid Data")
                continue
                
            # 조건별 평균(mean) 및 표준편차(std) 계산
            cov_m, cov_s = np.mean(res['coverage']), np.std(res['coverage'])
            ov_m, ov_s = np.mean(res['overlap_percent']), np.std(res['overlap_percent'])
            time_m, time_s = np.mean(res['cleaning_time']), np.std(res['cleaning_time'])
            
            print(f" ▶ Map Condition: Level {level} (Total tests: {len(res['coverage'])})")
            print(f"    - Coverage:     {cov_m*100:.2f}% ± {cov_s*100:.2f}%")
            print(f"    - Overlap Rate: {ov_m:.2f}% ± {ov_s:.2f}%")
            print(f"    - Cleaning Time: {time_m:.2f} min ± {time_s:.2f} min")
            print("-" * 50)

            # Level별 Best/Worst 결과 시각화
            vis_data = visualized_maps[level]
            if vis_data['best'] and vis_data['worst']:
                print(f"[Best Path]  {vis_data['best'][2]}")
                print(f"[Worst Path] {vis_data['worst'][2]}")
                
                # 이미지 출력 함수 호출 (환경 내부 지원 기능에 맞게 선택)
                display_image(vis_data['best'][1])
                display_image(vis_data['worst'][1])
            print("-" * 50)
            
        # 전체 총합 결과 (기존 코드 유지)
        total_coverage_mean = np.mean(total_coverage) if total_coverage else 0.0
        total_overlap_percent_mean = np.mean(total_overlap_percent) if total_overlap_percent else 0.0
        total_cleaning_time_mean = np.mean(total_cleaning_time) if total_cleaning_time else 0.0
        
        print("\n" + "="*60)
        print(f"[Overall Test result]\n"
            f"    Coverage mean: {total_coverage_mean*100:.2f}%\n"
            f"    Cleaning time mean: {total_cleaning_time_mean:.2f} min\n"
            f"    Overlap rate mean: {total_overlap_percent_mean:.2f}%\n"
            f"    Average computation time per map: {np.mean(computation_time):.2f} s")
        print("="*60)