import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import wandb
import vessl
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import cv2
import time
import os
from dataclasses import asdict
from itertools import product, cycle

import glob
from PIL import Image
from IPython.display import display, clear_output

# 앞서 정의한 클래스들을 임포트한다고 가정 (또는 같은 파일에 위치)
from path_planner.config import EnvConfig, TrainConfig, ANG_SEG_NUM, CLEANED_MAP_MAX, TRACE_MAP_MAX, LOCAL_VIEW_DIM
from path_planner.map_layer import MapConfigSchema
from path_planner.environment import CoverageEnv, ACTION_NUM, ALL_DIR_NUM
from path_planner.redqn_network import CNN_ReDQN
from path_planner.utils.utils import get_device_info, set_torch_seed
from path_planner.utils.map_utils import float_to_int_coord

class DQNAgent:
    
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
        
        # Environment 조성을 위한 config
        env_args = {
            k: v for k, v in vars(args).items() 
            if k in EnvConfig.__dataclass_fields__ and v is not None
        }
        self.env_cfg = EnvConfig(**env_args)
        
        # Policy network 조성
        network_config = {
            'action_size': ACTION_NUM,
            'all_dir_num': ALL_DIR_NUM,
            'use_noisy': args.use_noisy,
            'local_view_dim': LOCAL_VIEW_DIM,
            'do_normalize': args.do_normalize,
            'stack_steps': self.env_cfg.stack_steps,
        }
        self.policy_net = CNN_ReDQN(**network_config).to(self.device)
        
        # Heuristic trajectory
        self.dijkstra_traj = []
        
        # map_config_generator 생성
        levels = [1, 2, 3]
        eff_sizes_cm = [(400, 400), (300, 400), (300, 300)]
        self.map_config_combinations = list(product(levels, eff_sizes_cm))
        self.map_config_generator = cycle(self.map_config_combinations)
        
        
        # Train과 관련된 instance 변수는 mode가 train일 때만 생성
        if args.mode == 'train':
            
            # Train hyperparameter를 받음
            train_args = {
                k: v for k, v in vars(args).items() 
                if k in TrainConfig.__dataclass_fields__ and v is not None
            }
            self.train_cfg = TrainConfig(**train_args)
            
            # Train environment와 validation environment 조성
            self.env = CoverageEnv(self.env_cfg, seed=args.seed)
            if self.train_cfg.reset_only_start_pos:
                self.valid_env = CoverageEnv(self.env_cfg, seed=args.seed) # 훈련용 environment와 같은 환경을 구성하기 위해서
            else:
                self.valid_env = CoverageEnv(self.env_cfg, seed=args.seed+1000) # 훈련용 environment와 다른 환경을 구성하기 위해서
                
            # Training에 필요한 변수 설정
            self.total_steps = 0                        # Training을 진행한 steps 수 (Warmup 과정 포함)
            self.warmup_steps = 0                       # Buffer에 data를 채우기 위한 warmup steps 수
            self.no_warmup_steps = 0                    # total_steps에서 warmup 과정 동안 소요된 step 수를 제외한 step 수
            self.start_episode = 1                      # 시작 episode 번호: Checkpoint을 loading할 때 수정될 수 있음
            self.max_coverage_mean = 0.0                # Validation을 수행했을 때 가장 높았던 coverage
            self.min_overlap_percent_mean = float('inf')   # Validation을 수행했을 때 가장 낮았던 overlap rate
            self.min_cleaning_time_mean = float('inf')  # Validation을 수행했을 때 가장 낮았던 cleaning time    
            self.best_traj_img = None                   # Validation을 수행했을 때 가장 coverage를 잘 했던 trajectory를 img로 저장 (RGB image)
            self.num_heuristic = 0                      # Heuristic action이 선택된 횟수
            
            # Target network, Replay buffer 조성
            self.target_net = CNN_ReDQN(**network_config).to(self.device)   # Target network
            self.memory = deque(maxlen=self.train_cfg.buffer_size)          # Replay Buffer
            
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
        
        elif args.mode == 'test':
            if args.reset_only_start_pos:
                self.test_env = CoverageEnv(self.env_cfg, seed=args.seed) # Test environment 구현
            else:
                self.test_env = CoverageEnv(self.env_cfg, seed=args.seed+2000) # Test environment 구현
            
        # Loading model
        self._load_model(args)

        self.zigzag_fallback_path = None
        
    
    def _setup_logging(self):
        """
        Training process를 지켜 볼 tool 설정: tb, wandb, vessl
        """
        
        import pytz
        import datetime
        
        # 현재 시간 얻기
        kst = pytz.timezone('Asia/Seoul')
        current_time = datetime.datetime.now(kst).strftime("%y%m%d_%H%M")
        
        # Training 조건을 구별하기 위해 표시할 hyperparmeter 설정
        params_list_for_log = [
            'batch_size', 'lr', 'optimizer', 'momentum', 'epsilon_decay', 'use_softmax',
            'softmax_temp', 'use_noisy', 'target_with_noisy', 'gamma', 'reset_only_start_pos',
            'uncleaned_reward', 'cleaned_penalty', 'obstacle_penalty', 'turn_penalty', 'step_penalty'      
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
        - Checkpoint 파일명: (모델 이름)_(에피소드 번호)_(에피소드 보상).pth
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
            self.target_net.load_state_dict(data['model_state_dict'])
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
                self.no_warmup_steps = data['no_warmup_steps']
                self.start_episode = data['episode'] + 1
                print(f"Resumed training from episode {data['episode']} with reward {data['episode_reward']}")
                
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
    
    # def _load_memory(self, file_path: str):
        
    #     # 파일 존재 여부 확인
    #     if not os.path.exists(file_path):
    #         print(f"No memory file found at: {file_path}")
    #         return
        
    #     # 저장한 메모리 불러오기
    #     data = np.load(file_path)
    #     obs_keys = [k[4:] for k in data.files if k.startswith('obs_')]
        
    #     # Dictionary 재조립 (Reconstruct dict lists)
    #     num_samples = len(data['action'])
    #     reconstructed_obs = []
    #     reconstructed_next_obs = []

    #     for i in range(num_samples):
    #         # 각 스텝마다 딕셔너리 생성
    #         obs_dict = {k: data[f'obs_{k}'][i] for k in obs_keys}
    #         next_obs_dict = {k: data[f'next_obs_{k}'][i] for k in obs_keys}
            
    #         reconstructed_obs.append(obs_dict)
    #         reconstructed_next_obs.append(next_obs_dict)

    #     # 4. 기존 self.memory 구조인 [(s, a, r, ns, d), ...] 형태로 zip
    #     self.memory = list(zip(
    #         reconstructed_obs,
    #         data['action'],
    #         data['reward'],
    #         reconstructed_next_obs,
    #         data['done']
    #     ))

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
            episode = info['episode']; episode_reward = info['episode_reward']
            save_path = os.path.join(checkpoint_dir, model_name_base + f"_{episode}_{episode_reward:.2f}.pth")
            save_data.update({
                'episode': episode,
                'episode_reward': episode_reward,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'total_steps': self.total_steps,
                'no_warmup_steps': self.no_warmup_steps,
            })
            
        else:
            raise ValueError(f"Unsupported mode: {mode}. Expected 'model' or 'checkpoint'.")

        torch.save(save_data, save_path)
    
    # def _save_memory(self, file_path: str):
    #     # self.memory: [(s,a,r...), (s,a,r...)]
    #     # zip(*self.memory): (s,s...), (a,a...), (r,r...)
    #     obs_list, actions, rewards, next_obs_list, dones = zip(*self.memory)
        
    #     save_dict = {
    #         'action': np.array(actions, dtype=np.uint8),
    #         'reward': np.array(rewards, dtype=np.float32),
    #         'done': np.array(dones, dtype=np.bool_)
    #     }
        
    #     for key in obs_list[0].keys():
    #         save_dict[f'obs_{key}'] = np.array([o[key] for o in obs_list])
    #         save_dict[f'next_obs_{key}'] = np.array([no[key] for no in next_obs_list])
        
    #     np.savez_compressed(file_path, **save_dict)
     
    def _decide_action_masking(self, env: CoverageEnv, action: int, mode: str = 'test', reset: bool = False) -> bool:
        
        # Action masking을 결정하는 parameters
        ks = 15 if mode == 'train' else 7
        kp = 3
        
        if reset:
            max_len = max(ks, kp)
            self.coverage_hist = deque(maxlen=max_len)
            self.pos_hist = deque(maxlen=max_len)
        
        # Coverage와 position를 history로 저장
        check_ks = False; check_kp = False; check_collision = False
        
        # 다음 좌표 계산
        cx, cy = env.pos
        nx, ny = env.get_next_pos(dir=env.dir, action=action)
        
        # Action masking 여부 계산
        if len(self.coverage_hist) >= ks:
            check_ks = (self.coverage_hist[-ks] == env.coverage)
            
        if (len(self.coverage_hist) >= kp) and (len(self.pos_hist) >= kp):
            before_kp_pos_x, before_kp_pos_y = self.pos_hist[-kp]
            dist = abs(cx-before_kp_pos_x) + abs(cy-before_kp_pos_y)
            check_kp = (self.coverage_hist[-kp] == env.coverage) and (dist <= 1)
        
        check_collision = env.is_collide(nx, ny)
        
        # Coverage와 position의 history 저장
        self.coverage_hist.append(env.coverage)
        self.pos_hist.append(env.pos)
        
        return (check_ks or check_kp or check_collision)
        
        # except Exception as e:
        #     raise Exception(f"Unexpected error in action masking: {e}") from e
    
    
    def _get_action(self, env: CoverageEnv, processed_obs, mode: str='valid', reset: bool = False, warmup: bool = False) -> int:
        
        # mode 입력값 검사
        if mode not in ['train', 'valid', 'test']:
            raise ValueError(f"Unsupported mode: {mode}. Expected 'train', 'valid', or 'test'.")
        
        # action_mask로 가능한 action을 판단. 만약 모든 action이 불가능하면 오류를 발생시킴. 로봇이 갇혀 있는 경우는 아예 배제
        action_mask = processed_obs['action_mask']      # 충돌하는 action은 0, 충돌하지 않는 action은 1로 표시
        if not np.any(action_mask):
            raise ValueError(f"No valid actions available! All actions are masked at position {env.pos}.\n"
                             f"Check if the robot is trapped or if the map generation has an issue.")
        
        # Epsilon 기법: train mode일 때 action을 random하게 선택
        if mode == 'train' and self.train_cfg.use_epsilon: 
            
            # Epsilon 결정: Step 수가 늘어날수록 epsilon을 점점 줄임. Warmup 과정에서는 유지
            epsilon = max(self.train_cfg.epsilon_end, self.train_cfg.epsilon_start-self.no_warmup_steps/self.train_cfg.epsilon_decay)
            
            if self.train_rng.random() < epsilon:   # Action 선택
                valid_actions = np.where(action_mask == 1)[0]   # 충돌하지 않는 action 목록 list
                return self.train_rng.choice(valid_actions)
            
        # Greedy action을 얻음: Observation data를 tensor 변환 후 action을 얻음
        map_tensor = torch.from_numpy(processed_obs['map']).float().to(self.device).unsqueeze(0)
        vec_tensor = torch.from_numpy(processed_obs['vec']).to(self.device).unsqueeze(0)
        state = {'map': map_tensor, 'vec': vec_tensor}
        action_RL = self._get_RL_action(state, mode, action_mask) # Policy network를 통과시켜서 얻은 action
        action_masking = self._decide_action_masking(env, action_RL, mode, reset)
        # RL policy와 heuristic policy 중 무엇을 사용할지 결정
        # Heuristic action을 사용
        if env.is_zigzag_zone() or (mode == 'train' and (warmup or self.train_cfg.use_heuristic) and action_masking):
            # action_RL을 수행: next_state, reward, terminated 및 truncated 여부 등을 받음.
            # environment 내에서 trajectory의 변화는 없지만, replay buffer에는 transition data를 저장함.
            # 수행한 뒤, action을 수행하기 전의 환경으로 돌아감.
            if mode == 'train':
                self.num_heuristic += 1
                next_obs, reward, terminated, truncated, info = env.step(action_RL)
                next_processed_obs = self._pre_process_obs(next_obs, local_view_dim=LOCAL_VIEW_DIM)
                self.memory.append((processed_obs, action_RL, reward, next_processed_obs, terminated)) # processed_obs의 map data는 np.uint8 형태, Memory 용량을 줄임
                env.one_step_back()
            
            return self._get_heuristic_action(env) # Heuristic action을 return
        
        else: # RL policy action을 사용
            
            # Heuristic 방법으로 만든 trajectory를 RL policy가 따라가는 경우, self.dijkstra_traj를 유지
            if self.dijkstra_traj:
                next_pos_x, next_pos_y = env.get_next_pos(dir=env.dir, action=action_RL)   # RL_policy를 따랐을 때 다음 위치
                next_dji_pos_x, next_dji_pos_y = self.dijkstra_traj.pop(0)  # dijkstra_traj에서의 다음 위치
                
                # 두 위치가 일치하지 않는 경우 dijkstra_traj를 초기화
                eps = 1e-7
                if abs(next_pos_x - next_dji_pos_x) > eps or abs(next_pos_y - next_dji_pos_y) > eps:
                    self.dijkstra_traj = []
            
            return action_RL
        
        
    def _get_dijkstra_traj_from_parent(self, parent: dict, final_pos: tuple[int, int]):
        
        # dijkstra_traj를 빈 list로 초기화
        self.dijkstra_traj = []
        
        # 경로를 역순으로 생성
        self.dijkstra_traj.append(final_pos)
        next_child = parent[final_pos]
        while next_child is not None:
            self.dijkstra_traj.append(next_child)
            next_child = parent[next_child]
            
        # 경로를 다시 뒤집음
        self.dijkstra_traj.reverse()
        
        # 시작 지점은 제외
        self.dijkstra_traj.pop(0)
    
    # Heuristic action을 결정
    def _get_heuristic_action(self, env: CoverageEnv):
        
        if len(self.dijkstra_traj) > 0: # 이미 생성한 경로가 있으면 다음 action을 출력
            next_pos = self.dijkstra_traj.pop(0)
            next_dir = env.get_dir_from_next_pos(next_pos)
            env.dir = next_dir
            return 0 # Action 0: 앞으로 이동
        
        # BFS 탐색으로 경로 탐색
        queue = deque([env.pos])
        start = env.pos        # 현재 위치를 시작 지점으로 저장
        visited = {float_to_int_coord(*start)} # set 형태. 탐색을 빠르게 수행. BFS를 하면서 지나간 grid를 저장하기 위한 용도
        parent = {start: None}  # 해당 grid로 오기 위해 어떤 grid를 거쳤는지 저장.
            
        while queue:
            curr_pos = queue.popleft()
            for dir_vec in env.base_dir_vecs:
                
                next_pos = (curr_pos[0]+dir_vec[0], curr_pos[1]+dir_vec[1])
                
                if not env.is_collide(*next_pos) and (int(next_pos[0]+0.5), int(next_pos[1]+0.5)) not in visited:
                    visited.add((int(next_pos[0]+0.5), int(next_pos[1]+0.5)))
                    parent[next_pos] = curr_pos
                    queue.append(next_pos)
                
                if not env.is_zigzag_zone(*next_pos) and env.map_layers.has_uncleaned_grid(*next_pos): # 새로운 grid를 밟을 수 있는 위치를 발견하면 그 위치로 이동하는 경로를 생성
                    self._get_dijkstra_traj_from_parent(parent, next_pos)
                    next_pos = self.dijkstra_traj.pop(0)
                    next_dir = env.get_dir_from_next_pos(next_pos)
                    env.dir = next_dir
                    return 0 # Action 0: 앞으로 이동
                
        # base_dir_vec으로 갈 수 있는 위치가 없는 경우, 갈 수 있는 action을 찾아 return
        for action in range(ACTION_NUM):
            next_pos = env.get_next_pos(dir=env.dir, action=action)
            if not env.is_collide(*next_pos):
                return action
        else:
            raise ValueError(f"No valid actions available in heuristic action selection! All actions are blocked at position {env.pos}.\n")
        
        
    # State만 받는 것으로 수정    
    def _get_RL_action(self, state, mode: str = 'valid', action_mask: np.ndarray = None) -> int:
        """
        현재 state에서 RL action을 얻는 method

        Args:
            state: Q-network에 통과시키기 위한 state
            mode (str, optional): 'train', 'valid', 또는 'test' 중 선택할 mode를 결정. Defaults to 'valid'.

        Returns:
            int: 선택한 action
        """
        # action_mask는 test 시에는 반드시 사용. training 시에는 masking 유무를 선택
        # greedy action을 선택할 때만 사용. Warmup, epsilon 탐험, collision 이후 랜덤 탐험 시에는 action_mask 적용 X
        # Action을 Q-value로 뽑지 않더라도 일단 Q-value를 얻어서 관찰
        
        with torch.no_grad(): 
            if mode == 'train' and self.train_cfg.use_noisy:
                self.policy_net.reset_noise() # 논문 기법: 결정 전 노이즈 리셋
            map_tensor = state['map']; vec_tensor = state['vec']
            q_values = self.policy_net(map_tensor, vec_tensor) # Shape: (1, action_space)
            
        if mode == 'train' and self.train_cfg.use_softmax:
            probs = F.softmax(q_values / self.train_cfg.softmax_temp, dim=1).cpu().numpy().flatten() # 확률값 계산
            probs = probs / (probs.sum() + 1e-8) # 확률 합이 1이 안 될 경우를 대비해 normalize (안전장치)
            action = self.train_rng.choice(len(probs), p=probs)
        elif mode == 'test':
            assert action_mask is not None, "Action mask must be provided in test mode."
            masked_q_values = q_values.cpu().numpy().flatten()
            masked_q_values[action_mask == 0] = -np.inf
            action = masked_q_values.argmax().item()
        else:
            action = q_values.argmax().item()
        
        return action
        
    def _pre_process_obs(self, obs, local_view_dim=51) -> dict:
        
        # --------- map data 변환: H와 W의 local_view 구역을 local_view_dim 크기로 변환 ---------
        num_layers = obs['map'].shape[0]
        hwc_map = np.transpose(obs['map'], (1, 2, 0)) # obs의 map data가 (C, H, W) 형태이므로 이를 (H, W, C) 형태로 변환
        
        resized_hwc_map = np.zeros((local_view_dim+2, local_view_dim+2, num_layers), dtype=np.float32)
        
        do_resize = (hwc_map.shape[0] != local_view_dim+2) or (hwc_map.shape[1] != local_view_dim+2)
        
        if do_resize:
            for i in range(num_layers):
                # 중앙의 local_view를 resize
                resized_hwc_map[1:-1, 1:-1, i] = cv2.resize(
                    hwc_map[1:-1, 1:-1, i], 
                    (local_view_dim, local_view_dim), 
                    interpolation=cv2.INTER_AREA
                )
                
                # 테두리 resize
                resized_hwc_map[0, 1:-1, i] = cv2.resize(hwc_map[0:1, 1:-1, i], (local_view_dim, 1), interpolation=cv2.INTER_AREA)[0]
                resized_hwc_map[-1, 1:-1, i] = cv2.resize(hwc_map[-1:, 1:-1, i], (local_view_dim, 1), interpolation=cv2.INTER_AREA)[0]
                resized_hwc_map[1:-1, 0, i] = cv2.resize(hwc_map[1:-1, 0:1, i], (1, local_view_dim), interpolation=cv2.INTER_AREA)[:, 0]
                resized_hwc_map[1:-1, -1, i] = cv2.resize(hwc_map[1:-1, -1:, i], (1, local_view_dim), interpolation=cv2.INTER_AREA)[:, 0]

                if i%3 == 2: # trace layer: resize하면 수치 왜곡이 많이 생기므로 이를 보정
                    center = local_view_dim//2+1
                    radius = min(3, local_view_dim//2)
                    max_value = np.max(resized_hwc_map[center-radius:center+radius+1, center-radius:center+radius+1, i])
                    if max_value > 0:
                        resized_hwc_map[:, :, i] /= max_value
                
            # 네 모서리 그대로 복사
            resized_hwc_map[0, 0, :] = hwc_map[0, 0, :]
            resized_hwc_map[0, -1, :] = hwc_map[0, -1, :]
            resized_hwc_map[-1, 0, :] = hwc_map[-1, 0, :]
            resized_hwc_map[-1, -1, :] = hwc_map[-1, -1, :]
            
            processed_map = np.transpose(resized_hwc_map, (2, 0, 1))    # (C, H, W) 형태로 다시 변환
        else:
            processed_map = obs['map'].copy()
        # -----------------------------------------------------------------------------------

        # vec data 변환
        if obs['vec'].dtype == np.float32:
            processed_vec = obs['vec'].copy()
        else:
            processed_vec = obs['vec'].astype(np.float32)
            
        # action_mask 복사
        action_mask = obs['action_mask']
            
        return {'map': processed_map, 'vec': processed_vec, 'action_mask': action_mask}
    
    def _validation(self, episode: int):
        """
        reset_only_start_pos가 참이면 train한 map과 같은 map을 이용
        """
        
        self.policy_net.eval() # eval mode로 전환
        coverage = []; overlap_percent = []; cleaning_time = []
        max_coverage = 0.0; min_overlap_percent = float('inf'); min_cleaning_time = float('inf')
        coverage_threshold = 0.90; overlap_percent_gap = 0.10
        coverage_mean = 0.0; overlap_percent_mean = 0.0; cleaning_time_mean = 0.0
        best_traj_img = None # 가장 성능이 좋았던 map에서의 trajectory를 보여줌
        
        # Model의 성능 평가
        for level, (eff_H_cm, eff_W_cm) in self.map_config_combinations:
            
            map_config = MapConfigSchema(level=level, eff_H_cm=eff_H_cm, eff_W_cm=eff_W_cm)
            
            for _ in range(self.train_cfg.valid_map_num):
                
                # Validation 환경을 초기화
                if self.train_cfg.reset_only_start_pos:
                    obs, _ = self.valid_env.reset()
                else:
                    obs, _ = self.valid_env.reset(map_config=map_config)
                    
                for start_num in range(self.train_cfg.valid_start_point_num):
                    
                    # start_num == 0인 경우 map을 초기화하면서 시작 지점도 초기화가 되었으므로 reset을 실행하지 않음.
                    if start_num != 0:
                        obs, _ = self.valid_env.reset() # 시작 지점 초기화
                    cur_coverage, cur_overlap_percent, cur_cleaning_time = self._test_one_map(self.valid_env, obs, mode='valid') # Coverage 성능을 평가
                    
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

    
    def _generate_zigzag_trajectory(self, env: CoverageEnv, start_pos: tuple[int, int]) -> list[tuple[int, int]]:
        """
        저밀도 zone(mode_map==0) 전체를 커버하는 boustrophedon 경로를 생성.
        반환값: start_pos를 포함한, 인접한 칸들로만 이루어진 좌표 리스트.
        """
        cx0, cy0 = start_pos
        mode_map = env.map_layers.mode_map
        H = env.map_layers.map_info.eff_H; W = env.map_layers.map_info.eff_W

        def free(x, y):
            return 0 <= x < W and 0 <= y < H and mode_map[y, x] == 0 and not env.is_collide(x, y)

        # 1. Zone 영역(로봇 중심 reachable mask) BFS
        reachable = np.zeros((H, W), dtype=np.uint8)
        reachable[cy0, cx0] = 1
        queue = deque([(cx0, cy0)])
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if free(nx, ny) and not reachable[ny, nx]:
                    reachable[ny, nx] = 1
                    queue.append((nx, ny))

        ys, _ = np.nonzero(reachable)
        if len(ys) == 0:
            return [start_pos]
        y_min, y_max = int(ys.min()), int(ys.max())
        
        robot_diameter = max(1, env.cfg.robot_size)
        num_rows = int(np.ceil((y_max - y_min) / robot_diameter)) + 1
        row_ys = [int(y) for y in np.linspace(y_min, y_max, num_rows)]
        
        def bfs_path(src, dst):
            """reachable mask 안에an서 src -> dst 최단 경로(src 제외)를 구함"""
            if src == dst:
                return []
            q = deque([src])
            visited = {src}
            parent = {src: None}
            while q:
                x, y = q.popleft()
                if (x, y) == dst:
                    break
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H and reachable[ny, nx] and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        parent[(nx, ny)] = (x, y)
                        q.append((nx, ny))
            if dst not in parent:
                return []  # 이론상 같은 reachable 영역이면 항상 도달 가능
            path = []
            node = dst
            while node != src:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path

        all_segments = []
        for row_y in row_ys:
            row_xs = sorted(int(x) for x in np.nonzero(reachable[row_y, :])[0])
            if not row_xs:
                continue
            
            # 장애물 등으로 끊어진 x축 구간 분리
            seg_start = row_xs[0]
            prev = row_xs[0]
            for x in row_xs[1:]:
                if x != prev + 1:
                    all_segments.append((row_y, seg_start, prev))
                    seg_start = x
                prev = x
            all_segments.append((row_y, seg_start, prev))        
        
        trajectory = [(cx0, cy0)]
        current = (cx0, cy0)
        
        while all_segments:
            # 현재 위치에서 각 segment의 양 끝단(a, b)까지의 단순 거리(우회 비용 대용) 계산
            def get_closest_dist(seg):
                y, a, b = seg
                dist_a = abs(current[0] - a) + abs(current[1] - y)
                dist_b = abs(current[0] - b) + abs(current[1] - y)
                return min(dist_a, dist_b), dist_a <= dist_b

            # 가장 가까운 segment를 다음 목표로 결정하고 리스트에서 제거
            best_seg = min(all_segments, key=lambda s: get_closest_dist(s)[0])
            all_segments.remove(best_seg)
            
            y, a, b = best_seg
            _, a_is_closer = get_closest_dist(best_seg)
            
            # 더 가까운 쪽 진입점(x_from)과 끝점(x_to) 결정
            x_from, x_to = (a, b) if a_is_closer else (b, a)
            target_start = (x_from, y)
            
            # 6-A. 현재 위치에서 다음 청소 구간 시작점까지 장애물을 우회하여 이동 (A*나 BFS 경로)
            if current != target_start:
                hop = bfs_path(current, target_start)
                trajectory.extend(hop)
                current = target_start
                
            # 6-B. 진입한 segment를 쭉 한 방향으로 훑으며 청소 수행
            step = 1 if x_to >= x_from else -1
            for x in range(x_from + step, x_to + step, step):
                trajectory.append((x, y))
                current = (x, y)

        return trajectory

    def _get_zigzag_global_dir(self, env: CoverageEnv) -> int:
        """
        저밀도 zone에 진입하면 zone 전체를 커버하는 trajectory를 한 번에 생성하고,
        이후로는 그 경로를 한 칸씩 따라가며 방향만 반환.
        * 방향 매핑: 0: East, 1: North, 2: West, 3: South
        """
        cx, cy = float_to_int_coord(*env.pos)
        
        if env.cleaned_segment[cy, cx] == 1:
            return -1

        if not hasattr(self, 'zigzag_trajectory') or self.zigzag_trajectory is None:
            self.zigzag_trajectory = self._generate_zigzag_trajectory(env, (cx, cy))
            self.zigzag_trajectory.pop(0)  # 시작점(현재 위치) 제외

        if not self.zigzag_trajectory:
            return -1  # 경로를 다 따라갔으면 탈출 신호

        next_pos = self.zigzag_trajectory.pop(0)
        dx, dy = next_pos[0] - cx, next_pos[1] - cy
        dir_offsets = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
        return dir_offsets[(dx, dy)]
    
    
    def _test_one_map(self, env: CoverageEnv, obs: dict, mode: str = 'valid', debug: bool = False) -> tuple[float, float, float]:
        
        done = False
        reset = True
        last_obs = obs
        last_info = None
        debug_skip_count = 0
        self.dijkstra_traj = []
        
        # ────────────── [추가] 새 맵 시작 시 로컬 구역 초기화 플래그 리셋 ──────────────
        self.zigzag_trajectory = None
        
        # [1] 디버그 모드일 때 사용할 도화지(fig)를 미리 딱 한 번만 만듭니다.
        if debug:
            fig, axes = plt.subplots(3, 1, figsize=(18, 30))
            # 초기 이미지
            init_traj_img = env.get_visualized_img(img_choice='traj')
            init_obs_img = env.get_visualized_img(img_choice='obs')
            init_pro_obs_img = env.get_visualized_img(img_choice='obs', preprocessor=self._pre_process_obs)
            
            # 초기 빈 이미지 설치
            im_traj = axes[0].imshow(init_traj_img)
            im_obs = axes[1].imshow(init_obs_img)
            im_pro_obs = axes[2].imshow(init_pro_obs_img)
            text_traj = axes[0].text(0.5, 0, "", transform=axes[0].transAxes, ha="center", fontsize=15, color='black')
            axes[0].set_title("Trajectory", fontsize=20)
            axes[1].set_title("Observation", fontsize=20)
            axes[2].set_title("Processed Observation", fontsize=20)
            for ax in axes: ax.axis('off')
            plt.tight_layout()
            
            # Jupyter용 디스플레이 핸들을 생성 (이걸 통해 이미지만 쏙 바꿉니다)
            display_handle = display(fig, display_id=True)
            plt.close(fig) # 별도의 정적 출력이 생기지 않도록 닫기
        
        while not done:
            processed_obs = self._pre_process_obs(last_obs, local_view_dim=LOCAL_VIEW_DIM)
            
            # 발밑 격자가 0번 구역(Heuristic)인지 실시간 검사
            is_zigzag_zone = env.is_zigzag_zone()
            if is_zigzag_zone:
                self.dijkstra_traj = []
                global_dir = self._get_zigzag_global_dir(env)
                action = None

                if global_dir == -1:
                    is_zigzag_zone = False # 경로를 다 소진했으면 즉시 DQN 모드로 전환
                    env.mark_current_segment()

            if not is_zigzag_zone:
                self.zigzag_trajectory = None
                action = self._get_action(env, processed_obs, mode=mode, reset=reset)
                global_dir = None
                reset = False
            
            
            if debug:
                
                if debug_skip_count > 0:
                    debug_skip_count -= 1
                else:
                    
                    with torch.no_grad():
                        map_tensor = torch.from_numpy(processed_obs['map']).float().to(self.device).unsqueeze(0)
                        vec_tensor = torch.from_numpy(processed_obs['vec']).to(self.device).unsqueeze(0)
                        q_values = self.policy_net(map_tensor, vec_tensor).squeeze().cpu().numpy()
                    
                    q_str = ", ".join([f"{q:.2f}" for q in q_values])
                    action_info = (f"[Selected action]: {action}\n"
                                   f"[Q-values] [{q_str}]")
                    
                    # [3] 데이터만 가져와서 기존 이미지 객체에 덮어쓰기 (가장 핵심)
                    traj_img = env.get_visualized_img(img_choice='traj')
                    obs_img = env.get_visualized_img(img_choice='obs')
                    pro_obs_img = env.get_visualized_img(img_choice='obs', preprocessor=self._pre_process_obs)
                    
                    im_traj.set_data(traj_img)
                    im_obs.set_data(obs_img)
                    im_pro_obs.set_data(pro_obs_img)
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
                            continue
                        elif val > 0:
                            debug_skip_count = int(user_val) - 1
                        else:
                            continue
                    except ValueError:
                        print("숫자 또는 'q'를 입력해주세요.")

            if is_zigzag_zone:
                next_obs, _, terminated, truncated, info = env.step(global_dir=global_dir)
            else:
                next_obs, _, terminated, truncated, info = env.step(action=action)
            done = terminated or truncated
            last_obs = next_obs
            last_info = info
            
            
        coverage = last_info['Coverage']
        overlap_percent = env.overlap_percent
        cleaning_time = env.cleaning_time
        
        return coverage, overlap_percent, cleaning_time
    
    def train(self):
        
        assert self.args.mode == 'train'
        
        # Policy network를 train mode로 설정
        self.policy_net.train()
        
        # Target network를 train mode 또는 eval mode로 설정
        if self.train_cfg.target_with_noisy:
            self.target_net.train()
        else:
            self.target_net.eval()
        

        for episode in range(self.start_episode, self.train_cfg.max_episodes+1):
            
            # -------------------- Environment reset: seed와 options를 얻음 -------------------------
            # Seed: 처음 environment를 reset할 때만 
            if episode == self.start_episode:
                seed = self.seed
            else:
                seed = None
            
            # Map 설정: Level, map 크기
            map_config = None
            if not self.train_cfg.reset_only_start_pos:
                level, (eff_H_cm, eff_W_cm) = next(self.map_config_generator)
                map_config = MapConfigSchema(level=level, eff_H_cm=eff_H_cm, eff_W_cm=eff_W_cm)
            
            # Environment reset
            obs, info = self.env.reset(seed=seed, map_config=map_config)
            processed_obs = self._pre_process_obs(obs, local_view_dim=LOCAL_VIEW_DIM) # Map data를 resize, Observation data를 얻음
            # ----------------------------------------------------------------------------------------
            self.zigzag_trajectory = None
            
            # Environment에서 step을 처음 시작할 때 설정
            episode_reward = 0      # Episode에서 얻은 reward 총합
            steps = 0               # Episode의 진행한 step 수
            done_ep = False         # Episode 종료 조건: truncated(Episode를 조기 종료) | (terminate & Success) (Collision인 경우는 종료 X)
            warmup = False          # 현재 buffer에 data를 쌓고만 있는지 parameter도 같이 update 중인지 결정
            reset = True            # 첫 번째 action을 선택할 때, action masking을 위한 buffer를 초기화
            self.dijkstra_traj = [] # Heuristic action을 위한 dijkstra_traj 초기화
            self.num_heuristic = 0  # Heuristic action이 선택된 횟수 세기
            
            last_info = {}          # Episode 마지막에 얻은 info
            
            # Warmup 조건을 매 episode마다 확인
            # Warmup 시 action은 random 선택. Buffer 저장만 수행하고 parameter update은 하지 않음.
            assert self.train_cfg.buffer_size > self.train_cfg.warmup_tot_steps
            if len(self.memory) < self.train_cfg.warmup_tot_steps:
                warmup = True
                self.env.set_env_mode(warmup=True, ep_steps=self.train_cfg.warmup_ep_steps) # Environment를 warmup하는 mode로 전환
            else:
                warmup = False
                self.env.set_env_mode(warmup=False, ep_steps=self.train_cfg.warmup_ep_steps)
            
            first_q_value = None # Episode에서 처음으로 관찰되는 Q-value 저장 (Episode마다 초기화)
            cumulated_reward = 0.0 # Episode에서 누적된 reward 저장 (Episode마다 초기화)
            gamma_exp = 1.0 # 감쇠 계수
            # ~~~~ Episode 내에서 training 수행 ~~~~
            while not done_ep:
                
                # Steps 수 세기
                self.total_steps += 1
                steps += 1
                
                # Action 선택: Warmup 중에는 100% random, 그 이후에는 epsilon 기법 사용
                is_zigzag_zone = self.env.is_zigzag_zone()
                if is_zigzag_zone:
                    self.dijkstra_traj = []
                    global_dir = self._get_zigzag_global_dir(self.env)
                    action = None
                    
                    if global_dir == -1:
                        is_zigzag_zone = False
                        self.env.mark_current_segment()
                
                if not is_zigzag_zone:
                    self.zigzag_trajectory = None
                    action = self._get_action(self.env, processed_obs, mode='train', reset=reset, warmup=warmup)
                    global_dir = None
                    reset = False
                
                next_obs, reward, terminated, truncated, info = self.env.step(action=action, global_dir=global_dir)
                next_processed_obs = self._pre_process_obs(next_obs, local_view_dim=LOCAL_VIEW_DIM)
                
                # Episode를 종료하는 조건: Coverage에 성공한 경우 or Episode가 조기 종료된 경우 (Collision 포함)
                # Warmup 시에는 truncated일 때 info['Steps'] == args.warmup_ep_steps일 때만 종료
                if warmup:
                    done_ep = terminated or (truncated & (info['Steps'] >= self.train_cfg.warmup_ep_steps))
                else:
                    done_ep = terminated or truncated
                done = terminated # Buffer에 done이라고 저장하는 조건: Collision & Coverage 성공
                
                # Episode reward 계산
                episode_reward += reward
                cumulated_reward += reward * gamma_exp
                gamma_exp *= self.train_cfg.gamma
                
                if not is_zigzag_zone:
                    self.memory.append((processed_obs, action, reward, next_processed_obs, done)) # processed_obs의 map data는 np.uint8 형태, Memory 용량을 줄임
                    
                    if warmup:
                        self.warmup_steps += 1
                    else:
                        self.no_warmup_steps += 1
                        
                    # First q_value와 cumulate reward 계산
                    if first_q_value is None:
                        map_tensor = torch.from_numpy(processed_obs['map']).float().to(self.device).unsqueeze(0)
                        vec_tensor = torch.from_numpy(processed_obs['vec']).to(self.device).unsqueeze(0)
                        with torch.no_grad():
                            q_values = self.policy_net(map_tensor, vec_tensor) # Shape: (1, action_space)
                            first_q_value = q_values[0, action].item()
                
                
                    # 학습 수행
                    if not warmup and self.no_warmup_steps % self.train_cfg.policy_update == 0:
                        batch_indices = self.train_rng.choice(len(self.memory), size=self.train_cfg.batch_size, replace=False)
                        batch = [self.memory[i] for i in batch_indices]
                        
                        # batch 개별 요소의 구조: (processed_obs, action, reward, next_processed_obs, done)
                        ms_b = torch.from_numpy(np.array([b[0]['map'] for b in batch])).float().to(self.device)     # Map state: (B, 3, 51, 51)
                        vs_b = torch.from_numpy(np.array([b[0]['vec'] for b in batch])).float().to(self.device)     # Vector state: (B, 12)
                        a_b = torch.LongTensor([b[1] for b in batch]).unsqueeze(1).to(self.device)                  # Action: (B, 1)
                        r_b = torch.FloatTensor([b[2] for b in batch]).unsqueeze(1).to(self.device)                 # Reward: (B, 1)
                        nms_b = torch.from_numpy(np.array([b[3]['map'] for b in batch])).float().to(self.device)    # Next map state: (B, 3, 51, 51)
                        nvs_b = torch.from_numpy(np.array([b[3]['vec'] for b in batch])).float().to(self.device)    # Next vector state: (B, 12)
                        d_b = torch.FloatTensor([b[4] for b in batch]).unsqueeze(1).to(self.device)                 # Done: (B, 1)
                        
                        # Q(s, a) 계산
                        if self.train_cfg.use_noisy:
                            self.policy_net.reset_noise()
                        curr_q = self.policy_net(ms_b, vs_b).gather(dim=1, index=a_b) # Shape: (B, 4) -> (B, 1)
                        
                        # Target Q 계산
                        with torch.no_grad():
                            if self.train_cfg.use_noisy and self.train_cfg.target_with_noisy:
                                self.target_net.reset_noise()
                                
                            if self.train_cfg.double_dqn: # Double DQN을 사용
                                next_q_target_b = self.policy_net(nms_b, nvs_b) # 다음 state에 대한 Q-value를 얻음. Shape: (B, 4)
                                next_a_b = next_q_target_b.max(dim=1)[1].unsqueeze(1) # Shape: (B, 1)
                                next_q = self.target_net(nms_b, nvs_b).gather(dim=1, index=next_a_b) # Shape: (B, 1)
                            else: 
                                next_q = self.target_net(nms_b, nvs_b).max(dim=1)[0].unsqueeze(1) # Shape: (B, 1) (torch.max에 dimension을 지정하면 최댓값 tensor와 indices tensor를 tuple로 반환하기 때문에 [0]이 필요)
                            
                            target_q = r_b + (1 - d_b) * self.train_cfg.gamma * next_q
                        
                        # Loss 계산 후 parameter update
                        loss_func = nn.HuberLoss(delta=1.0)
                        loss = loss_func(curr_q, target_q.detach())
                        self.optimizer.zero_grad()
                        loss.backward()
                        
                        # Gradient clipping: gradient 폭주 방지
                        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
                        
                        # Weight update
                        self.optimizer.step()
                        
                        # TensorBoard 또는 wandb 기록
                        sigma_logs = self.policy_net.get_sigma_data()
                        if self.tb_writer:
                            self.tb_writer.add_scalar("Train/Loss", loss.item(), self.total_steps)
                            self.tb_writer.add_scalar("Train/Q_value_mean", curr_q.mean().item(), self.total_steps)
                        if self.wandb_run:
                            self.wandb_run.log({"Train/Loss": loss.item(),
                                                "Train/Q_value_mean": curr_q.mean().item(), **sigma_logs}, step=self.total_steps)
                        if self.args.use_vessl:
                            vessl.log(step=self.total_steps, 
                                            payload={"Train/Loss": loss.item(), 
                                                        "Train/Q_value_mean": curr_q.mean().item()})
                            
                    # Target Network 업데이트
                    if not warmup and self.no_warmup_steps % self.train_cfg.target_update == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                    
                # Episode 수 저장
                if self.wandb_run:
                    self.wandb_run.log({"Train/Episodes": episode}, step=self.total_steps)
                
                # State update
                processed_obs = next_processed_obs
                
                # info update
                last_info = info
                
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            # log로 보낼 info data를 얻음
            coverage = last_info.get("Coverage", 0.0)
            ep_collision = last_info.get("Episode_collision", 0)
            overlap_percent = self.env.overlap_percent
            cleaning_time = self.env.cleaning_time
            
            # 매 episode 마다 TensorBoard 또는 wandb 기록
            if self.tb_writer:
                self.tb_writer.add_scalar("Stats/Episode_reward", episode_reward, episode)
                self.tb_writer.add_scalar("Stats/Coverage_rate", coverage, episode)
                self.tb_writer.add_scalar("Stats/Collision_count", ep_collision, episode)
                self.tb_writer.add_scalar("Stats/overlap_percent", overlap_percent, episode)
                self.tb_writer.add_scalar("Stats/Cleaning_time", cleaning_time, episode)
            if self.wandb_run:
                self.wandb_run.log({"Stats/Episode_reward": episode_reward,
                                    "Stats/Cumulated_reward": cumulated_reward,
                                    "Stats/First_Q_value": first_q_value,
                                    "Stats/Difference_btw_first_Q_and_Cum_Q": first_q_value - cumulated_reward if first_q_value is not None else None,
                                    "Stats/Coverage_rate": coverage,
                                    "Stats/Collision_count": ep_collision,
                                    "Stats/overlap_percent": overlap_percent,
                                    "Stats/Cleaning_time": cleaning_time,
                                    "Stats/Num_Heuristic_actions": self.num_heuristic}, step=self.total_steps)
            if self.args.use_vessl:
                vessl.log(step=episode, payload={
                    "Stats/Episode_reward": episode_reward,
                    "Stats/Coverage_rate": coverage,
                    "Stats/Collision_count": ep_collision,
                    "Stats/overlap_percent": overlap_percent,
                    "Stats/Cleaning_time": cleaning_time
                })
            
            # Checkpoint 저장
            if not warmup and episode % self.train_cfg.ckp_freq == 0:
                checkpoint_info = {"episode": episode, "episode_reward": episode_reward}
                self._save_model(mode='checkpoint', info=checkpoint_info)
                
            # Validation 수행
            if not warmup and episode % self.train_cfg.valid_freq == 0:
                print("Validation...")
                self._validation(episode)
            
            # Map 기록 저장
            if (warmup and episode % 1 == 0) or (not warmup and episode % 5 == 0):
                map_img = self.env.get_visualized_img(img_choice='traj')
                if self.tb_writer:
                    self.tb_writer.add_image("Visualization/Robot_path", map_img, episode, dataformats="HWC")
                if self.wandb_run:
                    self.wandb_run.log({"Visualization/Robot_path": wandb.Image(map_img)}, step=self.total_steps)
                if self.args.use_vessl:
                    vessl.log(step=episode, payload={"Visualization/Robot_path": vessl.Image(map_img)})
            
            # Episode 결과 출력        
            print(f"Episode: {episode}, Warmup: {warmup}, Reward: {episode_reward:.2f}, Steps: {steps}, Total_steps: {self.total_steps}", flush=True)
                  
            if not warmup:
                print(f"\tLoss: {loss:.2f}", flush=True)

    def test(self, use_maps_folder: bool=True):
        self.policy_net.eval() # eval mode로 전환
        total_coverage = []; total_overlap_percent = []; total_cleaning_time = []
        computation_time = []
        options={"reset_only_start_pos": True} # 시작 지점만 초기화하기 위한 option
        reset_seed = self.seed
        
        condition_results = {} # Map condition 별로 test 결과를 저장하기 위한 dictionary
        
        maps_folder = os.path.join(self.args.map_save_dir, 'test') # Map을 저장한 폴더
        maps = None                          # Map file 이름
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
            
            obs, _ = self.test_env.reset(seed=reset_seed, map_config=map_config)
            eff_H_grid = self.test_env.map_layers.map_info.eff_H
            eff_W_grid = self.test_env.map_layers.map_info.eff_W
            level = self.test_env.map_layers.map_info.level
            
            # 딕셔너리 Key 생성 (예: (300, 300, 1) -> 3m x 3m, Level 1)
            condition_key = (eff_H_grid, eff_W_grid, level)
            if condition_key not in condition_results:
                condition_results[condition_key] = {
                    'coverage': [],
                    'overlap_percent': [],
                    'cleaning_time': []
                }
                
            coverage = []; overlap_percent = []; cleaning_time = []
            
            # 여러 starting point에서 model 성능을 test
            for start_idx in range(self.args.test_start_point_num):
            
                # start_idx == 0인 경우 map을 초기화하면서 시작 지점도 초기화가 되었으므로 reset을 실행하지 않음.
                if start_idx != 0:
                    obs, _ = self.test_env.reset() # 시작 지점 초기화
                start_time = time.time()
                cur_coverage, cur_overlap_percent, cur_cleaning_time = self._test_one_map(self.test_env, obs, mode='test', debug=self.args.debug) # Coverage 성능을 평가
                end_time = time.time()
                computation_time.append(end_time-start_time)
                self.test_env.show_visualized_img(img_choice='traj') # trajectory 시각화
                
                # Reachable grid의 수가 전체 grid 수의 절반을 넘는 경우에만 저장
                if self.test_env.map_layers.coverable.sum() >= self.test_env.H * self.test_env.W * 0.5:
                    # 각 지표의 평균을 구하기 위해 현재 test에서의 지표값을 저장
                    # coverage.append(cur_coverage)
                    # overlap_percent.append(cur_overlap_percent)
                    # cleaning_time.append(cur_cleaning_time)
                    
                    condition_results[condition_key]['coverage'].append(cur_coverage)
                    condition_results[condition_key]['overlap_percent'].append(cur_overlap_percent)
                    condition_results[condition_key]['cleaning_time'].append(cur_cleaning_time)
            
            # coverage_mean = np.mean(coverage) if coverage else 0.0
            # overlap_percent_mean = np.mean(overlap_percent) if overlap_percent else 0.0
            # cleaning_time_mean = np.mean(cleaning_time) if cleaning_time else 0.0
            
            # print(f"[Test result for {map_name}]\n"
            #     f"    Coverage mean: {coverage_mean*100:.2f}%\n"
            #     f"    Cleaning time mean: {cleaning_time_mean/60:.2f} min\n"
            #     f"    Overlap rate mean: {overlap_percent_mean*100:.2f}%")

            total_coverage.extend(coverage)
            total_overlap_percent.extend(overlap_percent)
            total_cleaning_time.extend(cleaning_time)
            
            reset_seed = None
        
        # =========================================================================
        # ⭐ [추가] 맵 조건(크기 및 난이도)별 최종 평균 및 표준편차 출력부
        # =========================================================================
        print("\n" + "="*60)
        print("📊 [Test Results Breakdown by Map Conditions]")
        print("="*60)
        
        # 가독성을 위해 정렬하여 순회
        for cond_key in sorted(condition_results.keys()):
            h_cm, w_cm, level = cond_key
            res = condition_results[cond_key]
            
            # FIXME: h_m를 반환할 수 있도록 수정. 지금은 표면상 잘 뜨도록 설정함.
            # 미터 단위 가독성 변환 (예: 300cm -> 3m)
            h_m, w_m = h_cm / 50 , w_cm / 50
            
            # 데이터 개수 체크
            if not res['coverage']:
                print(f" ▶ Map Condition: {h_m:.1f}mx{w_m:.1f}m | Level {level} -> No Valid Data")
                continue
                
            # 조건별 평균(mean) 및 표준편차(std) 계산
            cov_m, cov_s = np.mean(res['coverage']), np.std(res['coverage'])
            ov_m, ov_s = np.mean(res['overlap_percent']), np.std(res['overlap_percent'])
            time_m, time_s = np.mean(res['cleaning_time']), np.std(res['cleaning_time'])
            
            print(f" ▶ Map Condition: {h_m:.1f}mx{w_m:.1f}m | Level {level} (Total tests: {len(res['coverage'])})")
            print(f"    - Coverage:     {cov_m*100:.2f}% ± {cov_s*100:.2f}%")
            print(f"    - Overlap Rate: {ov_m:.2f}% ± {ov_s:.2f}%")
            print(f"    - Cleaning Time: {time_m:.2f} min ± {time_s:.2f} min")
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