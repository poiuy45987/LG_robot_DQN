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
import datetime
import pytz
import random
import time
import os

import glob
from PIL import Image
from IPython.display import display, clear_output

# 앞서 정의한 클래스들을 임포트한다고 가정 (또는 같은 파일에 위치)
from .config import EnvConfig, TrainConfig, DEFAULT_SEED, ACTION_NUM
from .environment import CoverageEnv
from .redqn_network import CNN_ReDQN

def get_device_info(device):
    
    print("-" * 30)
    print(f"  [System Configuration]")
    print(f"  > Device: {str(device).upper()}")
    
    if device.type == 'cuda':
        # 0 대신 현재 사용 중인 장치의 인덱스를 가져옵니다.
        current_idx = torch.cuda.current_device() 
        print(f"  > GPU Name: {torch.cuda.get_device_name(current_idx)}")
        # VRAM 정보까지 한 줄 추가하면 완벽!
        total_mem = torch.cuda.get_device_properties(current_idx).total_memory / 1e9
        print(f"  > VRAM: {total_mem:.2f} GB")
        
    print("-" * 30, flush=True)

class DQNAgent:
    
    def __init__(self, args):
        self.args = args
        self.seed = args.seed
        self.model_name = os.path.join(args.model_dir, args.model_name)
        
        # Device 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        get_device_info(self.device) # Device 정보 출력
        
        # Pytorch 난수 생성기의 seed 설정
        torch.manual_seed(args.seed) # PyTorch CPU 난수 고정
        if torch.cuda.is_available(): # PyTorch GPU 난수 고정
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed) # 멀티 GPU 사용 시
        # 결정론적 연산 설정 (속도는 조금 느려질 수 있지만 결과는 항상 동일)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Policy network 조성
        network_config = {
            'action_size': ACTION_NUM,
            'use_noisy': args.use_noisy,
            'grid_map_size': args.grid_map_size,
            'do_normalize': args.do_normalize,
        }
        self.policy_net = CNN_ReDQN(**network_config).to(self.device)
        
        # Environment 조성을 위한 config
        env_cfg = EnvConfig(**{k: v for k, v in vars(args).items() if k in EnvConfig.__dataclass_fields__})
        
        # Action masking 관련
        
        # Train과 관련된 instance 변수는 mode가 train일 때만 생성
        if args.mode == 'train':
            
            # Train environment와 validation environement 조성
            self.env = CoverageEnv(env_cfg, seed=args.seed)
            if args.reset_only_start_pos:
                self.valid_env = CoverageEnv(env_cfg, seed=args.seed) # 훈련용 environment와 같은 환경을 구성하기 위해서
            else:
                self.valid_env = CoverageEnv(env_cfg, seed=args.seed+1000) # 훈련용 environment와 다른 환경을 구성하기 위해서
                
            # Train hyperparameter를 받음
            self.train_cfg = TrainConfig(**{k: v for k, v in vars(args).items() if k in TrainConfig.__dataclass_fields__})         
            
            # Training에 필요한 변수 설정
            self.total_steps = 0                # Training을 진행한 steps 수 (Warmup 과정 포함)
            self.warmup_steps = 0               # Buffer에 data를 채우기 위한 warmup steps 수
            self.no_warmup_steps = 0            # total_steps에서 warmup 과정 동안 소요된 step 수를 제외한 step 수
            self.start_episode = 1              # 시작 episode 번호: Checkpoint을 loading할 때 수정될 수 있음
            self.best_coverage_mean = 0.0       # Validation을 수행했을 때 가장 높았던 coverage
            self.best_coverage_path_img = None  # Validation을 수행했을 때 가장 coverage를 잘 했던 trajectory를 img로 저장 (RGB image)
        
            # Target network, Replay buffer 조성
            self.target_net = CNN_ReDQN(**network_config).to(self.device)   # Target network
            self.memory = deque(maxlen=args.buffer_size)          # Replay Buffer
            
            # Training을 위한 난수 생성기의 seed 설정
            self.train_rng = np.random.default_rng(seed=args.seed)
            
            # ---- Training process를 지켜 볼 tool 설정: wandb 또는 tb ----
            kst = pytz.timezone('Asia/Seoul')
            current_time = datetime.datetime.now(kst).strftime("%y%m%d_%H%M")
            
            # Training 조건을 구별하기 위해 표시할 hyperparmeter 설정
            params_list_for_log = [
                'batch_size', 'lr', 'optimizer', 'momentum', 'epsilon_decay', 'use_softmax',
                'softmax_temp', 'use_noisy', 'target_with_noisy', 'gamma', 'reset_only_start_pos',
                'uncleaned_reward', 'cleaned_penalty', 'obstacle_penalty', 'turn_penalty', 'step_penalty'      
            ]
            params_config = {k: v for k, v in vars(args).items() if k in params_list_for_log}
            
            # TensorBoard 설정
            self.tb_writer = None
            if args.use_tb:
                tb_save_dir = os.path.join(args.tb_save_dir, current_time)
                if not os.path.exists(tb_save_dir):
                    os.makedirs(tb_save_dir)
                self.tb_writer = SummaryWriter(tb_save_dir)
                
            # wandb 설정
            self.wandb_run = None
            if args.use_wandb:
                self.wandb_run = wandb.init(
                    entity="lg-robot-cleaner",
                    project="Robot_vacuum_ReDQN",
                    config=params_config,
                    name=f"{current_time}_{args.model_name}_training"
                )
                
            # vessl 설정
            if args.use_vessl:
                vessl.init(
                    organization="snu-eng-gtx1080", 
                    project="lg-robot-ReDQN", 
                    hp=params_config,
                    # name=f"{current_time}_{args.model_name}_training"                  
                )
            # ---------------------------------------------------------
            
            # Optimizer 설정
            if args.optimizer == 'sgd':
                self.optimizer = optim.SGD(self.policy_net.parameters(), lr=args.lr, momentum=args.momentum)
            elif args.optimizer == 'adam':
                self.optimizer = optim.Adam(self.policy_net.parameters(), lr=args.lr)
            else:
                raise ValueError(f"Unsupported optimizer type: {args.optimizer}")
        
        elif args.mode == 'test':
            if args.reset_only_start_pos:
                self.test_env = CoverageEnv(env_cfg, seed=args.seed) # Test environment 구현
            else:
                self.test_env = CoverageEnv(env_cfg, seed=args.seed+2000) # Test environment 구현
            
        # Loading model
        self._load_model(args)

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
        # Model이 담긴 폴더: .../models
        model_dir = args.model_dir
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
            
        # Checkpoint가 담긴 폴더: .../models/(model_name)_checkpoint
        model_name_base, _ = os.path.splitext(args.model_name) # 확장자 '.pth'를 제거한 model_name
        checkpoint_dir = os.path.join(model_dir, model_name_base + "_checkpoints")
        if not os.path.isdir(checkpoint_dir): # Checkpoint 저장 폴더가 없으면 폴더를 생성 생성
            os.makedirs(checkpoint_dir)
        
        if args.mode == 'train':
            
            # 사전 학습된 모델을 사용하는 경우 loading 수행.
            if args.pre_model_name is not None:
                pre_trained_model_path = os.path.join(model_dir, args.pre_model_name)
                
                if not os.path.isfile(pre_trained_model_path): # Model이 없는 경우: Error
                    raise FileNotFoundError(f"Pre-trained model file not found: {pre_trained_model_path}")
                
                # /models 폴더에 저장된 model 정보를 불러옴
                pre_trained_model = torch.load(pre_trained_model_path, map_location=self.device, weights_only=False)
                self.policy_net.load_state_dict(pre_trained_model['model_state_dict'])
                self.target_net.load_state_dict(pre_trained_model['model_state_dict'])
                self.best_coverage_mean = pre_trained_model['best_coverage_mean']
                
                # 이전 modeld의 checkpoint 폴더에 저장된 best_coverage_path_img를 불러옴
                pre_model_name_base, _ = os.path.splitext(args.pre_model_name) # 확장자 '.pth'를 제거한 model_name
                pre_model_checkpoint_dir = os.path.join(model_dir, pre_model_name_base + "_checkpoints")
                if os.path.isdir(pre_model_checkpoint_dir): # Checkpoint 저장 폴더가 없으면 폴더를 생성 생성
                    best_coverage_path_img_path = os.path.join(checkpoint_dir, args.best_path_img_name)
                    if os.path.isfile(best_coverage_path_img_path):
                        loaded_img = cv2.imread(best_coverage_path_img_path) # BGR image
                        if loaded_img is not None: 
                            self.best_coverage_path_img = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2RGB) # BGR -> RGB 변환
                
                print(f"Loaded pre-trained model: {pre_trained_model_path}", flush=True)
            
            else: # 사전에 학습된 모델을 사용하지 않는 경우 checkpoint를 확인
                checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pth")) # 전체 checkpoint file 목록

                if checkpoints: # Checkpoint가 존재하면 이어서 학습을 진행
                    print("Checkpoint directory already exists. Continue training...", flush=True) # Checkpoint가 있으면 이어서 학습
                    
                    latest_checkpoint = max(checkpoints, key=os.path.getctime) # 가장 최근 파일 선택
                    print(f"Loading latest checkpoint: {latest_checkpoint}", flush=True)
                    checkpoint_data = torch.load(latest_checkpoint, map_location=self.device, weights_only=False)
                    self.policy_net.load_state_dict(checkpoint_data['model_state_dict'])
                    self.target_net.load_state_dict(checkpoint_data['model_state_dict'])
                    
                    try:
                        self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict']) # Optimizer도 checkpoint와 똑같이 유지
                    except (KeyError, ValueError):
                        print("Optimizer state mismatch. Initializing optimizer.")
                        # Optimizer를 다시 설정
                        if args.optimizer == 'sgd':
                            self.optimizer = optim.SGD(self.policy_net.parameters(), lr=args.lr, momentum=args.momentum)
                        elif args.optimizer == 'adam':
                            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=args.lr)
                        else:
                            raise ValueError(f"Unsupported optimizer type: {args.optimizer}")

                    self.total_steps = checkpoint_data['total_steps']
                    self.no_warmup_steps = checkpoint_data['no_warmup_steps']
                    self.start_episode = checkpoint_data['episode'] + 1
                    self.best_coverage_mean = checkpoint_data['best_coverage_mean']
                    
                    # checkpoint 폴더에 저장된 best_coverage_path_img를 불러옴
                    best_coverage_path_img_path = os.path.join(checkpoint_dir, args.best_path_img_name)
                    if os.path.isfile(best_coverage_path_img_path):
                        loaded_img = cv2.imread(best_coverage_path_img_path) # BGR image
                        if loaded_img is not None: 
                            self.best_coverage_path_img = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2RGB) # BGR -> RGB 변환
                    
                    print(f"Resumed training from episode {checkpoint_data['episode']} with reward {checkpoint_data['episode_reward']}", flush=True)

                    # Replay buffer loading
                    replay_buffer_file_path = os.path.join(checkpoint_dir, self.args.buffer_file_name)
                    self._load_memory(replay_buffer_file_path)
                    
                else:
                    print("No checkpoint files found!", flush=True)
                
        elif args.mode == 'test':
            
            test_model = os.path.join(model_dir, args.model_name)
            if not os.path.exists(test_model):
                raise FileNotFoundError(f"Test model file not found: {test_model}", flush=True)
            checkpoint = torch.load(test_model, map_location=self.device, weights_only=False)
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
    
    def _load_memory(self, file_path: str):
        
        # 파일 존재 여부 확인
        if not os.path.exists(file_path):
            print(f"No memory file found at: {file_path}")
            return
        
        # 저장한 메모리 불러오기
        data = np.load(file_path)
        obs_keys = [k[4:] for k in data.files if k.startswith('obs_')]
        
        # Dictionary 재조립 (Reconstruct dict lists)
        num_samples = len(data['action'])
        reconstructed_obs = []
        reconstructed_next_obs = []

        for i in range(num_samples):
            # 각 스텝마다 딕셔너리 생성
            obs_dict = {k: data[f'obs_{k}'][i] for k in obs_keys}
            next_obs_dict = {k: data[f'next_obs_{k}'][i] for k in obs_keys}
            
            reconstructed_obs.append(obs_dict)
            reconstructed_next_obs.append(next_obs_dict)

        # 4. 기존 self.memory 구조인 [(s, a, r, ns, d), ...] 형태로 zip
        self.memory = list(zip(
            reconstructed_obs,
            data['action'],
            data['reward'],
            reconstructed_next_obs,
            data['done']
        ))

    def _save_model(self, mode='model', info=None):
        
        # Model이 담긴 폴더명: .../models
        model_dir = self.args.model_dir
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir) # 모델을 담는 폴더가 없는 경우 폴더를 생성
        
        # Model을 저장하는 경로
        model_save_path = os.path.join(self.args.model_dir, self.args.model_name)
        
        # Checkpoint를 저장하는 폴더
        model_name_base, _ = os.path.splitext(self.args.model_name) # 확장자 '.pth'를 제거한 model_name
        checkpoint_dir = os.path.join(self.args.model_dir, model_name_base + "_checkpoints") # Checkpoint 저장 폴더
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        if mode == 'model':
            # model을 /models에 저장
            
            # Model 저장
            checkpoint = {
                'model_state_dict': self.policy_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_coverage_mean': self.best_coverage_mean,
            }
            torch.save(checkpoint, model_save_path)
            
            # Checkpoint 파일에 가장 coverage가 잘 수행된 trajectory 그림을 저장
            if self.best_coverage_path_img is not None:
                img_file_path = os.path.join(checkpoint_dir, self.args.best_path_img_name)
                img_bgr = cv2.cvtColor(self.best_coverage_path_img, cv2.COLOR_RGB2BGR) # OpenCV 저장용: (RGB -> BGR)
                cv2.imwrite(img_file_path, img_bgr)
                

        elif mode == 'checkpoint':
            
            assert info is not None
            episode = info['episode']; episode_reward = info['episode_reward'] # info에서 저장할 model의 episode 번호와 episode_reward 얻기
            
            # Checkpoint 저장 경로 설정
            model_name_base, _ = os.path.splitext(self.args.model_name) # 확장자 '.pth'를 제거한 model_name
            save_dir = os.path.join(self.args.model_dir, model_name_base + "_checkpoints") # Checkpoint 저장 폴더
            if not os.path.exists(save_dir): # Checkpoint 저장 폴더가 없으면 폴더를 생성 생성
                os.makedirs(save_dir)
            file_name = model_name_base + f"_{episode}_{episode_reward:.2f}.pth" # 파일 이름 형색 설정
            save_path = os.path.join(save_dir, file_name)
            
            # 체크포인트 데이터 구성
            checkpoint = {
                'episode': episode,
                'model_state_dict': self.policy_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'episode_reward': episode_reward,
                'total_steps': self.total_steps,
                'no_warmup_steps': self.no_warmup_steps,
                'best_coverage_mean': self.best_coverage_mean,
            }
                
            # 저장 실행
            torch.save(checkpoint, save_path)
            
            # Replay buffer 저장
            replay_buffer_file_path = os.path.join(checkpoint_dir, self.args.buffer_file_name)
            self._save_memory(replay_buffer_file_path)
            
        else:
            raise ValueError(f"Unsupported mode: {mode}. Expected 'model' or 'checkpoint'.")
    
    def _save_memory(self, file_path: str):
        # self.memory: [(s,a,r...), (s,a,r...)]
        # zip(*self.memory): (s,s...), (a,a...), (r,r...)
        obs_list, actions, rewards, next_obs_list, dones = zip(*self.memory)
        
        save_dict = {
            'action': np.array(actions, dtype=np.uint8),
            'reward': np.array(rewards, dtype=np.float32),
            'done': np.array(dones, dtype=np.bool_)
        }
        
        for key in obs_list[0].keys():
            save_dict[f'obs_{key}'] = np.array([o[key] for o in obs_list])
            save_dict[f'next_obs_{key}'] = np.array([no[key] for no in next_obs_list])
        
        np.savez_compressed(file_path, **save_dict)
     
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
        dx, dy = env.dir_vecs[action]
        nx = cx+dx; ny = cy+dy
        
        # Action masking 여부 계산
        if len(self.coverage_hist) >= ks:
            check_ks = (self.coverage_hist[-ks] == env.coverage)
            
        if (len(self.coverage_hist) >= kp) and (len(self.pos_hist) >= kp):
            before_kp_pos_x, before_kp_pos_y = self.pos_hist[-kp]
            dist = abs(cx-before_kp_pos_x) + abs(cy-before_kp_pos_y)
            check_kp = (self.coverage_hist[-kp] == env.coverage) and (dist <= 1)
        
        _, check_collision, _ = env.mark_trajectory(nx, ny, flag=False, virtual=True)
        
        # Coverage와 position의 history 저장
        self.coverage_hist.append(env.coverage)
        self.pos_hist.append(env.pos)
        
        return (check_ks or check_kp or check_collision)
        
        # except Exception as e:
        #     raise Exception(f"Unexpected error in action masking: {e}") from e
    
    # # RL policy와 heuristic action mode를 전환
    # def _switch_to_heuristic(self):
    #     self.get_heu_act = True
    
    # def _switch_to_RL_policy(self):
    #     self.get_heu_act = False
    
    def _get_action(self, env: CoverageEnv, processed_obs, mode: str='test', reset: bool = False) -> int:
        
        # # options 인자 받기
        # warmup = kwargs.get('warmup', False)
        # last_action = kwargs.get('last_action', None)
        # last_collision = kwargs.get('last_collision', False)
        
        # mode 입력값 검사
        if mode not in ['train', 'test']:
            raise ValueError(f"Unsupported mode: {mode}. Expected 'train' or 'test'.")
        
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
        action_RL = self._get_RL_action(state, mode) # Policy network를 통과시켜서 얻은 action
        # RL policy와 heuristic policy 중 무엇을 사용할지 결정
        if self._decide_action_masking(env, action_RL, mode, reset): # Heuristic action을 사용
            
            # action_RL을 수행: next_state, reward, terminated 및 truncated 여부 등을 받음.
            # environment 내에서 trajectory의 변화는 없지만, replay buffer에는 transition data를 저장함.
            # 수행한 뒤, action을 수행하기 전의 환경으로 돌아감.
            next_obs, reward, terminated, truncated, info = env.step(action_RL)
            next_processed_obs = self._pre_process_obs(next_obs, target_dim=self.train_cfg.grid_map_size)
            self.memory.append((processed_obs, action_RL, reward, next_processed_obs, terminated)) # processed_obs의 map data는 np.uint8 형태, Memory 용량을 줄임
            env.one_step_back()
            
            return self._get_heuristic_action(env) # Heuristic action을 return
        
        else: # RL policy action을 사용
            
            if not hasattr(self, 'dijkstra_traj'):
                self.dijkstra_traj = []
            
            # Heuristic 방법으로 만든 trajectory를 RL policy가 따라가는 경우, self.dijkstra_traj를 유지
            if self.dijkstra_traj:
                next_pos_x, next_pos_y = env.get_next_pos(action_RL)   # RL_policy를 따랐을 때 다음 위치
                next_dji_pos_x, next_dji_pos_y = self.dijkstra_traj.pop(0)  # dijkstra_traj에서의 다음 위치
                
                # 두 위치가 일치하지 않는 경우 dijkstra_traj를 초기화
                if next_pos_x != next_dji_pos_x or next_pos_y != next_dji_pos_y:
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
        
        if not hasattr(self, 'dijkstra_traj'):
            self.dijkstra_traj = []

        if len(self.dijkstra_traj) > 0: # 이미 생성한 경로가 있으면 다음 action을 출력
            next_pos = self.dijkstra_traj.pop(0)
            return env.get_next_action_from_next_pos(next_pos)
        
        # BFS 탐색으로 경로 탐색   
        queue = deque([env.pos])
        start = env.pos         # 현재 위치를 시작 지점으로 저장
        visited = {start}       # set 형태. 탐색을 빠르게 수행. BFS를 하면서 지나간 grid를 저장하기 위한 용도
        parent = {start: None}  # 해당 grid로 오기 위해 어떤 grid를 거쳤는지 저장.
        
        while queue:
            curr_pos = queue.popleft()
            for dir_vec in env.dir_vecs:
                next_pos = (curr_pos[0]+dir_vec[0], curr_pos[1]+dir_vec[1])
                new_cleaned_num, collided, _ = env.mark_trajectory(*next_pos, virtual=True) # FIXME
                if not collided and next_pos not in visited:
                    visited.add(next_pos)
                    parent[next_pos] = curr_pos
                    queue.append(next_pos)
                
                if new_cleaned_num > 0: # 새로운 grid를 밟을 수 있는 위치를 발견하면 그 위치로 이동하는 경로를 생성
                    self._get_dijkstra_traj_from_parent(parent, next_pos)
                    return env.get_next_action_from_next_pos(self.dijkstra_traj.pop(0))
        
        
    # State만 받는 것으로 수정    
    def _get_RL_action(self, state, mode: str = 'test') -> int:
        """
        환경과 state가 주어졌을 때 action을 얻는 method

        Args:
            env (CoverageEnv): action을 얻고자 하는 environment
            state: Q-network에 통과시키기 위한 state
            action_mask: 장애물에 충돌하는 action을 masking하기 위한 array
            use_masking (bool, optional): Action masking을 진행할지 결정. Defaults to True.
            mode (str, optional): 'train' 또는 'test' 중 선택할 mode를 결정. Defaults to 'test'.
            **kwargs: 추가적인 설정
                - warmup (bool): 현재가 buffer에 data를 쌓기 위한 warmup 과정인지 여부
                - last_action (int): 이전 step에서 수행한 action. 이전 step에서 충돌이 일어났을 때 다음 action으로 이 action을 제외하기 위해 받음
                - last_collision (bool): 이전 step에서 충돌했는지 여부.

        Returns:
            int: 선택한 action (0: E, 1: N, 2: W, 3: S)
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
        else:
            action = q_values.argmax().item()
        
        return action
            
            
    
    # # State만 받는 것으로 수정    
    # def _get_RL_action(self, env: CoverageEnv, state, action_mask, use_masking: bool=True, mode: str='test', **kwargs) -> int:
    #     """
    #     환경과 state가 주어졌을 때 action을 얻는 method

    #     Args:
    #         env (CoverageEnv): action을 얻고자 하는 environment
    #         state: Q-network에 통과시키기 위한 state
    #         action_mask: 장애물에 충돌하는 action을 masking하기 위한 array
    #         use_masking (bool, optional): Action masking을 진행할지 결정. Defaults to True.
    #         mode (str, optional): 'train' 또는 'test' 중 선택할 mode를 결정. Defaults to 'test'.
    #         **kwargs: 추가적인 설정
    #             - warmup (bool): 현재가 buffer에 data를 쌓기 위한 warmup 과정인지 여부
    #             - last_action (int): 이전 step에서 수행한 action. 이전 step에서 충돌이 일어났을 때 다음 action으로 이 action을 제외하기 위해 받음
    #             - last_collision (bool): 이전 step에서 충돌했는지 여부.

    #     Returns:
    #         int: 선택한 action (0: E, 1: N, 2: W, 3: S)
    #     """
    #     # action_mask는 test 시에는 반드시 사용. training 시에는 masking 유무를 선택
    #     # greedy action을 선택할 때만 사용. Warmup, epsilon 탐험, collision 이후 랜덤 탐험 시에는 action_mask 적용 X
        
    #     def _get_greedy_action(use_softmax=False) -> int:
    #         # Action을 Q-value로 뽑지 않더라도 일단 Q-value를 얻어서 관찰
    #         with torch.no_grad(): 
    #             if mode == 'train' and self.train_cfg.use_noisy:
    #                 self.policy_net.reset_noise() # 논문 기법: 결정 전 노이즈 리셋
    #             map_tensor = state['map']; vec_tensor = state['vec']
    #             q_values = self.policy_net(map_tensor, vec_tensor) # Shape: (1, action_space)
            
    #         # 사방이 막힌 경우
    #         if not torch.any(action_mask > 0):
    #             return q_values.argmax().item()
            
    #         # Action masking 여부에 따라 action_mask를 수정
    #         masked_q_values = q_values.clone()
    #         if use_masking:
    #             masked_q_values[action_mask == 0] = -1e9
                
    #         if use_softmax:
    #             probs = F.softmax(masked_q_values / self.train_cfg.softmax_temp, dim=1).cpu().numpy().flatten() # 확률값 계산
    #             if use_masking:
    #                 probs[action_mask.cpu().numpy().flatten() == 0] = 0  # action_mask가 0인 곳은 확률을 0으로 만듦
    #             probs = probs / (probs.sum() + 1e-8) # 확률 합이 1이 안 될 경우를 대비해 normalize (안전장치)
    #             action = self.train_rng.choice(len(probs), p=probs)
    #         else:
    #             action = masked_q_values.argmax().item()
            
    #         return action
        
    #     if mode == 'train':
            
    #         # options 인자 받기
    #         warmup = kwargs.get('warmup', False)
    #         last_action = kwargs.get('last_action', None)
    #         last_collision = kwargs.get('last_collision', False)
            
    #         # 1. 이전 step에서 collision이 일어났으면 이전 step에서 수행한 action을 제외하고 action을 랜덤 선택: Warmup인 경우에도 똑같이 수행
    #         if last_collision and last_action is not None: 
    #             all_actions = list(range(env.action_space.n))
    #             if last_action in all_actions:
    #                 all_actions.remove(last_action)
    #             return self.train_rng.choice(all_actions)
            
    #         # 2. Warmup 상황인 경우 action을 random하게 뽑음
    #         if warmup: 
    #             return env.action_space.sample()
                
    #         # 3. Epsilon 기법을 사용하는 경우
    #         if self.train_cfg.use_epsilon: 
                
    #             # Epsilon 결정: Step 수가 늘어날수록 epsilon을 점점 줄임. Warmup 과정에서는 유지
    #             epsilon = max(self.train_cfg.epsilon_end, self.train_cfg.epsilon_start-self.no_warmup_steps/self.train_cfg.epsilon_decay)
    #             if self.train_rng.random() < epsilon:
    #                 return env.action_space.sample()
            
    #         # 4. 그 외에는 전부 greedy action을 사용
    #         return _get_greedy_action(use_softmax=self.train_cfg.use_softmax)
        
    #     elif mode == 'test':
    #         return _get_greedy_action(use_softmax=False)
        
    #     else:
    #         raise ValueError(f"Unsupported mode: {mode}. Expected 'train' or 'test'.")
        
    def _pre_process_obs(self, obs, target_dim=51) -> dict:
        
        # map data 변환: H와 W를 target_dim으로 변환
        hwc_map = np.transpose(obs['map'], (1, 2, 0)) # obs의 map data가 (C, H, W) 형태이므로 이를 (H, W, C) 형태로 변환
        resized_hwc_map = cv2.resize(hwc_map, (target_dim, target_dim), interpolation=cv2.INTER_NEAREST)
        processed_map = np.transpose(resized_hwc_map, (2, 0, 1))

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
        coverage = []
        max_coverage = 0.0
        coverage_mean = 0.0
        max_coverage_traj_img = None # 가장 성능이 좋았던 map에서의 trajectory를 보여줌
        options={"reset_only_start_pos": True} # 시작 지점만 초기화하기 위한 option
        
        # Coverage 성능을 평가
        for _ in range(self.train_cfg.valid_map_num):
            
            # Validation 환경을 초기화
            if self.train_cfg.reset_only_start_pos:
                obs, _ = self.valid_env.reset(options=options)
            else:
                obs, _ = self.valid_env.reset()
                
            for start_num in range(self.train_cfg.valid_start_point_num):
                
                # start_num == 0인 경우 map을 초기화하면서 시작 지점도 초기화가 되었으므로 reset을 실행하지 않음.
                if start_num != 0:
                    obs, _ = self.valid_env.reset(options=options) # 시작 지점 초기화
                cur_coverage = self._test_one_map(self.valid_env, obs) # Coverage 성능을 평가
                
                # Coverage 값이 유효한 경우에만 저장: Reachable grid의 수가 전체 grid 수의 절반은 넘어야 함.
                if self.valid_env.coverable.sum() >= self.valid_env.H * self.valid_env.W * 0.5:
                    coverage.append(cur_coverage) # Coverage 평균을 구하기 위해 cur_coverage를 저장
                    if cur_coverage > max_coverage:
                        max_coverage = cur_coverage
                        max_coverage_traj_img = self.valid_env.get_visualized_img()
        
        if coverage:
            coverage_mean = np.mean(coverage)
        
        # Coverage 성능 평가 후 이전 model의 coverage 성능보다 더 좋으면 model을 저장
        if coverage_mean > self.best_coverage_mean:
            self.best_coverage_mean = coverage_mean
            self.best_coverage_path_img = max_coverage_traj_img
            self._save_model(mode='model') # Coverage 성능이 가장 좋았던 model을 저장
        
        # Validation 결과를 tensorboard와 wandb에 기록
        if self.tb_writer:
            self.tb_writer.add_scalar('Validation/Coverage_mean', coverage_mean, episode)
        if self.wandb_run:
            self.wandb_run.log({'Validation/Coverage_mean': coverage_mean,
                                'Validation/Best_path': wandb.Image(max_coverage_traj_img)}, step=self.total_steps)
        if self.args.use_vessl:
            vessl.log(step=episode, payload={'Validation/Coverage_mean': coverage_mean,
                                             'Validation/Best_path': vessl.Image(max_coverage_traj_img)})
        
        # Validation 결과 출력
        print(f"[Validation] Episode {episode}: Coverage Mean = {coverage_mean:.4f}, Best Coverage Mean = {self.best_coverage_mean:.4f}", flush=True)
            
        self.policy_net.train() # train mode로 전환
        
    def _test_one_map(self, env: CoverageEnv, obs: dict, debug: bool = False) -> float:
        
        done = False
        reset = True
        last_obs = obs
        last_info = None
        debug_skip_count = 0
        
        # [1] 디버그 모드일 때 사용할 도화지(fig)를 미리 딱 한 번만 만듭니다.
        if debug:
            fig, axes = plt.subplots(2, 1, figsize=(18, 18))
            # 초기 이미지
            init_traj_img = env.get_visualized_img(img_choice='traj')
            init_obs_img = env.get_visualized_img(img_choice='obs')
            # 초기 빈 이미지 설치
            im_traj = axes[0].imshow(init_traj_img)
            im_obs = axes[1].imshow(init_obs_img)
            text_traj = axes[0].text(0.5, 0, "", transform=axes[0].transAxes, ha="center", fontsize=15, color='black')
            axes[0].set_title("Trajectory", fontsize=20)
            axes[1].set_title("Observation", fontsize=20)
            for ax in axes: ax.axis('off')
            plt.tight_layout()
            
            # Jupyter용 디스플레이 핸들을 생성 (이걸 통해 이미지만 쏙 바꿉니다)
            display_handle = display(fig, display_id=True)
            plt.close(fig) # 별도의 정적 출력이 생기지 않도록 닫기
        
        while not done:
            processed_obs = self._pre_process_obs(last_obs, target_dim=self.train_cfg.grid_map_size)
            action = self._get_action(env, processed_obs, mode='test', reset=reset)
            
            if debug: # FIXME: backstep 가는 method 추가
                
                if debug_skip_count > 0:
                    debug_skip_count -= 1
                else:
                    
                    with torch.no_grad():
                        map_tensor = torch.from_numpy(processed_obs['map']).float().to(self.device).unsqueeze(0)
                        vec_tensor = torch.from_numpy(processed_obs['vec']).to(self.device).unsqueeze(0)
                        q_values = self.policy_net(map_tensor, vec_tensor).squeeze().cpu().numpy()
                    
                    action_list = ['E', 'N', 'W', 'S']
                    action_info = (f"[Selected action]: {action_list[action]}\n"
                                   f"[Q-values] E: {q_values[0]:.2f}, N: {q_values[1]:.2f}, W: {q_values[2]:.2f}, S: {q_values[3]:.2f}")
                    
                    # [3] 데이터만 가져와서 기존 이미지 객체에 덮어쓰기 (가장 핵심)
                    traj_img = env.get_visualized_img(img_choice='traj')
                    obs_img = env.get_visualized_img(img_choice='obs')
                    
                    im_traj.set_data(traj_img)
                    im_obs.set_data(obs_img)
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
                            env.backstep(backstep_num)
                            continue
                        elif val > 0:
                            debug_skip_count = int(user_val) - 1
                        else:
                            continue
                    except ValueError:
                        print("숫자 또는 'q'를 입력해주세요.")

            next_obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            last_obs = next_obs
            last_info = info
            
            reset = False
        
        #print(env.traj)
        return last_info['Coverage']        
    
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
            
            # Options    
            if self.train_cfg.reset_only_start_pos:
                env_options = {"reset_only_start_pos": True}
            else:
                env_options = {"reset_only_start_pos": False}
            
            # Environment reset
            obs, info = self.env.reset(seed=seed, options=env_options)
            processed_obs = self._pre_process_obs(obs, target_dim=self.train_cfg.grid_map_size) # Map data를 resize, Observation data를 얻음
            # ----------------------------------------------------------------------------------------
            
            # Environment에서 step을 처음 시작할 때 설정
            episode_reward = 0  # Episode에서 얻은 reward 총합
            steps = 0           # Episode의 진행한 step 수
            done_ep = False     # Episode 종료 조건: truncated(Episode를 조기 종료) | (terminate & Success) (Collision인 경우는 종료 X)
            warmup = False      # 현재 buffer에 data를 쌓고만 있는지 parameter도 같이 update 중인지 결정
            reset = True        # 첫 번째 action을 선택할 때, action masking을 위한 buffer를 초기화
            
            # FIXME: last_action, last_collision 삭제
            # # Collision 시 다음 step에서는 이전에 수행한 action을 하지 않기 위해 이전에 수행한 action을 저장
            # last_action = None      # 이전 step에서 수행한 action
            # last_collision = False  # 이전 step에서 collision이 일어났는지 여부
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
                
            # ~~~~ Episode 내에서 training 수행 ~~~~
            while not done_ep:
                
                # Steps 수 세기
                self.total_steps += 1
                steps += 1
                if warmup:
                    self.warmup_steps += 1
                else:
                    self.no_warmup_steps += 1
                
                # Episode 수 저장
                if self.wandb_run:
                    self.wandb_run.log({"Train/Episodes": episode}, step=self.total_steps)
                
                # Action 선택: Warmup 중에는 100% random, 그 이후에는 epsilon 기법 사용
                action = self._get_action(self.env, processed_obs, mode='train', reset=reset)
                reset = False
                
                # if self.get_heu_act:
                #     action = self._get_heuristic_action(self.env)
                # else:
                #     action = self._get_RL_action(self.env, state, action_mask, use_masking=self.train_cfg.use_action_masking, mode='train', 
                #                                  warmup=warmup, last_action=last_action, last_collision=last_collision)
                
                # ----------------------------- Action 수행 ---------------------------------
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                next_processed_obs = self._pre_process_obs(next_obs, target_dim=self.train_cfg.grid_map_size)
                
                # Episode를 종료하는 조건: Coverage에 성공한 경우 or Episode가 조기 종료된 경우 (Collision 포함)
                # Warmup 시에는 truncated일 때 info['Steps'] == args.warmup_ep_steps일 때만 종료
                if warmup:
                    done_ep = terminated or (truncated & (info['Steps'] >= self.train_cfg.warmup_ep_steps))
                else:
                    done_ep = terminated or truncated
                
                done = terminated # Buffer에 done이라고 저장하는 조건: Collision & Coverage 성공
                # ---------------------------------------------------------------------------
                
                # FIXME: last_action, last_collision 삭제
                # last_action, last_collision, last_info 저장
                # last_action = action
                # last_collision = info['Collision']
                last_info = info
                
                # 메모리 저장
                self.memory.append((processed_obs, action, reward, next_processed_obs, done)) # processed_obs의 map data는 np.uint8 형태, Memory 용량을 줄임
                
                # State update
                processed_obs = next_processed_obs
                
                # Episode reward 계산
                episode_reward += reward             
                
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
                    # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
                    
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
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            # log로 보낼 info data를 얻음
            coverage = last_info.get("Coverage", 0.0)
            ep_collision = last_info.get("Episode_collision", 0)
            
            # 매 episode 마다 TensorBoard 또는 wandb 기록
            if self.tb_writer:
                self.tb_writer.add_scalar("Stats/Episode_reward", episode_reward, episode)
                self.tb_writer.add_scalar("Stats/Coverage_rate", coverage, episode)
                self.tb_writer.add_scalar("Stats/Collision_count", ep_collision, episode)
            if self.wandb_run:
                self.wandb_run.log({"Stats/Episode_reward": episode_reward,
                           "Stats/Coverage_rate": coverage,
                           "Stats/Collision_count": ep_collision}, step=self.total_steps)
            if self.args.use_vessl:
                vessl.log(step=episode, payload={
                    "Stats/Episode_reward": episode_reward,
                    "Stats/Coverage_rate": coverage,
                    "Stats/Collision_count": ep_collision
                })
            
            # Checkpoint 저장
            if not warmup and episode % self.train_cfg.ckp_freq == 0:
                checkpoint_info = {"episode": episode, "episode_reward": episode_reward}
                self._save_model(mode='checkpoint', info=checkpoint_info)
                
            # Validation 수행
            if not warmup and episode % self.train_cfg.valid_freq == 0:
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

    def test(self):
        self.policy_net.eval() # eval mode로 전환
        coverage = []
        max_coverage = 0.0
        min_coverage = 100.0
        coverage_mean = 0.0
        options={"reset_only_start_pos": True} # 시작 지점만 초기화하기 위한 option
        reset_seed = self.seed
        
        # Coverage 성능을 평가
        for _ in range(self.args.valid_map_num):
            
            obs, _ = self.test_env.reset(seed=reset_seed) # Test 환경을 reset하여 초기 state 얻음
                
            for start_num in range(self.args.valid_start_point_num):
                
                # start_num == 0인 경우 map을 초기화하면서 시작 지점도 초기화가 되었으므로 reset을 실행하지 않음.
                if start_num != 0:
                    obs, _ = self.test_env.reset(options=options) # 시작 지점 초기화
                cur_coverage = self._test_one_map(self.test_env, obs, self.args.debug) # Coverage 성능을 평가
                self.test_env.show_visualized_img(img_choice='traj') # trajectory 시각화
                
                # Coverage 값이 유효한 경우에만 저장: Reachable grid의 수가 전체 grid 수의 절반은 넘어야 함.
                if self.test_env.reachable.sum() >= self.test_env.H * self.test_env.W * 0.5:
                    coverage.append(cur_coverage) # Coverage 평균을 구하기 위해 cur_coverage를 저장
                    if cur_coverage > max_coverage:
                        max_coverage = cur_coverage
                    if cur_coverage < min_coverage:
                        min_coverage = cur_coverage
            
            reset_seed = None
        
        if coverage:
            coverage_mean = np.mean(coverage)
            
        print(f"[Test result]\n"
              f"Coverage mean: {coverage_mean*100:.2f}%\n"
              f"Max coverage:  {max_coverage*100:.2f}%\n"
              f"Min coverage:  {min_coverage*100:.2f}%") 