import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

from path_planner.LSTM_agent import LSTMAgent
from path_planner.config import DEFAULT_SEED, MAP_SAVE_DIR, MODEL_SAVE_DIR, TB_SAVE_DIR

def parse_args():
    
    parser = argparse.ArgumentParser()
    
    # Debugging 여부 설정
    parser.add_argument('--debug', action='store_true', help='Test 시 debugging 여부 결정')
    
    # Seed 설정
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help=f'Set seed: (Default: {DEFAULT_SEED})')
    
    # Mode 설정, tensorboard와 wandb의 사용 여부를 결정
    parser.add_argument('--mode', choices=['train', 'test', 'see_map', 'see_weight'], default='train', help='Mode: train, test, see_map, see_weight (see_map: seed로 생성한 map을 보는 기능 / see_weight: 훈련된 model의 parameter 중 일부를 시각화 하는 기능)')
    parser.add_argument('--use_wandb', action='store_true', help='wandb 사용')
    parser.add_argument('--use_tb', action='store_true', help='tb 사용')
    # parser.add_argument('--use_vessl', action='store_true', help='vessl 사용')
    
    # Model, map, tensorboard 저장 경로 설정
    parser.add_argument('--model_dir', type=str, default=MODEL_SAVE_DIR, help=f'Model 파일이 저장되는 폴더 경로: {MODEL_SAVE_DIR}')
    parser.add_argument('--map_save_dir', type=str, default=MAP_SAVE_DIR, help=f'Map 파일이 저장되어 있는 폴더 경로: {MAP_SAVE_DIR}')
    parser.add_argument('--tb_save_dir', type=str, default=TB_SAVE_DIR, help=f'Tensorboard 데이터가 저장되는 폴더 경로: {TB_SAVE_DIR}')

    # 모델 이름 또는 이어서 학습할 모델 이름 설정
    parser.add_argument('--pre_model_name', type=str, default=None, help='Pre-trained model file 이름. Parameter만 불러오고 step 수, optimizer 상태 등은 초기화됨.')
    parser.add_argument('--model_name', type=str, default='model.pth', help='저장하거나 불러올 model 이름 설정. Train mode에서 이 model의 checkpoint가 있을 경우, 최신 checkpoint부터 학습을 재개. 확장자 .pth를 쓰거나 확장자를 아예 쓰지 않고 입력.')
    parser.add_argument('--best_traj_img_name', type=str, default='best_coverage_path.png', help='Model이 생성한 최적 경로 이미지의 파일 이름')

    # Test 시 사용할 map, map 개수 등 설정
    parser.add_argument('--test_map_folder_name', type=str, default='test', help=f'Test 시 사용할 map이 담긴 폴더 이름 설정. {MAP_SAVE_DIR}에 담긴 폴더 중에 선택. (Default: test)')
    parser.add_argument('--test_map_num_per_level', type=int, default=25, help='Test 시 map level별로 사용할 map 수 (Default: 25)')
    parser.add_argument('--test_start_point_num', type=int, default=3, help='Test 시 한 map당 테스트해볼 start_poit 수 (Default: 3)')
    parser.add_argument('--not_use_maps_folder', action='store_true', help='Test 시, maps 폴더에 저장된 map이 아닌 random하게 생성한 map을 사용')
    parser.add_argument('--vis_test_map_num', type=int, default=3, help='Test 시, level별로 성능이 좋은 map, 중간인 map, 안 좋은 map을 몇 개씩 보여줄지 결정 (Default: 3)')
    
    # Model dimension 관련 설정
    parser.add_argument('--action_enc_dim', type=int, default=128, help='Encoded action의 dimension')
    parser.add_argument('--lstm_in_prj_dim', type=int, default=32, help='LSTM cell block의 input vector의 dimension')
    parser.add_argument('--lstm_hid_dim', type=int, default=512, help='LSTM cell block의 hidden state vector의 dimension')
    
    # ---- Training hyperparameter 설정 ----
    
    train_set_group = parser.add_argument_group('Training setting')
    
    # 데이터를 쌓는 warmup 과정 설정, Replay buffer 설정
    train_set_group.add_argument('--use_train_maps', action='store_true', help='Train 시 maps 폴더에 저장된 map을 training set으로 이용')
    train_set_group.add_argument('--max_step_per_eps', type=int, help='Train 시 한 map에 대해서 훈련시키는 최대 횟수 (Default: 20)')
    train_set_group.add_argument('--max_eps_per_map_size', type=int, help='Train 시 한 map size에 대해서 훈련시키는 episode 최대 횟수 (Default: 100)')
    train_set_group.add_argument('--path_reward_thres', type=float, help='Train 시 한 map size를 바꾸는 path reward threshold (Default: 0.85)')
    
    # 훈련시킬 최대 episode 수 설정
    train_set_group.add_argument('--max_episodes', type=int, help='Total episodes (Default: 30,000)')
    
    # batch_size 설정
    train_set_group.add_argument('--batch_size', type=int, help='Batch size (Default: 64)')
    
    # Optimizer 설정, Update 주기 설정
    train_set_group.add_argument('--optimizer', choices=['sgd', 'adam'], help='Optimizer 설정: sgd, adam (Default: sgd)')
    train_set_group.add_argument('--lr', type=float, help='Learning rate for the optimizer (Default: 1e-4)')
    train_set_group.add_argument('--momentum', type=float, help='Momentum for SGD optimizer  (Default: 0.9)')
    train_set_group.add_argument('--min_lr', type=float, help='Learning rate scheduler의 최소 learning rate (Default: 1e-6)')
    train_set_group.add_argument('--scheduler_max_step', type=int, help='Cosine Annealing learning rate scheuler가 최저 learning rate까지 도달하기 위한 episode 수 (Default: 500)')
    train_set_group.add_argument('--detach_period', type=int, help='LSTM detaching period (Default: 20)')
    
    # Validation 주기 및 checkpoint 저장 주기 설정, Validation 설정
    train_set_group.add_argument('--valid_freq', type=int, help='Validation을 수행하는 주기 (episodes) (Default: 100)')
    train_set_group.add_argument('--ckp_freq', type=int, help='Checkpoint를 저장하는 주기 (episodes) (Default: 20)')
    train_set_group.add_argument('--valid_map_num', type=int, help='Validation 시 사용할 map 수 (Default: 5)')
    train_set_group.add_argument('--valid_start_point_num', type=int, help='Validation 시 한 map당 테스트해볼 start_poit 수 (Default: 3)')
    # -------------------------
        
    # ---- Environment 관련 설정 ----
    
    env_set_group = parser.add_argument_group('Environment setting')
    
    env_set_group.add_argument('--max_steps', type=int, help='Maximum steps per episode (Default: 1,500)')
    env_set_group.add_argument('--max_no_progress_steps', type=int, help='Coverage가 증가하지 않을 때 max_steps (Default: 60)')
    env_set_group.add_argument('--max_no_progress_steps_final', type=int, help='Coverage가 높은 상태에서 coverage가 증가하지 않을 때 max_stes (Default: 100)')
    env_set_group.add_argument('--final_coverage_thres', type=int, help='Coverage가 높은 상태를 정의하는 threshold (Default: 0.90)')
    env_set_group.add_argument('--target_coverage', type=float, help='Target coverage (Default: 0.95)')
    env_set_group.add_argument('--local_view', type=float, help='Observation으로 출력할 local view의 크기: 단위 cm (Default: 200.0)')
    env_set_group.add_argument('--max_forward', type=float, help='한 방향으로 이동할 수 있는 최대 거리를 정규화하기 위한 수치: 단위 cm (Default: 50.0)')
    env_set_group.add_argument('--robot_size', type=float, help='로봇의 지름: 단위 cm (Default: 36.0)')
    env_set_group.add_argument('--stack_steps', type=int, help='Map의 observation data의 step 수 (Default: 1)')
    
    # Reward function 관련 설정
    env_set_group.add_argument('--uncleaned_reward', type=float, help='Uncleaned grid reward (Default: 1.0)')
    env_set_group.add_argument('--cleaned_penalty', type=float, help='Cleaned grid penalty (Default: 0.1)')
    env_set_group.add_argument('--obstacle_penalty', type=float, help='Obstalce penalty (Default: 10.0)')
    env_set_group.add_argument('--turn_penalty', type=float, help='Turning penalty (Default: 0.1)')
    env_set_group.add_argument('--step_penalty', type=float, help='Step penalty (Default: 0.01)')
    env_set_group.add_argument('--complete_reward', type=float, help='Complete_reward (Default: 10.0)')
    env_set_group.add_argument('--intrinsic_reward', type=float, help='Intrinsic_reward (Default: 1.0)')
    # -----------------------------
    
    args = parser.parse_args()
    
    # model_name에 확장자가 없으면 추가
    name, ext = os.path.splitext(args.model_name)
    if not ext:
        args.model_name = name + ".pth"
    
    return args


def visualize_test_map(seed: int = DEFAULT_SEED):
    """
    생성한 맵을 빠르게 보고싶을 때 사용하는 method

    Args:
        seed (int, optional): 환경 구성에 사용하는 seed 번호. Defaults to None.
    """
    
    from .config import EnvConfig
    from .environment import CoverageEnv
    
    if seed is None:
        seed = DEFAULT_SEED
        
    cfg = EnvConfig()
    env = CoverageEnv(cfg, seed=seed)
    env.reset()
    env.show_visualized_img(img_choice = 'traj')
     
    
def main():
    
    # args parsing
    args = parse_args()
    lstm_agent = LSTMAgent(args)
    if args.mode == 'train':
        lstm_agent.train()
    elif args.mode == 'test':
        lstm_agent.test()
    elif args.mode == 'see_map':
        visualize_test_map(seed=args.seed)
    elif args.mode == 'see_weight':
        lstm_agent.see_weight()

if __name__ == "__main__":
    main()