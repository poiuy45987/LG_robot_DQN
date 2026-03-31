import argparse
import os

from .agent import DQNAgent
from .config import DEFAULT_SEED

def parse_args():
    
    parser = argparse.ArgumentParser()
    
    # Debugging 여부 설정
    parser.add_argument('--debug', action='store_true', help='Debugging 여부 결정')
    
    # Seed 설정
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help=f'Set seed: (Default: {DEFAULT_SEED})')
    
    # Mode 설정, tensorboard와 wandb의 사용 여부를 결정
    parser.add_argument('--mode', choices=['train', 'test', 'see_map'], default='train', help='Mode: train, test, see_map')
    parser.add_argument('--use_wandb', action='store_true', help='wandb 사용')
    parser.add_argument('--use_tb', action='store_true', help='tb 사용')
    parser.add_argument('--use_vessl', action='store_true', help='vessl 사용')
    
    # Model과 tb 저장 경로 설정, 이어서 학습할 모델과 모델 이름 설정
    # pre_model_name: 특정 모델부터 학습을 진행하고 싶을 때 사용. Step 수 등이 초기화됨.
    # model_name: 저장할 model 이름 또는 test시 loading할 model 이름. 
    #             이 모델의 학습이 끝나지 않았을 경우, model_name을 가지고 checkpoint를 탐색한 뒤 최신 checkpoint부터 
    #             학습을 재개함.
    parser.add_argument('--model_dir', type=str, default='src/robot_vacuum_redqn/robot_vacuum_redqn/models', help='Model file name for saving or loading')
    parser.add_argument('--tb_save_dir', type=str, default='src/robot_vacuum_redqn/robot_vacuum_redqn/logs', help='Tensorboard save directory')
    parser.add_argument('--pre_model_name', type=str, default=None, help='Pre-trained model file name for continued training')
    parser.add_argument('--model_name', type=str, default='model.pth', help='Model file name for saving or loading')
    parser.add_argument('--best_path_img_name', type=str, default='best_coverage_path.png', help='Best path image file name')
    # parser.add_argument('--buffer_file_name', type=str, default='replay_buffer_latest.npz', help='Replay buffer file name')
    
    # ---- Training hyperparameter 설정 ----
    
    train_set_group = parser.add_argument_group('Training setting')
    
    # 데이터를 쌓는 warmup 과정 설정, Replay buffer 설정
    train_set_group.add_argument('--buffer_size', type=int, default=500000, help='Replay buffer size (Default: 500,000)')
    # train_set_group.add_argument('--warmup_episodes', type=int, default=20, help='Warmup을 수행하는 episode 수 (Default: 20)')
    train_set_group.add_argument('--warmup_ep_steps', type=int, default=10000, help='Warmup 시 episode당 step 수 (Default: 10,000)')
    train_set_group.add_argument('--warmup_tot_steps', type=int, default=200000, help='Warmup을 완료하는 최소 전체 step 수 (Default: 200,000)')
    
    # 훈련시킬 최대 episode 수 설정
    train_set_group.add_argument('--max_episodes', type=int, default=30000, help='Total episodes to train (Default: 30,000)')
    
    # batch_size 설정
    train_set_group.add_argument('--batch_size', type=int, default=64, help='Batch size for training (Default: 64)')
    
    # Optimizer 설정, Update 주기 설정
    train_set_group.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd', help='Optimizer to use for training (Default: sgd)')
    train_set_group.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer (Default: 1e-4)')
    train_set_group.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer  (Default: 0.9)')
    train_set_group.add_argument('--target_update', type=int, default=1000, help='Target network update frequency (steps) (Default: 1,000)')
    train_set_group.add_argument('--policy_update', type=int, default=20, help='Policy network update frequency (steps)  (Default: 20)')
    
    # Validation 주기 및 checkpoint 저장 주기 설정, Validation 설정
    train_set_group.add_argument('--valid_freq', type=int, default=100, help='Validation을 수행하는 주기 (episodes) (Default: 100)')
    train_set_group.add_argument('--ckp_freq', type=int, default=20, help='Checkpoint를 저장하는 주기 (episodes) (Default: 20)')
    train_set_group.add_argument('--valid_map_num', type=int, default=5, help='Validation 시 사용할 map 수 (Default: 100)')
    train_set_group.add_argument('--valid_start_point_num', type=int, default=3, help='Validation 시 한 map당 테스트해볼 start_poit 수 (Default: 20)')
    
    # Exploration 관련 설정: Episilon 기법, Softmax 기법, Noisy layer 사용
    train_set_group.add_argument('--use_epsilon', action='store_true', help='Use epsilon-greedy strategy')
    train_set_group.add_argument('--epsilon_start', type=float, default=1.0, help='Starting value of epsilon for epsilon-greedy policy (Default: 1.0)')
    train_set_group.add_argument('--epsilon_end', type=float, default=0.1, help='Final value of epsilon after decay (Default: 0.1')
    train_set_group.add_argument('--epsilon_decay', type=int, default=200000, help='Number of steps to decay epsilon from start to end (Default: 200,000)')
    
    train_set_group.add_argument('--use_softmax', action='store_true', help='Use softmax action selection instead of epsilon-greedy')
    train_set_group.add_argument('--softmax_temp', type=float, default=1.0, help='Temperature parameter for softmax action selection (Default: 1.0)')
    
    train_set_group.add_argument('--use_noisy', action='store_true', help='Use noisy layers in the network')
    train_set_group.add_argument('--target_with_noisy', action='store_true', help='Use noisy layers in the target network') 
    
    # Action masking 설정
    train_set_group.add_argument('--use_action_masking', action='store_true', help='Use action masking on training') 
    
    # State pre-processing 설정
    train_set_group.add_argument('--grid_map_size', type=int, default=51, help='Network input으로 넣어주는 grid map의 height와 width를 설정')
    train_set_group.add_argument('--do_normalize', action='store_true', help='Do normalize on grid map data')
    
    # Q-value 관련 설정
    train_set_group.add_argument('--gamma', type=float, default=0.99, help='Discount factor for future rewards (default: 0.99)')
    
    # Training을 하나의 map으로만 시킬지 결정
    train_set_group.add_argument('--reset_only_start_pos', action='store_true', help='Episode가 바뀌면 map을 변경하지 않고 시작점만 변경')
    
    # DQN extension 설정
    train_set_group.add_argument('--double_dqn', action='store_true', help='Double DQN을 사용할지 여부를 결정')
    
    # -------------------------
        
    # ---- Map 관련 설정 ----
    map_set_group = parser.add_argument_group('Map setting')
    
    map_set_group.add_argument('--grid_size', type=float, default=4.0, help='Grid size: 단위 cm (default: 4.0)')
    map_set_group.add_argument('--map_height', type=float, default=800.0, help='Map의 세로 길이: 단위 cm (default: 800.0)')
    map_set_group.add_argument('--map_width', type=float, default=800.0, help='Map의 가로 길이: 단위 cm (default: 800.0)')
    
    map_set_group.add_argument('--boundary_thickness', type=int, default=1, help='Map 전체를 둘러싼 벽의 두께: Grid 단위 (default: 1)')
    map_set_group.add_argument('--not_use_house_map', action='store_true', help='의자와 책상이 포함된 house map을 이용할지 말지 결정')
    
    map_set_group.add_argument('--num_tables', type=int, default=10, help='Table 수의 최댓값 (default: 10)')
    map_set_group.add_argument('--max_table_size', type=float, default=100.0, help='Table 한 변 길이의 최댓값: 단위 cm (default: 100.0)')
    map_set_group.add_argument('--min_table_size', type=float, default=80.0, help='Table 한 변 길이의 최솟값: 단위 cm (default: 80.0)')
    map_set_group.add_argument('--table_leg_size', type=float, default=8.0, help='Table 다리 두께: 단위 cm (default: 8.0)')
    
    map_set_group.add_argument('--chair_size', type=float, default=50.0, help='의자의 한 변 길이: 단위 cm (default: 50.0)')
    map_set_group.add_argument('--chairs_per_table_min', type=int, default=2, help='Table 하나에 배치하는 의자의 최소 개수 (default: 2)')
    map_set_group.add_argument('--chairs_per_table_max', type=int, default=4, help='Table 하나에 배치하는 의자의 최대 개수 (default: 4)')
    map_set_group.add_argument('--chair_leg_size', type=float, default=4.0, help='의자의 다리 두께: 단위 cm (default: 4.0)')
    map_set_group.add_argument('--chair_spread', type=float, default=15.0, help='의자가 책상으로부터 떨어진 정도: 단위 cm (default: 15.0)')
    
    map_set_group.add_argument('--small_obs_size_max', type=float, default=15.0, help='작은 장애물의 크기 최댓값: 단위 cm (default: 15.0)')
    map_set_group.add_argument('--small_obs_size_min', type=float, default=10.0, help='작은 장애물의 크기 최솟값: 단위 cm (default: 10.0)')
    map_set_group.add_argument('--window_size', type=float, default=100.0, help='작은 장애물을 배치하기 위한 window 크기: 단위 cm (default: 100.0)')
    map_set_group.add_argument('--small_obs_num_per_window_max', type=int, default=2, help='Window 하나에 작은 장애물을 배치하는 개수 최댓값 (default: 2)')
    map_set_group.add_argument('--small_obs_num_per_window_min', type=int, default=1, help='Window 하나에 작은 장애물을 배치하는 개수 최솟값 (default: 1)')
    # -----------------------
    
    # ---- Environment 관련 설정 ----
    
    env_set_group = parser.add_argument_group('Environment setting')
    
    env_set_group.add_argument('--max_steps', type=int, default=100000, help='Maximum steps per episode (Default: 100,000)')
    env_set_group.add_argument('--max_no_progress_steps', type=int, default=300, help='Coverage가 증가하지 않을 때 max_steps (Default: 300)')
    env_set_group.add_argument('--max_no_progress_steps_final', type=int, default=100000, help='Coverage가 높은 상태에서 coverage가 증가하지 않을 때 max_stes (Default: 500)')
    env_set_group.add_argument('--final_coverage_thres', type=int, default=0.9, help='Coverage가 높은 상태를 정의하는 threshold (Default: 0.9)')
    env_set_group.add_argument('--target_coverage', type=float, default=0.95, help='Target coverage (Default: 0.95)')
    env_set_group.add_argument('--local_view', type=float, default=200.0, help='Observation으로 출력할 local view의 크기: 단위 cm (Default: 200.0)')
    env_set_group.add_argument('--max_forward', type=float, default=50.0, help='한 방향으로 이동할 수 있는 최대 거리를 정규화하기 위한 수치: 단위 cm (Default: 50.0)')
    env_set_group.add_argument('--robot_size', type=float, default=36.0, help='로봇의 지름: 단위 cm (Default: 36.0)')
    env_set_group.add_argument('--stack_steps', type=int, default=3, help='Map의 observation data의 step 수 (Default: 3)')
    
    
    # Reward function 관련 설정
    env_set_group.add_argument('--uncleaned_reward', type=float, default=1.0, help='Uncleaned grid reward (Default: 1.0)')
    env_set_group.add_argument('--cleaned_penalty', type=float, default=-0.1, help='Cleaned grid penalty (Default: 0.1)')
    env_set_group.add_argument('--obstacle_penalty', type=float, default=-10.0, help='Obstalce penalty (Default: 10.0)')
    env_set_group.add_argument('--turn_penalty', type=float, default=-0.1, help='Turning penalty (Default: 0.1)')
    env_set_group.add_argument('--step_penalty', type=float, default=-0.01, help='Step penalty (Default: 0.01)')
    env_set_group.add_argument('--complete_reward', type=float, default=10.0, help='Complete_reward (Default: 10.0)')
    env_set_group.add_argument('--intrinsic_reward', type=float, default=1.0, help='Intrinsic_reward (Default: 1.0)')
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
    dqn_agent = DQNAgent(args)
    if args.mode == 'train':
        dqn_agent.train()
    elif args.mode == 'test':
        dqn_agent.test()
    elif args.mode == 'see_map':
        visualize_test_map(seed=args.seed)

if __name__ == "__main__":
    main()