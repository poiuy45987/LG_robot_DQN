import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

from path_planner.LSTM_agent import LSTMAgent
from path_planner.config import DEFAULT_SEED, MAP_SAVE_DIR

def parse_args():
    
    parser = argparse.ArgumentParser()
    
    # Debugging 여부 설정
    parser.add_argument('--debug', action='store_true', help='Debugging 여부 결정')
    
    # Seed 설정
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help=f'Set seed: (Default: {DEFAULT_SEED})')
    
    # Mode 설정, tensorboard와 wandb의 사용 여부를 결정
    parser.add_argument('--mode', choices=['train', 'test', 'see_weight'], default='train', help='Mode: train, test, see_map')
    parser.add_argument('--use_wandb', action='store_true', help='wandb 사용')
    parser.add_argument('--use_tb', action='store_true', help='tb 사용')
    parser.add_argument('--use_vessl', action='store_true', help='vessl 사용')
    
    # Model과 tb 저장 경로 설정, 이어서 학습할 모델과 모델 이름 설정
    # pre_model_name: 특정 모델부터 학습을 진행하고 싶을 때 사용. Step 수 등이 초기화됨.
    # model_name: 저장할 model 이름 또는 test시 loading할 model 이름. 
    #             이 모델의 학습이 끝나지 않았을 경우, model_name을 가지고 checkpoint를 탐색한 뒤 최신 checkpoint부터 
    #             학습을 재개함.
    parser.add_argument('--model_dir', type=str, default='src/path_planner/path_planner/models', help='Model file name for saving or loading')
    parser.add_argument('--tb_save_dir', type=str, default='src/path_planner/path_planner/logs', help='Tensorboard save directory')
    parser.add_argument('--pre_model_name', type=str, default=None, help='Pre-trained model file name for continued training')
    parser.add_argument('--model_name', type=str, default='model.pth', help='Model file name for saving or loading')
    parser.add_argument('--best_traj_img_name', type=str, default='best_coverage_path.png', help='Best path image file name')
    parser.add_argument('--map_save_dir', type=str, default=MAP_SAVE_DIR, help='Directory for saving generated maps')
    
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
    train_set_group.add_argument('--max_step_per_eps', type=int, help='Train 시 한 map에 대해서 훈련시키는 최대 횟수')
    train_set_group.add_argument('--max_eps_per_map_size', type=int, help='Train 시 한 map size에 대해서 훈련시키는 episode 최대 횟수')
    train_set_group.add_argument('--path_reward_thres', type=float, help='Train 시 한 map size를 바꾸는 path reward threshold')
    
    # 훈련시킬 최대 episode 수 설정
    train_set_group.add_argument('--max_episodes', type=int, help='Total episodes to train (Default: 30,000)')
    
    # batch_size 설정
    train_set_group.add_argument('--batch_size', type=int, help='Batch size for training (Default: 64)')
    
    # Optimizer 설정, Update 주기 설정
    train_set_group.add_argument('--optimizer', choices=['sgd', 'adam'], help='Optimizer to use for training (Default: sgd)')
    train_set_group.add_argument('--lr', type=float, help='Learning rate for the optimizer (Default: 1e-4)')
    train_set_group.add_argument('--momentum', type=float, help='Momentum for SGD optimizer  (Default: 0.9)')
    train_set_group.add_argument('--detach_period', default=20, type=int, help='LSTM detaching period (Default: 20)')
    
    # Validation 주기 및 checkpoint 저장 주기 설정, Validation 설정
    train_set_group.add_argument('--valid_freq', type=int, help='Validation을 수행하는 주기 (episodes) (Default: 100)')
    train_set_group.add_argument('--ckp_freq', type=int, help='Checkpoint를 저장하는 주기 (episodes) (Default: 20)')
    train_set_group.add_argument('--valid_map_num', type=int, help='Validation 시 사용할 map 수 (Default: 5)')
    train_set_group.add_argument('--valid_start_point_num', type=int, help='Validation 시 한 map당 테스트해볼 start_poit 수 (Default: 3)')
    # -------------------------
        
    # ---- Environment 관련 설정 ----
    
    env_set_group = parser.add_argument_group('Environment setting')
    
    env_set_group.add_argument('--max_steps', type=int, help='Maximum steps per episode (Default: 100,000)')
    env_set_group.add_argument('--max_no_progress_steps', type=int, help='Coverage가 증가하지 않을 때 max_steps (Default: 300)')
    env_set_group.add_argument('--max_no_progress_steps_final', type=int, help='Coverage가 높은 상태에서 coverage가 증가하지 않을 때 max_stes (Default: 500)')
    env_set_group.add_argument('--final_coverage_thres', type=int, help='Coverage가 높은 상태를 정의하는 threshold (Default: 0.9)')
    env_set_group.add_argument('--target_coverage', type=float, help='Target coverage (Default: 0.95)')
    env_set_group.add_argument('--local_view', type=float, help='Observation으로 출력할 local view의 크기: 단위 cm (Default: 200.0)')
    env_set_group.add_argument('--max_forward', type=float, help='한 방향으로 이동할 수 있는 최대 거리를 정규화하기 위한 수치: 단위 cm (Default: 50.0)')
    env_set_group.add_argument('--robot_size', type=float, help='로봇의 지름: 단위 cm (Default: 36.0)')
    env_set_group.add_argument('--stack_steps', type=int, help='Map의 observation data의 step 수 (Default: 3)')
    
    
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
    elif args.mode == 'see_weight':
        import numpy as np
        import matplotlib.pyplot as plt
        import torch.nn as nn
        
        model = lstm_agent.policy_net
        cmps_weight = model.cmps_net.weight.detach().cpu().numpy()

        map_dim = model.map_feat_dim
        vec_dim = model.vec_feat_dim
        lstm_dim = model.lstm_hid_dim

        # =========================================================================
        # 1. cmps_net 가중치 지분율 시각화 (Total Weight Share - Sum 기준)
        # =========================================================================
        w_map = cmps_weight[:, :map_dim]
        w_vec = cmps_weight[:, map_dim : map_dim + vec_dim]
        w_lstm = cmps_weight[:, map_dim + vec_dim :]

        # 각 파트별 가중치 절대값의 '총합' (진짜 지분율)
        imp_map = np.abs(w_map).sum()
        imp_vec = np.abs(w_vec).sum()
        imp_lstm = np.abs(w_lstm).sum()

        total = imp_map + imp_vec + imp_lstm
        ratios = [(imp_map / total) * 100, (imp_vec / total) * 100, (imp_lstm / total) * 100]
        labels = [f'Map Feat\n({map_dim}d)', f'Vector Feat\n({vec_dim}d)', f'LSTM State\n({lstm_dim}d)']

        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2, 1]})

        # [왼쪽] cmps_net Weight Matrix 전체 히트맵
        im = ax1.imshow(cmps_weight, cmap='seismic', aspect='auto', vmin=-np.max(np.abs(cmps_weight)), vmax=np.max(np.abs(cmps_weight)))
        ax1.set_title("cmps_net Weight Matrix Heatmap", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Input Feature Dimensions (Map | Vector | LSTM)")
        ax1.set_ylabel("Output Dimension (action_enc_dim)")

        # 그룹별 경계선(구분선) 그리기
        ax1.axvline(x=map_dim - 0.5, color='black', linestyle='--', linewidth=1.5)
        ax1.axvline(x=map_dim + vec_dim - 0.5, color='black', linestyle='--', linewidth=1.5)

        cbar = fig1.colorbar(im, ax=ax1)
        cbar.set_label('Weight Value')

        # [오른쪽] 그룹별 총 지분율 Bar Chart
        colors = ['#4C72B0', '#55A868', '#C44E52']
        bars = ax2.bar(labels, ratios, color=colors, alpha=0.85, width=0.5)
        ax2.set_title("Feature Group Importance Ratio (%)", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Total Weight Share Ratio (%)")
        ax2.set_ylim(0, max(ratios) * 1.2)

        for bar, pct in zip(bars, ratios):
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.show()

        
        # =========================================================================
        # 3. 최초 입력 3개 Map/Channel별 가중치(Weight) 비중 분석
        # =========================================================================
        # map_enc의 첫 번째 Conv2d 레이어 찾기
        first_conv = None
        for layer in model.map_enc:
            if isinstance(layer, nn.Conv2d):
                first_conv = layer
                break

        # w_first Shape: (32, 3, 3, 3) -> (out_channels, in_channels, K_h, K_w)
        w_first = first_conv.weight.detach().cpu().numpy()
        in_channels = w_first.shape[1]  # 보통 3 (또는 3 * stack_steps)

        # 입력 채널별 가중치 절대값 합(Sum) 및 평균(Mean) 계산
        ch_sums = [np.abs(w_first[:, c, :, :]).sum() for c in range(in_channels)]
        ch_means = [np.abs(w_first[:, c, :, :]).mean() for c in range(in_channels)]

        total_ch_sum = sum(ch_sums)
        ch_ratios = [(s / total_ch_sum) * 100 for s in ch_sums]

        # 입력 채널 라벨 설정 (필요시 이름 수정)
        ch_labels = [f'Input Ch {i+1}' for i in range(in_channels)]

        print("=" * 60)
        print("📊 [First Conv2d Layer - Input Channel Weight Analysis]")
        print("=" * 60)
        for i in range(in_channels):
            print(f"  - {ch_labels[i]} : Total Share = {ch_ratios[i]:5.2f}% | Mean |W| = {ch_means[i]:.6f}")
        print("=" * 60)

        # 시각화 (입력 채널별 가중치 비율 Bar Chart)
        fig3, (ax6, ax7) = plt.subplots(1, 2, figsize=(11, 4))
        ch_colors = ['#4C72B0', '#55A868', '#C44E52']

        # [좌] 입력 채널별 총 가중치 지분율 (%)
        bars6 = ax6.bar(ch_labels, ch_ratios, color=ch_colors[:in_channels], alpha=0.85, width=0.4)
        ax6.set_title("Input Map Channels Weight Share Ratio (%)", fontsize=11, fontweight='bold')
        ax6.set_ylabel("Weight Share (%)")
        ax6.set_ylim(0, max(ch_ratios) * 1.25)
        for bar, pct in zip(bars6, ch_ratios):
            ax6.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 1.0, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

        # [우] 입력 채널별 커널 파라미터 평균 크기
        bars7 = ax7.bar(ch_labels, ch_means, color=ch_colors[:in_channels], alpha=0.85, width=0.4)
        ax7.set_title("Input Map Channels Mean |Weight|", fontsize=11, fontweight='bold')
        ax7.set_ylabel("Mean Magnitude")
        ax7.set_ylim(0, max(ch_means) * 1.25)
        for bar, val in zip(bars7, ch_means):
            ax7.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.0005, f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()