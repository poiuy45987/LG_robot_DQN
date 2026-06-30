from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

if TYPE_CHECKING:
    from path_planner.map_layer import MapLayers


# FIXME: 좀 더 일반적인 형태로 바꿀 수 있을 듯.
# map_generator에서 색깔 목록과 범례 목록을 미리 주기
# config에서 정의한 장애물 종류를 map_generator에서 받고 그 목록을 미리 주기
# config에서 미리 색깔을 정의하는 것도 좋을 듯.
def get_map_img(obstacles: np.ndarray, fig: Figure, canvas: FigureCanvasAgg, 
                map_name: str, visualized: bool = True) -> np.ndarray:
    """
    Map을 canvas에 그린 후, 좌표축, 범례 등을 포함한 최종 이미지를 numpy array로 반환하는 method
    
    Args:
        obstacles (np.ndarray): Map data (2D array)
        fig (Figure): Map을 그릴 Figure 객체
        canvas (FigureCanvasAgg): fig에 그림을 그리는 canvas 객체
        map_name (str): Map 이름. 제목 설정에 사용됨.
        visualized (bool, optional): 그림이 시각화를 위해 장애물들을 다른 색으로 칠한 버전인지 여부. Defaults to True.

    Returns:
        np.ndarray: _description_
    """
    # fig와 canvas의 연결 보장
    if fig.canvas != canvas:
        canvas.__init__(fig) # 혹은 FigureCanvasAgg(fig)를 통해 내부 포인터를 동기화
    
    # Figure 설정
    fig.clear()
    fig.set_size_inches(6, 6)
    ax = fig.add_subplot(1, 1, 1)
    
    # Legend 설정 및 map 시각화
    if visualized:
        custom_cmap = ListedColormap(['white', 'black', 'red', 'blue', 'purple'])
        im = ax.imshow(obstacles, cmap=custom_cmap, origin='lower', vmin=0, vmax=4)
        legend_elements = [
            Patch(facecolor='black', edgecolor='black', label='Wall'),
            Patch(facecolor='red', edgecolor='red', label='Table Leg'),
            Patch(facecolor='blue', edgecolor='blue', label='Chair Leg'),
            Patch(facecolor='purple', edgecolor='purple', label='More obstacle')
        ]
    else:
        custom_cmap = ListedColormap(['white', 'black'])
        im = ax.imshow(obstacles, cmap=custom_cmap, origin='lower', vmin=0, vmax=1)
        legend_elements = [
            Patch(facecolor='black', edgecolor='black', label='Obstacle'),
        ]
    
    ax.legend(
        handles=legend_elements, 
        loc='upper left', 
        bbox_to_anchor=(1.05, 1), # 그래프 오른쪽 살짝 바깥에 배치
        title="Obstacle Types",
        title_fontsize='12',
        fontsize='10'
    )
    
    ax.set_title(f"{map_name}", fontsize=15)
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    
    fig.tight_layout()
    
    canvas.draw()
    
    return np.array(canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3] # [H, W, C]


def display_image(img_array: np.ndarray):
    """
    numpy 배열 이미지를 주피터 또는 시스템 뷰어에 최적화된 방식으로 보여주는 method
    """
    if img_array is None:
        return
        
    img_pil = Image.fromarray(img_array)
    
    try:
        # 주피터 셀 환경일 때: 여러 번 호출하면 셀 아래에 그림이 순서대로 쭉 나열됩니다.
        from IPython.display import display
        display(img_pil)
    except (ImportError, NameError):
        # 터미널 환경일 때: 시스템 이미지 뷰어로 창을 띄웁니다. 
        # 여러 번 호출하면 창이 여러 개 뜹니다.
        img_pil.show()


def visualize_saved_map(map_file_path: str):
    """
    시각화하고 싶은 map의 파일 이름을 입력하면, 이를 시각화해주는 method. mode에 train 또는 test를 입력하여
    해당 map이 train용인지 test용인지 적음.

    Args:
        map_folder_path (str): map 파일이 저장된 폴더 경로
        map_file_name (str): map 파일 이름
    """

    if not os.path.exists(map_file_path):
        print(f"Error: {map_file_path} 파일이 존재하지 않습니다.")
        return
    
    obstacles = np.load(map_file_path)
    map_file_name = os.path.basename(map_file_path)
    
    # Map 시각화
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    map_img = get_map_img(obstacles, fig=fig, canvas=canvas, map_name=map_file_name, visualized=False)
    display_image(map_img)


def visualize_mask(robot_mask: np.ndarray):
    plt.figure(figsize=(6, 6))
    plt.imshow(robot_mask, cmap='gray_r') # 0은 검정, 1은 흰색으로 표시
    plt.title(f"Robot Mask")
    plt.colorbar(label='Mask Value')
    plt.show()
    
    
def draw_layer(map_layers: MapLayers, fig: Figure, pos: tuple[float, float], last_coverage: float):
    
    fig.clear() # 이전 그림 지우기
    fig.set_size_inches(8, 8)

    axes = fig.subplots(2, 3)

    # obstacles rmfjg
    axes[0, 0].imshow(map_layers.obstacles, cmap='gray_r', origin='lower')
    axes[0, 0].set_title("Original Obstacles")

    # collision_map (Dilation 결과)
    axes[1, 0].imshow(map_layers.collision_map, cmap='gray_r', origin='lower')
    axes[1, 0].set_title("Collision Map (Dilation)")

    # cleaned
    # 현재 로봇 위치를 점으로 찍음
    axes[0, 1].imshow(map_layers.cleaned, cmap='gray_r', origin='lower')
    cx, cy = pos
    axes[0, 1].plot(cx, cy, 'r.') # 로봇 위치를 빨간 점으로 표시
    axes[0, 1].set_title(f"Cleaned Area (Coverage: {last_coverage:.2%})")
    
    # visited
    # 현재 로봇 위치를 점으로 찍음
    axes[1, 1].imshow(map_layers.uncleaned, cmap='gray_r', origin='lower')
    cx, cy = pos
    axes[1, 1].plot(cx, cy, 'r.') # 로봇 위치를 빨간 점으로 표시
    axes[1, 1].set_title(f"Uncleaned Area")
    
    # reachable
    axes[0, 2].imshow(map_layers.reachable, cmap='gray_r', origin='lower')
    axes[0, 2].set_title(f"Reachable grid")
    
    # coverable
    axes[1, 2].imshow(map_layers.coverable, cmap='gray', origin='lower')
    axes[1, 2].set_title(f"Coverable map")
    
    fig.tight_layout()
    
    
def draw_traj(map_layers: MapLayers, fig: Figure, pos: tuple[float, float], traj_arr: np.ndarray, 
              coverage: float, overlap_percent: float, cleaning_time: float, cleaned_map_max: int):
        
    fig.clear() # 이전 그림 지우기
    fig.set_size_inches(18, 8)
    agent_layer = map_layers.get_agent_layer(*pos)
    
    ax1 = fig.add_subplot(1, 2, 1) # 장애물, cleaned 영역, 로봇이 움직인 경로 등을 보여줌
    ax2 = fig.add_subplot(1, 2, 2) # 장애물, visited map을 보여줌. 로봇이 같은 지점을 얼마나 자주 방문했는지 보여줌.
    
    # --------------------------------------------------------------------
    # [Trajectory map 시각화]
    # --------------------------------------------------------------------
    traj_map = np.zeros_like(map_layers.obstacles)
    traj_map[map_layers.coverable == 0] = 4 # Cover 불가능한 영역을 칠함
    traj_map[map_layers.obstacles == 1] = 1 # Obstacle 표시
    traj_map[map_layers.cleaned > 0] = 2   # Cleaned 영역 표시
    traj_map[agent_layer == 1] = 3    # 현재 로봇 위치 표시
    
    # 색깔 설정
    color_bg = [1, 1, 1]                    # 흰색 (Index 0)
    color_obs = [0, 0, 0]                   # 검정 (Index 1)
    color_cleaned = [0.72, 0.88, 1.0]       # 빨강 (Index 2)
    color_robot = [0, 0, 1]                 # 진한 파랑 (Index 3)
    color_uncoverable = [1.0, 0.65, 0.65]   # 하늘색 (Index 4)
    color_traj = [0.1216, 0.4667, 0.7059]   # 연한 파랑
    color_start = 'purple'
    color_end = 'green'
    
    custom_cmap = ListedColormap([color_bg, color_obs, color_cleaned, color_robot, color_uncoverable])
    
    ax1.imshow(traj_map, cmap=custom_cmap, origin='lower', vmin=0, vmax=4)
    # 로봇이 움직인 궤적 표시
    if len(traj_arr) > 1:
        ax1.plot(traj_arr[:, 0], traj_arr[:, 1], color=color_traj, linewidth=1.5, alpha=0.8, zorder=5)
        ax1.plot(traj_arr[0, 0], traj_arr[0, 1], color=color_start, marker='o', zorder=6)
    legend_elements_1 = [
        Patch(facecolor=color_obs, edgecolor=color_obs, label='Obstacles'),
        Patch(facecolor=color_cleaned, edgecolor=color_cleaned, label='Cleaned region'),
        Patch(facecolor=color_robot, edgecolor=color_robot, label='Robot'),
        Patch(facecolor=color_uncoverable, edgecolor=color_uncoverable, label='Uncoverable region'),
        Line2D([0], [0], color=color_traj, lw=1.5, label='Trajectory'),
        Line2D([0], [0], marker='o', color=color_start, label='Start_pos', markerfacecolor=color_start, linestyle='None'),
    ]
    ax1.legend(handles=legend_elements_1, loc='upper left', bbox_to_anchor=(1.05, 1))
    ax1.set_title(f"Coverage: {coverage*100:.2f}%, Overlap rate: {overlap_percent:.2f}%")
    ax1.text(0, -0.1, f"Path length: {len(traj_arr)}\nCleaning time: {cleaning_time:.2f} min", transform=ax1.transAxes, ha="left", va="top", fontsize=11, color='black')
    
    # debug_text = (f"[Debugging text]\n"
    # f"- Steps: {steps}\n"
    # f"- Covered grid num: {coveraged_area}\n"
    # f"- Real covered grid num: {np.sum(cleaned > 0)}\n"
    # f"- Negative grid num: {np.sum(cleaned < 0)}")
    # ax1.text(0, -0.5, debug_text, transform=ax1.transAxes, ha="left", va="top", fontsize=11, color='black')
    
    # --------------------------------------------------------------------
    
    # --------------------------------------------------------------------
    # [Cleaned map 시각화]
    # --------------------------------------------------------------------
    
    # Color 지정: RGB값
    obs_c = [0, 0, 0]; unc_c = [1.0, 0.65, 0.65]
    
    cleaned_data = map_layers.cleaned.astype(np.float32).copy()
    cleaned_data[cleaned_data == 0] = np.nan
    
    img2 = ax2.imshow(cleaned_data, cmap='viridis', origin='lower', vmin=0, vmax=cleaned_map_max) # trace을 시각화
    cbar = fig.colorbar(img2, ax=ax2, 
                                orientation='horizontal',
                                pad=0.1,
                                shrink=0.8,
                                aspect=30,
                                fraction=0.05) # colorbar 추가
    cbar.set_label('Cover Count (Penalty Intensity)', fontsize=10)
    
    # Obstacle을 표시
    obs_map = np.zeros((*map_layers.cleaned.shape, 4)) # RGBA 형식
    obs_map[map_layers.obstacles == 1] = [*obs_c, 1]
    ax2.imshow(obs_map, origin='lower')
    
    # Uncoverable 영역을 표시
    unc_map = np.zeros((*map_layers.cleaned.shape, 4)) # RGBA 형식
    unc_map[map_layers.coverable == 0] = [*unc_c, 1]
    ax2.imshow(unc_map, origin='lower')
    
    # 시작 위치와 현재 위치 표시
    if len(traj_arr) > 1:
        ax2.plot(traj_arr[0, 0], traj_arr[0, 1], color=color_start, marker='o', zorder=3)
        ax2.plot(traj_arr[-1, 0], traj_arr[-1, 1], color=color_end, marker='o', zorder=4)

    legend_elements_2 = [
        Patch(facecolor=obs_c, edgecolor=obs_c, label='Obstacles'),
        Patch(facecolor=unc_c, edgecolor=unc_c, label='Uncoverable region'),
        Line2D([0], [0], marker='o', color=color_start, label='Start_pos', markerfacecolor=color_start, linestyle='None'),
        Line2D([0], [0], marker='o', color=color_end, label='Final_pos', markerfacecolor=color_end, linestyle='None'),
    ]
    ax2.legend(handles=legend_elements_2, loc='upper left', bbox_to_anchor=(1.05, 1))
    ax2.set_title(f"Cleaned map, Overlap rate: {overlap_percent:.2f}%")
    # --------------------------------------------------------------------
    
    fig.tight_layout()
    
    
def draw_obs(obs: dict, fig: Figure, stack_steps: int, local_view_dim: int, cleaned_map_max: int, 
             ray_data_indices: tuple[int, int], coverage_idx: int, preprocessor=None):
        
    fig.clear() # 이전 그림 지우기
    fig.set_size_inches(15, 12)
    
    # 그림을 그릴 창을 4행 3열로 나눔. 첫 번째 행의 높이는 두 번째 행보다 2배 높게 설정.
    height_ratios = [4]*stack_steps + [1]
    gs = GridSpec(stack_steps+1, 3, figure=fig, height_ratios=height_ratios)
    
    # obs data를 얻음
    if preprocessor is not None:
        obs = preprocessor(obs, local_view_dim)
    H, W = obs['map'][0].shape
    
    # ---- obs의 map data 그리기 ----
    
    for i in range(stack_steps):
        # gs의 첫 번째 행을 그래프를 그리는 데 사용
        ax1 = fig.add_subplot(gs[i, 0])
        ax2 = fig.add_subplot(gs[i, 1])
        ax3 = fig.add_subplot(gs[i, 2])
        axes = [ax1, ax2, ax3]
            
        # collision_map
        img0 = axes[0].imshow(obs['map'][3*i], cmap='gray_r', origin='lower')
        axes[0].plot(H//2, W//2, 'r.')
        axes[0].set_title(f"Collision_map local view(Last {stack_steps-i-1} steps ago)")
        legend_elements0 = [
            Patch(facecolor='black', edgecolor='black', label='Obstacles'),
            Patch(facecolor='white', edgecolor='black', label='Free space'),
        ]
        axes[0].legend(handles=legend_elements0, loc='upper left', bbox_to_anchor=(0, -0.1))
        # 그래프 위치를 맞추기 위한 가상의 colorbar
        cbar_fake0 = fig.colorbar(img0, ax=axes[0], orientation='horizontal', 
                                    pad=0.1, shrink=0.8, aspect=30, fraction=0.05)
        cbar_fake0.ax.set_visible(False)

        # cleaned
        # 현재 로봇 위치를 점으로 찍음
        cleaned_map = (obs['map'][3*i+1]*cleaned_map_max).astype(np.float32).copy()
        cleaned_map[cleaned_map == 0] = np.nan
        img1 = axes[1].imshow(cleaned_map, cmap='viridis', origin='lower', vmin=0, vmax=cleaned_map_max)
        axes[1].plot(H//2, W//2, 'r.') # 로봇 위치를 빨간 점으로 표시
        axes[1].set_title(f"Cleaned map local view(Last {stack_steps-i-1} steps ago)")
        cbar = fig.colorbar(img1, ax=axes[1], 
                                orientation='horizontal',
                                pad=0.1,
                                shrink=0.8,
                                aspect=30,
                                fraction=0.05) # colorbar 추가
        cbar.set_label('Cover Count (Penalty Intensity)', fontsize=10)
        # 그래프 위치를 맞추기 위한 가상의 colorbar
        cbar_fake1 = fig.colorbar(img1, ax=axes[1], orientation='horizontal', 
                                    pad=0.1, shrink=0.8, aspect=30, fraction=0.05)
        cbar_fake1.ax.set_visible(False)
        
        # # trace
        # # 현재 로봇 위치를 점으로 찍음
        # img2 = axes[2].imshow(obs['map'][3*i+2]*TRACE_MAP_MAX, cmap='gray_r', origin='lower', vmin=0, vmax=TRACE_MAP_MAX) # trace을 시각화
        # cbar = fig.colorbar(img2, ax=axes[2], 
        #                         orientation='horizontal',
        #                         pad=0.1,
        #                         shrink=0.8,
        #                         aspect=30,
        #                         fraction=0.05) # colorbar 추가
        # cbar.set_label('Trace value', fontsize=10)
        # axes[2].plot(H//2, W//2, 'r.') # 로봇 위치를 빨간 점으로 표시
        # axes[2].set_title(f"Trace layer local view(Last {cfg.stack_steps-i-1} steps ago)")
        
        # collision_map
        img2 = axes[2].imshow(obs['map'][3*i+2], cmap='gray_r', origin='lower')
        axes[2].plot(H//2, W//2, 'r.')
        axes[2].set_title(f"Uncleaned map local view(Last {stack_steps-i-1} steps ago)")
        legend_elements0 = [
            Patch(facecolor='black', edgecolor='black', label='Obstacles'),
            Patch(facecolor='white', edgecolor='black', label='Free space'),
        ]
        axes[2].legend(handles=legend_elements0, loc='upper left', bbox_to_anchor=(0, -0.1))
        # 그래프 위치를 맞추기 위한 가상의 colorbar
        cbar_fake0 = fig.colorbar(img2, ax=axes[2], orientation='horizontal', 
                                    pad=0.1, shrink=0.8, aspect=30, fraction=0.05)
        cbar_fake0.ax.set_visible(False)
    # -----------------------------
    
    # ---- obs의 vec data를 text 형식으로 출력하기 ----
    
    # gs의 두 번째 행을 text 출력에 사용
    ax_txt = fig.add_subplot(gs[-1, :])
    ax_txt.axis('off')
    
    # text 출력
    ray_data = obs['vec'][ray_data_indices[0]:ray_data_indices[1]]
    ray_str = np.array2string(ray_data, separator=', ', formatter={'float_kind': lambda x: f"{x:.2f}"})
    
    vec_info_text = (f"[Observation Status]\n"
    f"- Normlized position: ({obs['vec'][0]:.2f}, {obs['vec'][1]:.2f})\n"
    f"- Normalized ray: {ray_str}\n"
    f"- Normalized coverage: {obs['vec'][coverage_idx]:.2f}\n")
    ax_txt.text(0, 0.5, vec_info_text, transform=ax_txt.transAxes, fontsize=14,
                va='top', ha='left', family='monospace')
    # ---------------------------------------------
    
    fig.tight_layout()