import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
import zipfile

from .config import EnvConfig, DEFAULT_SEED, MAP_SAVE_DIR
from .utils import display_image

# Map을 시각화할 때 table, chair을 구별하기 위해 설정한 값
# 실제로 훈련 또는 validation 과정에서 map을 생성할 때는 장애물을 전부 1로 바꿈
BLANK = 0
WALL = 1
TABLE = 2
CHAIR = 3
MORE_OBS = 4

# 직사각형을 좀 더 정확하게 그리는 방법
# Grid 크기가 충분히 작을 때 유용할 것이라고 생각
def get_rect_indices_for_big_rect(
    H: int, W: int, 
    cx: float, cy: float, 
    width: float, height: float, 
    theta: float) -> list[tuple[int, int]]:
    """
    Grid map에서 직사각형을 그릴 때, 색칠할 grid의 indices를 출력하는 method

    Args:
        H, W: 장애물이 배치되는 map의 height와 width
        cx, cy: 그릴 직사각형의 중심 좌표
        width, height: 그릴 직사각형의 가로, 세로 길이
        theta: 직사각형이 반시계 방향으로 회전한 정도(rad 단위)
    """
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    grid_indices = []
    
    # 각 grid가 직사각형 영역에 속하는지 검사
    # 검사 범위는 직사각형의 대각선 길이를 한 변의 길이롤 하는 정사각형
    r = math.sqrt(width**2 + height**2) / 2
    x_min = max(0, int(math.floor(cx - r)))
    x_max = min(W - 1, int(math.ceil(cx + r)))
    y_min = max(0, int(math.floor(cy - r)))
    y_max = min(H - 1, int(math.ceil(cy + r)))

    # 각 그리드 칸의 중심점이 직사각형 내부에 있는지 검사
    for ix in range(x_min, x_max + 1):
        for iy in range(y_min, y_max + 1):
            # 그리드 칸의 중심점: (ix.5, iy.5)
            # (dx, dy): 직사각형 중심점 (cx, cy)를 원점으로 봤을 때 grid 중심점의 좌표
            dx = (ix + 0.5) - cx
            dy = (iy + 0.5) - cy
            
            # 역회전 변환 (Rotated frame -> Local axis-aligned frame)
            nx = dx * cos_t + dy * sin_t
            ny = -dx * sin_t + dy * cos_t
            
            # 범위 판정 (부동소수점 오차 방지를 위해 아주 작은 값 1e-9 추가)
            if abs(nx) <= (width / 2) + 1e-9 and abs(ny) <= (height / 2) + 1e-9:
                grid_indices.append((ix, iy))
    
    return grid_indices

def get_rect_indices_for_small_rect(
    H: int, W: int, 
    cx: float, cy: float, 
    width: float, height: float, 
    theta: float) -> list[tuple[int, int]]:
    """
    Grid map에서 직사각형을 그릴 때, 색칠할 grid의 indices를 출력하는 method

    Args:
        H, W: 장애물이 배치되는 map의 height와 width
        cx, cy: 그릴 직사각형의 중심 좌표
        width, height: 그릴 직사각형의 가로, 세로 길이
        theta: 직사각형이 반시계 방향으로 회전한 정도(rad 단위)
    """
    
    # 직사각형을 그릴 grid의 시작과 끝 좌표를 int 형식으로 얻음
    x_min = max(0, int(cx - width/2 + 0.5))
    x_max = min(W, int(cx + width/2 + 0.5))
    y_min = max(0, int(cy - height/2 + 0.5))
    y_max = min(H, int(cy + height/2 + 0.5))
    
    grid_indices = [(x, y) for x in range(x_min, x_max) for y in range(y_min, y_max)]
    
    return grid_indices

def add_table_legs(
    obstacles: np.ndarray,
    cx: float,
    cy: float,
    table_width: int,
    table_height: int,
    table_leg_size: int,
    theta: float = 0.0,   # 회전 각도 (rad)
) -> bool:
    """_summary_

    Args:
        obstacles (np.ndarray): Obstacle이 배치된 grid map 정보
        cx, cy (int): Table의 중심 좌표
        table_row_size (int): Table의 가로 길이
        table_col_size (int): Table의 세로 길이
        table_leg_size (int, optional): Table 다리의 두께. Defaults to 2.
        theta (float, optional): Table이 반시계 방향으로 회전한 각도. Defaults to 0.0.

    Returns:
        bool: Table 배치가 성공했는지 여부
    """
    # cx, cy: 테이블 중심 좌표, row_size 또는 col_size가 짝수인 경우 중심 위치로부터 약간 틀어져 있음
    # 테이블 기본 다리 좌표 (local frame)
    
    # 테이블의 각도가 0 rad일 때 각 leg 중심의 상대적인 좌표
    no_rotate_leg_centers = [
        (-table_width/2.0 + table_leg_size/2.0, -table_height/2.0 + table_leg_size/2.0),
        ( table_width/2.0 - table_leg_size/2.0, -table_height/2.0 + table_leg_size/2.0),
        (-table_width/2.0 + table_leg_size/2.0,  table_height/2.0 - table_leg_size/2.0),
        ( table_width/2.0 - table_leg_size/2.0,  table_height/2.0 - table_leg_size/2.0),
    ]
    
    leg_grid_indices = []
    
    c = math.cos(theta)
    s = math.sin(theta)

    for px, py in no_rotate_leg_centers:
        # (px, py): Table 중심을 원점으로 했을 때, table leg의 중심 위치
        
        # Table이 돌아간 각도 theta에 따라 (px, py)를 회전 s변환
        rx = c * px - s * py
        ry = s * px + c * py

        # (x, y): Table leg 중심 위치의 global coordinate
        x = cx + rx; y = cy + ry
        
        # Table 다리가 차지하는 grid의 indices를 얻음
        H, W = obstacles.shape
        if table_leg_size < 10:
            indices = get_rect_indices_for_small_rect(H, W, x, y, table_leg_size, table_leg_size, theta)
        else:
            indices = get_rect_indices_for_big_rect(H, W, x, y, table_leg_size, table_leg_size, theta)
        leg_grid_indices.extend(indices)
    
    if not leg_grid_indices:
        return False
    
    leg_grid_indices_np = np.array(leg_grid_indices)
    
    # 장애물과 table 다리 위치가 겹치면 table을 배치하지 않음    
    if not np.all(obstacles[leg_grid_indices_np[:, 1], leg_grid_indices_np[:, 0]] == BLANK):
        return False
    
    # 장애물이 겹치지 않으면 table을 배치
    obstacles[leg_grid_indices_np[:, 1], leg_grid_indices_np[:, 0]] = TABLE
    return True

def add_chair_legs(
    obstacles: np.ndarray,
    cx: int,
    cy: int,
    chair_width: int,
    chair_height: int,
    chair_leg_size: int = 1,
    theta: float = 0.0,   # 회전 각도 (rad)
) -> bool:
    """_summary_

    Args:
        obstacles (np.ndarray): Obstacle이 배치된 grid map 정보
        cx, cy (int): Table의 중심 좌표
        table_row_size (int): Table의 가로 길이
        table_col_size (int): Table의 세로 길이
        table_leg_size (int, optional): Table 다리의 두께. Defaults to 2.
        theta (float, optional): Table이 반시계 방향으로 회전한 각도. Defaults to 0.0.

    Returns:
        bool: Table 배치가 성공했는지 여부
    """
    # cx, cy: 테이블 중심 좌표, row_size 또는 col_size가 짝수인 경우 중심 위치로부터 약간 틀어져 있음
    # 테이블 기본 다리 좌표 (local frame)
    
    # 테이블의 각도가 0 rad일 때 각 leg 중심의 상대적인 좌표
    no_rotate_leg_centers = [
        (-chair_width/2.0 + chair_leg_size/2.0, -chair_height/2.0 + chair_leg_size/2.0),
        ( chair_width/2.0 - chair_leg_size/2.0, -chair_height/2.0 + chair_leg_size/2.0),
        (-chair_width/2.0 + chair_leg_size/2.0,  chair_height/2.0 - chair_leg_size/2.0),
        ( chair_width/2.0 - chair_leg_size/2.0,  chair_height/2.0 - chair_leg_size/2.0),
    ]
    
    leg_grid_indices = []
    
    c = math.cos(theta)
    s = math.sin(theta)

    for px, py in no_rotate_leg_centers:
        # (px, py): Table 중심을 원점으로 했을 때, table leg의 중심 위치
        
        # Table이 돌아간 각도 theta에 따라 (px, py)를 회전 변환
        rx = c * px - s * py
        ry = s * px + c * py

        # (x, y): Table leg 중심 위치의 global coordinate
        x = cx + rx; y = cy + ry
        
        # Table 다리가 차지하는 grid의 indices를 얻음
        H, W = obstacles.shape
        if chair_leg_size < 10:
            indices = get_rect_indices_for_small_rect(H, W, x, y, chair_leg_size, chair_leg_size, theta)
        else:
            indices = get_rect_indices_for_big_rect(H, W, x, y, chair_leg_size, chair_leg_size, theta)
        leg_grid_indices.extend(indices)
    
    if not leg_grid_indices:
        return False
    
    leg_grid_indices_np = np.array(leg_grid_indices)
    
    # 장애물과 table 다리 위치가 겹치면 table을 배치하지 않음    
    if not np.all(obstacles[leg_grid_indices_np[:, 1], leg_grid_indices_np[:, 0]] == BLANK):
        return False
    
    # 장애물이 겹치지 않으면 table을 배치
    obstacles[leg_grid_indices_np[:, 1], leg_grid_indices_np[:, 0]] = CHAIR
    return True

def add_chairs_around_table(
    obstacles: np.ndarray,
    cx: int,
    cy: int,
    table_width: int,
    table_height: int,
    num_chairs: int,
    rng: np.random.Generator,
    theta: float,
    chair_dist_offset: int = 4,
    #chair_size_offset: int = 2,
    max_chair_size: int = 24,
    min_chair_size: int = 20,
    chair_leg_size: int = 1,
):
    
    radius_angle_set = [
        (0, table_width/2.0 + chair_dist_offset),
        (math.pi/2, table_height/2.0 + chair_dist_offset),
        (math.pi, table_width/2.0 + chair_dist_offset),
        (3*math.pi/2, table_height/2.0 + chair_dist_offset),
    ]
    rng.shuffle(radius_angle_set)
    radius_angle_set = radius_angle_set[:num_chairs]

    for (a, r) in radius_angle_set:
        
        chair_size_local = max_chair_size
        while chair_size_local >= min_chair_size:
            # 각도/거리 노이즈
            ang = theta + a + rng.normal(scale=0.1)
            radius = r + rng.integers(-1, 2)

            # 의자의 중심 좌표
            x = int(round(cx + radius * math.cos(ang)))
            y = int(round(cy + radius * math.sin(ang)))
            
            # 의자가 배치 가능한 경우 의자 다리를 생성
            if add_chair_legs(
                obstacles, 
                x, y, 
                chair_width=chair_size_local, 
                chair_height=chair_size_local, 
                chair_leg_size=chair_leg_size, 
                theta=ang):
                break
            # 의자 다리가 장애물과 겹치거나 맵을 넘는 경우 의자 크기를 줄여가며 재시도
            else:
                chair_size_local -= 1

def _is_placeable(
    obs: np.ndarray,
    x_min: int, x_max: int,
    y_min: int, y_max: int,
) -> bool:
    
    H, W = obs.shape
    x_min = max(0, x_min-1); x_max = min(W, x_max+1)
    y_min = max(0, y_min-1); y_max = min(H, y_max+1)
    if np.all(obs[x_min:x_max, y_min:y_max] == BLANK):
        return True
    return False
    
def add_small_obstacles_in_window(
    obs: np.ndarray, 
    rng: np.random.Generator,
    x_min: int, x_max: int,
    y_min: int, y_max: int,
    obs_num: int, 
    small_obs_size_max: int,
    small_obs_size_min: int
):
    # Window에 장애물이 하나도 없는 경우에만 장애물을 배치
    if np.all(obs[x_min:x_max, y_min:y_max] == BLANK):
        for _ in range(obs_num):
            obs_width = rng.integers(small_obs_size_min, small_obs_size_max)
            obs_height = rng.integers(small_obs_size_min, small_obs_size_max)
            x = rng.integers(x_min, x_max-obs_width)
            y = rng.integers(y_min, y_max-obs_height)
            if _is_placeable(
                obs, 
                x_min=x, x_max=x+obs_width,
                y_min=y, y_max=y+obs_height,
            ):
                obs[x:x+obs_width, y:y+obs_height] = MORE_OBS

def set_map_boundary(obs: np.ndarray, eff_size: tuple[float, float], map_info: dict) -> tuple[int, int]:
    """
    더 크기가 작은 map을 생성하기 위해 벽을 세우는 method

    Args:
        obs (np.ndarray): 장애물을 배치할 map 정보가 담긴 numpy 배열
        eff_size (tuple[float, float]): Map에서 사용할 영역의 (가로 길이, 세로 길이)를 cm 단위로 받음.
        map_info (dict): Map의 정보(전체 map의 크기, grid 크기 등)

    Returns:
        tuple[int, int]: Map에서 사용할 영역의 (가로 길이, 세로 길이)를 grid 단위로 반환
    """
    
    H, W = obs.shape
    
    # 유효성 검증
    is_valid = True
    if eff_size[0] > map_info['map_width']:
        print(f"Warning: Active area width ({eff_size[0]:.2f} cm) is larger than the map width ({map_info['map_width']:.2f} cm)")
        is_valid = False
    if eff_size[1] > map_info['map_height']:
        print(f"Warning: Active area height ({eff_size[1]:.2f} cm) is larger than the map height ({map_info['map_height']:.2f} cm)")
        is_valid = False
        
    if is_valid:
        eff_W = min(int(eff_size[0] // map_info['grid_size']), W)
        eff_H = min(int(eff_size[1] // map_info['grid_size']), H)
        
        obs[:, :] = WALL
        obs[:eff_H, :eff_W] = BLANK
        
        return eff_W, eff_H
    
    return None, None

def generate_house_like_obstacles(cfg: EnvConfig, rng: np.random.Generator, 
                                  visualize: bool = False, eff_size: tuple[float, float] = None) -> np.ndarray:
    """
    책상, 의자, 작은 장애물 등 집과 비슷한 형태의 map을 형성하는 method

    Args:
        cfg (EnvConfig): Map 환경의 정보
        rng (np.random.Generator): Random generator
        visualize (bool, optional): Map 시각화 시 책상, 의자, 작은 장애물을 구별해서 보여줄지 여부. Defaults to False.
        eff_size (tuple[float, float], optional): Map의 공간을 제한적으로 사용하고 싶을 때, 사용할 영역의 (가로 길이, 세로 길이)를 cm 단위로 입력. Defaults to None.

    Returns:
        np.ndarray: Obstacle이 배치된 grid map의 numpy 배열.
    """
    H = cfg.H; W = cfg.W
    
    max_table_size = cfg.max_table_size
    min_table_size = cfg.min_table_size
    table_leg_size = cfg.table_leg_size
    
    max_chair_size = cfg.max_chair_size
    min_chair_size = cfg.min_chair_size
    chair_leg_size = cfg.chair_leg_size
    chair_spread = cfg.chair_spread
    
    obs = np.zeros((H, W), dtype=np.uint8)
    if eff_size is not None:
        map_info = {
            "map_width": cfg.map_width,
            "map_height": cfg.map_height,
            "grid_size": cfg.grid_size,
        }
        eff_W, eff_H = set_map_boundary(obs, eff_size, map_info)
        if eff_W is not None and eff_H is not None:
            W, H = eff_W, eff_H
    
    # 책상과 의자를 배치
    table_centers = [] # 테이블 중심 좌표와 테이블 radius를 저장
    max_trial = 20000
    for table_num in range(cfg.num_tables):
        for _ in range(max_trial):  # 최대 20000번 시도
            
            # 테이블의 크기 설정
            table_width = rng.integers(min_table_size, max_table_size + 1)
            table_height = rng.integers(min_table_size, max_table_size + 1)
            
            table_radius = int(math.ceil(max(
                math.sqrt((table_width/2.0)**2 + (table_height/2.0)**2),
                math.sqrt((table_width/2.0 + max_chair_size/2.0 + chair_spread)**2 + (max_chair_size/2.0)**2),
                math.sqrt((table_height/2.0 + max_chair_size/2.0 + chair_spread)**2 + (max_chair_size/2.0)**2),
            ))) + 2
            
            # 테이블의 중심 좌표 설정: (cx, cy)
            if table_width >= W - table_radius:
                continue
            if table_height >= H - table_radius:
                continue
            cx = int(rng.integers(table_radius, W-table_radius))
            cy = int(rng.integers(table_radius, H-table_radius))
            cx += (table_width % 2) / 2.0
            cy += (table_height % 2) / 2.0
            
            ok = True # 테이블과 의자가 겹치지 않고 배치될 수 있는지 알려주는 flag
            
            # 테이블이 겹치는지 여부를 테이블 중심 좌표로 대략적으로 확인
            for (px, py, pr) in table_centers:
                if (cx - px)**2 + (cy - py)**2 < (table_radius + pr)**2:
                    ok = False
                    break # 겹치는 테이블이 있는 경우, cx, cy, table_width, table_height를 재생성
                    
            # 테이블 중심 좌표로 판단했을 때 테이블이 다른 장애물과 겹치지 않으면 테이블 배치를 시도
            if ok: 
                
                theta = rng.uniform(-math.pi/6, math.pi/6) # 테이블이 회전한 각도 결정
                
                # 테이블을 map에 추가. 장애물이 겹치지 않은 경우 의자 배치 시도
                if add_table_legs(
                    obs,
                    cx, cy, 
                    table_width, table_height,
                    table_leg_size=table_leg_size,
                    theta=theta
                ):
                    table_centers.append((cx, cy, table_radius)) # 테이블의 중심 좌표를 list에 추가
                else:
                    ok = False
                    continue # cx, cy, table_width, table_height를 재생성
            
            # 테이블을 실제로 배치했을 때, 다른 장애물과 겹치지 않으면 의자 배치를 시도
            if ok:

                # 의자 개수 결정
                num_chairs = int(rng.integers(cfg.chairs_per_table_min, cfg.chairs_per_table_max + 1))
                
                # 의자  
                add_chairs_around_table(
                    obs,
                    cx, cy,
                    table_width, table_height,
                    num_chairs=num_chairs,
                    rng=rng,
                    theta=theta,
                    chair_dist_offset=chair_spread,
                    max_chair_size=max_chair_size,
                    min_chair_size=min_chair_size,
                    chair_leg_size=chair_leg_size,
                )
                break # 의자 배치까지 완료했으면 책상 배치를 그만 시도
        
        else:
            print(f"Table {table_num+1}: Failed to place a table after max trials.")
                    
    # 고밀도 환경 조성을 위해 작은 장애물을 더 배치
    window_size = cfg.window_size
    gap = int(window_size // 3)
    small_obs_size_max = cfg.small_obs_size_max
    small_obs_size_min = cfg.small_obs_size_min
    
    for x in range(0, W, gap):
        for y in range(0, H, gap):
            
            obs_num = rng.integers(cfg.small_obs_num_per_window_min, cfg.small_obs_num_per_window_max+1)
            x_last = min(W, x+window_size)
            y_last = min(H, y+window_size)
            obs_num = int(obs_num * (x_last-x)*(y_last-y) / (window_size**2))
            add_small_obstacles_in_window(
                obs,
                rng=rng,
                x_min=x, x_max=x_last,
                y_min=y, y_max=y_last,
                obs_num=obs_num,
                small_obs_size_max=small_obs_size_max,
                small_obs_size_min=small_obs_size_min
            )

    if not visualize:
        obs = (obs != 0).astype(np.uint8)

    return obs

def get_test_maps(map_num: int, start_seed: int, visualize: bool = False, map_size_list: list[tuple[float, float]] = None):
    
    # Map을 저장할 폴더 생성
    map_folder_name = MAP_SAVE_DIR
    
    if not os.path.exists(map_folder_name):
        os.makedirs(map_folder_name)
    
    # 시각화를 하는 경우, fig와 canvas 생성
    fig = None; canvas = None
    if visualize:
        fig = Figure()
        canvas = FigureCanvasAgg(fig)
    
    # Map 생성 및 저장
    cfg = EnvConfig()
    for map_id, seed in enumerate(range(start_seed, start_seed+map_num)):
        
        map_file_name = f"test_map_lch_{seed:02d}.npy"
        
        # Random generator 생성
        rng = np.random.default_rng(seed=seed)
        
        # Map에서 실제 청소기가 움직이고 장애물이 배치될 영역 정의. None이면 map 전체 영역 사용
        eff_size = map_size_list[map_id] if map_size_list and map_id < len(map_size_list) else None
        
        # Map에 장애물 배치
        obstacles = generate_house_like_obstacles(cfg, rng, visualize=visualize, eff_size=eff_size)
        
        # Map을 시각화
        if visualize:
            map_img = get_map_img(obstacles, fig=fig, canvas=canvas, map_name=map_file_name, visualized=visualize)
            display_image(map_img)
            obstacles = (obstacles != 0).astype(np.uint8)
        
        np.save(os.path.join(map_folder_name, map_file_name), obstacles)

def get_map_img(obstacles: np.ndarray, fig: Figure, canvas: FigureCanvasAgg, 
                map_name: str, visualized: bool = True) -> np.ndarray:
    
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

def generate_map_by_seed_and_visualize(seed: int = DEFAULT_SEED, eff_size: tuple[float, float] = None):
    
    cfg = EnvConfig()
    rng = np.random.default_rng(seed=seed)  # 재현성을 위해 시드 설정

    # Map 생성
    obstacles = generate_house_like_obstacles(cfg, rng, visualize=True, eff_size=eff_size)
    H, W = obstacles.shape
    
    # Map 시각화
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    map_img = get_map_img(obstacles, fig=fig, canvas=canvas, map_name=f"Map size: {H}x{W} / Map seed: {seed}", visualized=True)
    display_image(map_img)
    
def visualize_saved_map(map_file_name: str):
    
    map_folder_name = MAP_SAVE_DIR
    seed = map_file_name.split("_")[-1].split(".")[0]
    obstacles = np.load(f"{map_folder_name}/{map_file_name}")
    
    # Map 시각화
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    map_img = get_map_img(obstacles, fig=fig, canvas=canvas, map_name=map_file_name, visualized=False)
    display_image(map_img)

def zip_map_files(zip_file_name: str = 'maps.zip'):
    
    map_folder_name = MAP_SAVE_DIR
    if not os.path.exists(map_folder_name):
        print(f"Error: {map_folder_name} 폴더가 존재하지 않습니다.")
        return
    
    zip_file_name = os.path.join(map_folder_name, zip_file_name)
    with zipfile.ZipFile(zip_file_name, 'w') as zip_file:
        for file_name in os.listdir(map_folder_name):
            if file_name.endswith('.npy'):
                zip_file.write(os.path.join(map_folder_name, file_name), arcname=file_name)
    
    print(f"압축 완료: {zip_file_name}")

# 실행
if __name__ == "__main__":
    generate_map_by_seed_and_visualize(seed=DEFAULT_SEED, eff_size=None)