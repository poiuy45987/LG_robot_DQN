import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from config import EnvConfig, DEFAULT_SEED

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


def add_boundary_walls(obstacles: np.ndarray, thickness: int=1, value=WALL):
    H, W = obstacles.shape
    obstacles[0:thickness, :] = value
    obstacles[H-thickness:H, :] = value    
    obstacles[:, 0:thickness] = value          
    obstacles[:, W-thickness:W] = value        


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
        if table_leg_size <= 5:
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
        if chair_leg_size < 5:
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
    chair_size_offset: int = 2,
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
        
        chair_size_local = chair_size_offset
        while chair_size_local > chair_leg_size:
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

def add_wall_furniture(
    obstacles: np.ndarray,
    side: str,
    thickness: int,
    span_start: int,
    span_len: int,
):
    # large wall-attached obstacle (sofa/bed/cabinet-like)
    H, W = obstacles.shape
    if side in ("left", "right"):
        y0 = max(0, span_start)
        y1 = min(H, span_start + span_len)
        if side == "left":
            obstacles[y0:y1, :thickness] = 1
        else:
            obstacles[y0:y1, W - thickness:] = 1
    else:
        x0 = max(0, span_start)
        x1 = min(W, span_start + span_len)
        if side == "top":
            obstacles[:thickness, x0:x1] = 1
        else:
            obstacles[H - thickness:, x0:x1] = 1

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
    

def generate_house_like_obstacles(cfg: EnvConfig, rng: np.random.Generator, visualize: bool = False) -> np.ndarray:
    
    H = cfg.H; W = cfg.W
    
    max_table_size = cfg.max_table_size
    min_table_size = cfg.min_table_size
    table_leg_size = cfg.table_leg_size
    
    chair_size = cfg.chair_size
    chair_leg_size = cfg.chair_leg_size
    chair_spread = cfg.chair_spread
    
    obs = np.zeros((H, W), dtype=np.uint8)
    
    # Map의 벽면을 장애물로 설정: 로봇이 grid map 밖을 나가지 못하도록 하기 위한 용도
    add_boundary_walls(obs, cfg.boundary_thickness, value=WALL)
    
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
                math.sqrt((table_width/2.0 + chair_size/2.0 + chair_spread)**2 + (chair_size/2.0)**2),
                math.sqrt((table_height/2.0 + chair_size/2.0 + chair_spread)**2 + (chair_size/2.0)**2),
            ))) + 2
            
            # 테이블의 중심 좌표 설정: (cx, cy)
            cx = int(rng.integers(cfg.boundary_thickness+table_radius, W-cfg.boundary_thickness-table_radius))
            cy = int(rng.integers(cfg.boundary_thickness+table_radius, H-cfg.boundary_thickness-table_radius))
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
                    chair_size_offset=chair_size,
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
    
    for x in range(cfg.boundary_thickness, W-cfg.boundary_thickness, gap):
        for y in range(cfg.boundary_thickness, H-cfg.boundary_thickness, gap):
            
            obs_num = rng.integers(cfg.small_obs_num_per_window_min, cfg.small_obs_num_per_window_max+1)
            x_last = min(W-cfg.boundary_thickness, x+window_size)
            y_last = min(H-cfg.boundary_thickness, y+window_size)
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

# 시각화를 위한 메인 코드
def visualize_house_map(seed=DEFAULT_SEED):
    
    cfg = EnvConfig()
    rng = np.random.default_rng(seed=seed)  # 재현성을 위해 시드 설정

    obstacles = generate_house_like_obstacles(cfg, rng, visualize=True)
    
    custom_cmap = ListedColormap(['white', 'black', 'red', 'blue', 'purple'])

    plt.figure(figsize=(6, 6))
    im = plt.imshow(obstacles, cmap=custom_cmap, origin='lower', vmin=0, vmax=4)
    
    # Legend 정의
    legend_elements = [
        Patch(facecolor='black', edgecolor='black', label='Wall'),
        Patch(facecolor='red', edgecolor='red', label='Table Leg'),
        Patch(facecolor='blue', edgecolor='blue', label='Chair Leg'),
        Patch(facecolor='purple', edgecolor='purple', label='More obstacle')
    ]
    plt.legend(
        handles=legend_elements, 
        loc='upper left', 
        bbox_to_anchor=(1.05, 1), # 그래프 오른쪽 살짝 바깥에 배치
        title="Obstacle Types",
        title_fontsize='12',
        fontsize='10'
    )
    
    plt.title(f"House-like Obstacles ({cfg.H}x{cfg.W})", fontsize=15)
    plt.xlabel("Width")
    plt.ylabel("Height")
    
    # 격자 표시 (선택 사항)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.show()

# 실행
if __name__ == "__main__":
    visualize_house_map(seed=DEFAULT_SEED)