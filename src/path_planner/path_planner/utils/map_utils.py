import math
import numpy as np
import cv2
from collections import deque
from dataclasses import dataclass


def float_to_int_coord_1D(x):
    """
    실수 좌표 x(단일 숫자 또는 여러 숫자)를 정수 좌표로 변환하는 method.
    X.5 ~ (X+1).5를 X로 변환.
    """
    
    # 1. 단일 숫자(float, int)인 경우 -> 순수 파이썬으로 초고속 처리
    if isinstance(x, (int, float, np.floating, np.integer)):
        return int(math.floor(x + 0.5))

    # 2. 리스트나 배열인 경우 -> 넘파이의 벡터화 연산 활용
    x_np = np.asanyarray(x)
    return np.floor(x_np + 0.5).astype(int)


def float_to_int_coord(cx, cy):
    """
    실수 좌표 (x, y) (단일 좌표 또는 여러 좌표)를 정수 좌표로 변환하는 method.
    X.5 ~ (X+1).5를 X로 변환.
    """
    
    return float_to_int_coord_1D(cx), float_to_int_coord_1D(cy)


def dilation_obstacles(obs_map: np.ndarray, dilation_mask: np.ndarray):
    obs_map_with_wall = np.pad(obs_map, pad_width=1, mode='constant', constant_values=1) # 맵 바깥에 가상의 벽을 설치
    extend_collision_map = cv2.dilate(
        obs_map_with_wall, 
        dilation_mask,
        borderType=cv2.BORDER_CONSTANT, 
        borderValue=1
    )
    return extend_collision_map[1:-1, 1:-1] # dilation된 맵에서 맵 바깥에 설치한 벽 부분을 제거하여 collision_map 생성


def generate_navigation_mode_map(
    obs: np.ndarray, 
    crop_size: int, 
    robot_diameter: int, 
    stride: int, 
    eff_size: tuple[int, int] = None,
) -> np.ndarray:
    """
    로봇의 이동 경로를 막지 않는 안전한 빈 공간만 찾아 장애물로 채우는 함수.

    Args:
        obs (np.ndarray): 현재 장애물 맵 (H, W)
        crop_size (int): Crop할 정사각형 크기 (grid 단위)
        robot_diameter (int): 로봇의 지름 (grid 단위)
        stride (int): 탐색 윈도우의 이동 간격 (grid 단위)
        eff_size (tuple[int, int]): Map의 실제 크기 (세로, 가로) (grid 단위)
    """
    H, W = obs.shape
    if eff_size is not None:
        H, W = eff_size[0], eff_size[1]

    # 1. 기본적으로 모든 영역을 DQN 구역(1)으로 초기화한 '새로운 제어 맵' 생성
    # 검증된 청정 구역만 Heuristic(0)으로 깎아 나가는 방식이 안전합니다.
    mode_map = np.ones((obs.shape[0], obs.shape[1]), dtype=np.int8)
    
    y_indices = list(range(0, H - crop_size + 1, stride))
    if (H - crop_size) not in y_indices:
        y_indices.append(H - crop_size)  # 하단 끝 좌표 강제 추가

    x_indices = list(range(0, W - crop_size + 1, stride))
    if (W - crop_size) not in x_indices:
        x_indices.append(W - crop_size)  # 우측 끝 좌표 강제 추가ㅡ
        
    robot_radius = robot_diameter // 2 + 1

    # 탐색 루프 (기존 맵 내부에서 crop_size만큼 움직임)
    for y in y_indices:
        for x in x_indices:
            
            # --- [3번 & 6번 요구사항: 안전 마진 확장 및 Index Out 방지] ---
            # 원래 크기에서 로봇 지름만큼 확장한 검사 영역(Check Window) 계산
            check_y_min = max(0, y - robot_radius)
            check_y_max = min(H, y + crop_size + robot_radius)
            check_x_min = max(0, x - robot_radius)
            check_x_max = min(W, x + crop_size + robot_radius)
            
            # 확장된 영역 슬라이싱
            check_window = obs[check_y_min:check_y_max, check_x_min:check_x_max]
            
            # --- [2번 요구사항: 주변 경로를 막지 않도록 검사] ---
            # 확장 영역 내에 '장애물(BLANK가 아닌 값)'이 단 하나도 없어야 안전한 공간으로 판단
            if np.all(check_window == 0):
                
                # --- [4번 요구사항: 실제 기록은 늘리기 전 원래 정사각형 사이즈만] ---
                mode_map[y : y + crop_size, x : x + crop_size] = 0

    return mode_map


def make_robot_mask(robot_size: int) -> np.ndarray:
    # 로봇 청소기 모양을 원형으로 만듦
    # robot_size: 로봇 청소기의 지름
    
    center = robot_size / 2.0; radius = robot_size / 2.0
    grid = np.arange(robot_size) + 0.5 - center
    x, y = np.meshgrid(grid, grid, sparse=True)
    mask = (x**2 + y**2 <= radius**2).astype(np.uint8)
    
    return mask


def compute_reachable_centers(collision_map: np.ndarray, start_cx: int, start_cy: int) -> np.ndarray:

    H, W = collision_map.shape
    reachable = np.zeros((H, W), dtype=np.uint8)

    q = deque()
    start_cx = int(start_cx); start_cy = int(start_cy)
    reachable[start_cy, start_cx] = 1
    q.append((start_cx, start_cy))

    while q:
        x, y = q.popleft()
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]: # Reachable grid는 상, 하, 좌, 우 방향으로만 확인.
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if reachable[ny, nx]:
                continue
            if collision_map[ny, nx]:
                continue
            reachable[ny, nx] = 1
            q.append((nx, ny))

    return reachable


def compute_coverable_cells_from_reachable(obstacles: np.ndarray, reachable: np.ndarray, robot_mask_offsets: np.ndarray) -> np.ndarray:

    H, W = reachable.shape
    coverable = np.zeros((H, W), dtype=np.uint8)

    # OR-shift reachable by footprint offsets -> union of all swept cells
    for dx, dy in robot_mask_offsets:
        dx = int(dx); dy = int(dy)

        src_y0 = max(0, -dy)
        src_y1 = H - max(0, dy)
        src_x0 = max(0, -dx)
        src_x1 = W - max(0, dx)

        dst_y0 = max(0, dy)
        dst_y1 = H - max(0, -dy)
        dst_x0 = max(0, dx)
        dst_x1 = W - max(0, -dx)

        coverable[dst_y0:dst_y1, dst_x0:dst_x1] |= reachable[src_y0:src_y1, src_x0:src_x1]

    # never count obstacle cells
    coverable &= (obstacles == 0).astype(np.uint8)
    return coverable


def crop_raw_patch(arr: np.ndarray, cx: float, cy: float, crop_radius: int, value: int | list | tuple = 0) -> np.ndarray:
    """
    전체 map data에서 local map data를 뽑는 method: 회전 변환 등 다른 변환은 하지 않고 raw patch만 뽑는 method

    Args:
        arr (np.ndarray): 전체 map data, Shape: (C, H, W) or (H, W)
        cx, cy (int): Local map 중심의 좌표
        value (int | list | tuple, optional): 각 채널에서 local data를 뽑을 때 padding value. 채널 순서대로 지정

    Returns:
        np.ndarray: Local map data, Shape: (C, H, W) or (H, W)
    """
    # 1. Crop할 크기 설정
    cx, cy = float_to_int_coord(cx, cy)
    px, py = cx+crop_radius, cy+crop_radius  # 전체 map에서 padding 추가 이후, (cx, cy)의 좌표 변화
    
    # 2. Padding 값 설정 및 padding 수행: 2차원 데이터일 때랑 3차원 데이터일 때 구별
    if arr.ndim == 3:
        num_channels = arr.shape[0]
        if isinstance(value, (list, tuple)):
            if len(value) != num_channels:
                raise ValueError(f"Length of value ({len(value)}) must match num_channels ({num_channels})")
            pad_vals = value
        else:
            pad_vals = [value] * num_channels
        arr = arr.transpose(1, 2, 0) # (C, H, W) -> (H, W, C) 변환
    elif arr.ndim == 2:
        assert isinstance(value, int)
        pad_vals = value
    else:
        raise ValueError(f"ERROR on _crop_patch(): Not support array dimension {arr.ndim}")
    
    padded = cv2.copyMakeBorder(
        arr, crop_radius, crop_radius, crop_radius, crop_radius, 
        cv2.BORDER_CONSTANT, value=pad_vals
    )
    
    raw_patch = padded[py-crop_radius:py+crop_radius+1, px-crop_radius:px+crop_radius+1]
    
    return raw_patch

@dataclass(frozen=True)
class BoundingBox:
    x_min: int
    x_max: int
    y_min: int
    y_max: int

def get_eff_size_from_obs_map(obs: np.ndarray, BLANK: int=0) -> BoundingBox:
    """
    obs map에서 로봇 청소기가 움직일 수 있는 영역의 bounding box를 출력하는 함수

    Args:
        obs (np.ndarray): 장애물의 위치가 나타나 있는 map
        BLANK (int, optional): 장애물이 없는 grid cell의 값. Defaults to 0.

    Returns:
        BoundingBox: x_min, x_max, y_min, y_max로 구성
    """
    has_blank_in_row = np.any(obs == BLANK, axis=1)  # Shape: (H,)
    has_blank_in_col = np.any(obs == BLANK, axis=0)  # Shape: (W,)

    if not np.any(has_blank_in_row):
        return BoundingBox(0, 0, 0, 0)

    # True인 indices만 추출
    y_indices = np.flatnonzero(has_blank_in_row)
    x_indices = np.flatnonzero(has_blank_in_col)

    return BoundingBox(
        x_min=x_indices[0].item(),
        x_max=x_indices[-1].item(),
        y_min=y_indices[0].item(),
        y_max=y_indices[-1].item()
    )
    
def crop_obstacle_area(obs_map: np.ndarray, robot_diameter: int, BLANK: int=0) -> np.ndarray:
    """
    obs_map에서 장애물이 존재하는 영역만 cropping하는 함수.
    로봇 청소기가 장애물 주위를 자유롭게 움직일 수 있도록 로봇 청소기 지름만큼의 margin만 남기고 나머지 영역은 제거.

    Args:
        obs_map (np.ndarray): 장애물의 위치가 나타나 있는 map
        robot_diameter (int): 로봇 청소기 지름
        BLANK (int, optional): 장애물이 없는 grid cell의 값. Defaults to 0.

    Returns:
        np.ndarray: 장애물 영역만 남김 cropped map
    """
    bbox = get_eff_size_from_obs_map(obs_map, BLANK)
    eff_obs_map = obs_map[bbox.y_min:bbox.y_max+1, bbox.x_min:bbox.x_max+1]
    
    has_obstacle_in_row = np.any(eff_obs_map != BLANK, axis=1)
    has_obstacle_in_col = np.any(eff_obs_map != BLANK, axis=0)
    
    # 장애물이 전혀 없는 경우
    if not np.any(has_obstacle_in_row):
        return np.full((robot_diameter, robot_diameter), BLANK, dtype=obs_map.dtype)
    
    y_indices = np.flatnonzero(has_obstacle_in_row)
    x_indices = np.flatnonzero(has_obstacle_in_col)
    
    H, W = eff_obs_map.shape
    
    y_min = max(0, y_indices[0]-robot_diameter)
    y_max = min(H-1, y_indices[-1]+robot_diameter)
    x_min = max(0, x_indices[0]-robot_diameter)
    x_max = min(W-1, x_indices[-1]+robot_diameter)
    
    cropped_map = eff_obs_map[y_min:y_max+1, x_min:x_max+1]
    return cropped_map.copy()

# FIXME: vectorization 필요, 나중에 numpy 형태로 변환
def get_rect_indices_for_big_rect(
    cx: float, cy: float, 
    rect_width: float, rect_height: float,
    map_width: int, map_height: int, 
    theta: float) -> list[tuple[int, int]]:
    """
    Grid map에서 직사각형을 그릴 때, 색칠할 grid의 indices를 출력하는 method

    Args:
        cx, cy: 그릴 직사각형의 중심 좌표
        width, height: 그릴 직사각형의 가로, 세로 길이
        theta: 직사각형이 반시계 방향으로 회전한 정도(rad 단위)
    """
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    grid_indices = []
    
    # 각 grid가 직사각형 영역에 속하는지 검사
    # 검사 범위는 직사각형의 대각선 길이를 한 변의 길이로 하는 정사각형
    r = math.sqrt(rect_width**2 + rect_height**2) / 2
    x_min = max(0, int(math.floor(cx - r)))
    x_max = min(map_width - 1, int(math.ceil(cx + r)))
    y_min = max(0, int(math.floor(cy - r)))
    y_max = min(map_height - 1, int(math.ceil(cy + r)))

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
            if abs(nx) <= (rect_width / 2) + 1e-9 and abs(ny) <= (rect_height / 2) + 1e-9:
                grid_indices.append((ix, iy))
    
    return grid_indices

# FIXME: vectorization 필요
def get_rect_indices_for_small_rect(
    cx: float, cy: float, 
    rect_width: float, rect_height: float,
    map_width: int, map_height: int, 
    theta: float) -> list[tuple[int, int]]:
    """
    Grid map에서 직사각형을 그릴 때, 색칠할 grid의 indices를 출력하는 method

    Args:
        cx, cy: 그릴 직사각형의 중심 좌표
        width, height: 그릴 직사각형의 가로, 세로 길이
        theta: 직사각형이 반시계 방향으로 회전한 정도(rad 단위)
    """
    
    # 직사각형을 그릴 grid의 시작과 끝 좌표를 int 형식으로 얻음
    x_min = max(0, int(cx - rect_width/2 + 0.5))
    x_max = min(map_width - 1, int(cx + rect_width/2 + 0.5))
    y_min = max(0, int(cy - rect_height/2 + 0.5))
    y_max = min(map_height - 1, int(cy + rect_height/2 + 0.5))
    
    grid_indices = [(x, y) for x in range(x_min, x_max + 1) for y in range(y_min, y_max + 1)]
    
    return grid_indices

def get_leg_indices(
    cx: float, cy: float,
    map_width: int, map_height: int,
    area_width: int, area_height: int,
    leg_size: int,
    theta: float = 0.0,   # 회전 각도 (rad)
) -> np.ndarray:
    """_summary_

    Args:
        cx, cy (int): 책상 또는 의자의 중심 좌표
        area_width (int): 책상 또는 의자의 가로 길이
        area_height (int): 책상 또는 의자의 세로 길이
        leg_size (int, optional): 책상 또는 의자 다리의 두께.
        theta (float, optional): 책상 또는 의자가 반시계 방향으로 회전한 각도. Defaults to 0.0.

    Returns:
        np.ndarray: 책상 또는 의자 다리가 차지하는 grid의 indices (Shape: (N, 2))
    """
    # cx, cy: 테이블 중심 좌표, row_size 또는 col_size가 짝수인 경우 중심 위치로부터 약간 틀어져 있음
    # 테이블 기본 다리 좌표 (local frame)
    
    # 테이블의 각도가 0 rad일 때 각 leg 중심의 상대적인 좌표
    no_rotate_leg_centers = [
        (-area_width/2.0 + leg_size/2.0, -area_height/2.0 + leg_size/2.0),
        ( area_width/2.0 - leg_size/2.0, -area_height/2.0 + leg_size/2.0),
        (-area_width/2.0 + leg_size/2.0,  area_height/2.0 - leg_size/2.0),
        ( area_width/2.0 - leg_size/2.0,  area_height/2.0 - leg_size/2.0),
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
        if leg_size < 10:
            indices = get_rect_indices_for_small_rect(x, y, leg_size, leg_size, 
                                                      map_width, map_height, theta)
        else:
            indices = get_rect_indices_for_big_rect(x, y, leg_size, leg_size, 
                                                    map_width, map_height, theta)
        leg_grid_indices.extend(indices)
    
    leg_grid_indices_np = np.array(leg_grid_indices)
    
    return leg_grid_indices_np

def get_circle_indices(
    cx: float, cy: float, radius: float,
    map_width: int, map_height: int) -> list[tuple[int, int]]:
    
    indices = []
    
    # 1. 탐색 범위를 원이 포함되는 사각형 영역(Bounding Box)으로 제한 (속도 최적화)
    # 맵 경계를 벗어나지 않도록 클리핑(clipping) 해줍니다.
    min_x = max(0, int(math.floor(cx - radius)))
    max_x = min(map_width - 1, int(math.ceil(cx + radius)))
    min_y = max(0, int(math.floor(cy - radius)))
    max_y = min(map_height - 1, int(math.ceil(cy + radius)))
    
    # 반지름의 제곱을 미리 계산 (루트 연산을 피하기 위함)
    radius_sq = radius ** 2
    
    # 2. 사각형 영역 내에서 원의 방정식(x^2 + y^2 <= r^2)을 만족하는 픽셀만 필터링
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            # 그리드 중심점(x + 0.5, y + 0.5)과 원의 중심 사이의 거리 계산
            # (정밀도를 위해 그리드 칸의 중심을 기준으로 잡는 것이 좋습니다)
            dx = (x + 0.5) - cx
            dy = (y + 0.5) - cy
            
            if (dx ** 2 + dy ** 2) <= radius_sq:
                indices.append((x, y))
                
    return indices