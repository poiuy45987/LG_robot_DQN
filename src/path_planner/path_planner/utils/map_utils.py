from __future__ import annotations

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


@dataclass
class MapStartAnalysis:
    """A valid robot start and the map layers derived while selecting it."""

    start_pos: tuple[float, float]
    collision_map: np.ndarray
    reachable: np.ndarray
    coverable: np.ndarray
    coverable_area_rate: float


def analyze_map_startability(
    obstacles: np.ndarray,
    robot_size: int,
    *,
    start_mode: str,
    rng: np.random.Generator,
    min_coverable_area_rate: float = 0.1,
    eff_size: BoundingBox | None = None,
    collision_map: np.ndarray | None = None,
) -> MapStartAnalysis | None:
    """Select a usable start position and compute its static map layers once.

    ``start_mode`` is ``"edge"`` for a randomly selected edge position and
    ``"corner"`` for the first usable corner in a deterministic order.  A map
    is usable only when the selected start has at least
    ``min_coverable_area_rate`` of its effective area coverable.
    """
    if not 0.0 <= min_coverable_area_rate <= 1.0:
        raise ValueError("min_coverable_area_rate must be between 0 and 1.")

    obstacles = (obstacles != 0).astype(np.uint8, copy=False)
    H, W = obstacles.shape
    if eff_size is None:
        eff_size = get_eff_size_from_obs_map(obstacles)

    x_min, x_max = eff_size.x_min, eff_size.x_max
    y_min, y_max = eff_size.y_min, eff_size.y_max
    if not (0 <= x_min <= x_max < W and 0 <= y_min <= y_max < H):
        raise ValueError("eff_size must be an inclusive bounding box inside obstacles.")

    robot_half_size = robot_size // 2
    min_x, max_x = x_min + robot_half_size, x_max - robot_half_size
    min_y, max_y = y_min + robot_half_size, y_max - robot_half_size
    if min_x > max_x or min_y > max_y:
        return None

    robot_mask = make_robot_mask(robot_size)
    y_idx, x_idx = np.nonzero(robot_mask)
    robot_mask_offsets = np.column_stack([
        x_idx - robot_half_size,
        y_idx - robot_half_size,
    ]).astype(np.int32)
    if collision_map is None:
        collision_map = dilation_obstacles(obstacles, robot_mask)
    elif collision_map.shape != obstacles.shape:
        raise ValueError("collision_map must have the same shape as obstacles.")
    total_area = (x_max - x_min + 1) * (y_max - y_min + 1)

    if start_mode == "edge":
        x_range = np.arange(min_x, max_x + 1)
        y_range = np.arange(min_y + 1, max_y)
        candidates = np.concatenate([
            np.stack([x_range, np.full_like(x_range, max_y)], axis=1),
            np.stack([x_range, np.full_like(x_range, min_y)], axis=1),
            np.stack([np.full_like(y_range, min_x), y_range], axis=1),
            np.stack([np.full_like(y_range, max_x), y_range], axis=1),
        ])
        # A rejected connected component cannot become valid at another one
        # of its edge positions, so evaluate each component only once.
        remaining = candidates[collision_map[candidates[:, 1], candidates[:, 0]] == 0]
        while len(remaining) > 0:
            chosen_idx = int(rng.integers(len(remaining)))
            start_x, start_y = remaining[chosen_idx]
            reachable = compute_reachable_centers(collision_map, start_x, start_y)
            coverable = compute_coverable_cells_from_reachable(
                obstacles, reachable, robot_mask_offsets
            )
            coverable_area_rate = float(coverable.sum(dtype=np.int64) / total_area)
            if coverable_area_rate >= min_coverable_area_rate:
                return MapStartAnalysis(
                    start_pos=(float(start_x), float(start_y)),
                    collision_map=collision_map,
                    reachable=reachable,
                    coverable=coverable,
                    coverable_area_rate=coverable_area_rate,
                )
            remaining = remaining[reachable[remaining[:, 1], remaining[:, 0]] == 0]

    elif start_mode == "corner":
        candidates = (
            (min_x, max_y),
            (max_x, max_y),
            (max_x, min_y),
            (min_x, min_y),
        )
        for start_x, start_y in candidates:
            if collision_map[start_y, start_x]:
                continue
            reachable = compute_reachable_centers(collision_map, start_x, start_y)
            coverable = compute_coverable_cells_from_reachable(
                obstacles, reachable, robot_mask_offsets
            )
            coverable_area_rate = float(coverable.sum(dtype=np.int64) / total_area)
            if coverable_area_rate >= min_coverable_area_rate:
                return MapStartAnalysis(
                    start_pos=(float(start_x), float(start_y)),
                    collision_map=collision_map,
                    reachable=reachable,
                    coverable=coverable,
                    coverable_area_rate=coverable_area_rate,
                )
    else:
        raise ValueError('start_mode must be "edge" or "corner".')

    return None


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

def get_rect_indices_for_big_rect(
    cx: float, cy: float, 
    rect_width: float, rect_height: float,
    map_width: int, map_height: int, 
    theta: float, allow_crop: bool=False) -> np.ndarray:
    """
    Grid map에서 직사각형을 그릴 때, 색칠할 grid의 indices를 출력하는 method
    

    Args:
        cx, cy: 그릴 직사각형의 중심 좌표
        rect_width, rect_height: 그릴 직사각형의 가로, 세로 길이
        map_width, map_height: 직사각형을 그릴 map의 가로, 세로 길이
        theta: 직사각형이 반시계 방향으로 회전한 정도(rad 단위)
        allow_crop:
            - False (기본값): 식탁의 일부라도 맵 밖으로 나가면 배치 불가로 보고 빈 배열 반환.
            - True: 맵 밖으로 나가는 부분은 잘라내고(crop), 맵 안에 들어온 픽셀만 반환. 
    
    Returns:
        np.ndarray:
            - Map 상에서 직사각형이 차지하는 grid의 index를 (N, 2) 형태로 반환.
            - x좌표: [:, 0] / y좌표: [:, 1]
            - 직사각형이 유효하지 않은 경우 빈 배열 반환
    """
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    hw, hh = rect_width / 2.0, rect_height / 2.0
    
    # 직사각형이 map 내부에 완전히 들어와야 하는 경우(allow_crop=False): 4개 모서리가 맵 내부인지 사전 검사
    if not allow_crop:
        local_corners = np.array([
            [-hw, -hh],
            [ hw, -hh],
            [ hw,  hh],
            [-hw,  hh]
        ])
        world_x = cx + (local_corners[:, 0] * cos_t - local_corners[:, 1] * sin_t)
        world_y = cy + (local_corners[:, 0] * sin_t + local_corners[:, 1] * cos_t)

        # 모서리 중 하나라도 맵 밖으로 나가면 빈 array를 반환
        if np.any(world_x < 0) or np.any(world_x >= map_width) or \
           np.any(world_y < 0) or np.any(world_y >= map_height):
            return np.empty((0, 2), dtype=int)
    
    # 각 grid가 직사각형 영역에 속하는지 검사
    # 검사 범위는 직사각형의 대각선 길이를 한 변의 길이로 하는 정사각형
    r = math.sqrt(rect_width**2 + rect_height**2) / 2
    x_min = max(0, int(math.floor(cx - r)))
    x_max = min(map_width - 1, int(math.ceil(cx + r)))
    y_min = max(0, int(math.floor(cy - r)))
    y_max = min(map_height - 1, int(math.ceil(cy + r)))

    # 예외 처리: 범위를 벗어난 경우
    if x_min > x_max or y_min > y_max:
        return np.empty((0, 2), dtype=int)
    
    # 검사 범위 정사각형 범위 내의 grid 좌표 배열 생성
    ix_vals = np.arange(x_min, x_max + 1)
    iy_vals = np.arange(y_min, y_max + 1)
    IX, IY = np.meshgrid(ix_vals, iy_vals, indexing='xy')
    
    # 그리드 칸의 중심점: (ix.5, iy.5)
    # (dx, dy): 직사각형 중심점 (cx, cy)를 원점으로 봤을 때 grid 중심점의 좌표
    DX = (IX + 0.5) - cx
    DY = (IY + 0.5) - cy
    
    # 역회전 변환 (Rotated frame -> Local axis-aligned frame)
    NX = DX * cos_t + DY * sin_t
    NY = -DX * sin_t + DY * cos_t
    
    # NX, NY 중 직사각형에 포함되는 grid index를 고름
    mask = (np.abs(NX) <= (rect_width / 2.0) + 1e-9) & \
           (np.abs(NY) <= (rect_height / 2.0) + 1e-9)
    valid_ix = IX[mask]
    valid_iy = IY[mask]
    
    return np.column_stack((valid_ix, valid_iy))

def get_rect_indices_for_small_rect(
    cx: float, cy: float, 
    rect_width: float, rect_height: float,
    map_width: int, map_height: int, 
    theta: float, allow_crop: bool=False) -> list[tuple[int, int]]:
    """
    Grid map에서 직사각형을 그릴 때, 색칠할 grid의 indices를 출력하는 method

    Args:
        cx, cy: 그릴 직사각형의 중심 좌표
        rect_width, rect_height: 그릴 직사각형의 가로, 세로 길이
        map_width, map_height: 직사각형을 그릴 map의 가로, 세로 길이
        theta: 직사각형이 반시계 방향으로 회전한 정도(rad 단위)
        allow_crop:
            - False (기본값): 식탁의 일부라도 맵 밖으로 나가면 배치 불가로 보고 빈 배열 반환.
            - True: 맵 밖으로 나가는 부분은 잘라내고(crop), 맵 안에 들어온 픽셀만 반환.
    
    Returns:
        np.ndarray:
            - Map 상에서 직사각형이 차지하는 grid의 index를 (N, 2) 형태로 반환.
            - x좌표: [:, 0] / y좌표: [:, 1]
            - 직사각형이 유효하지 않은 경우 빈 배열 반환
    """
    # 직사각형이 map에서 잘리는지 검사
    raw_x_min = int(cx - rect_width / 2.0 + 0.5)
    raw_x_max = int(cx + rect_width / 2.0 + 0.5)
    raw_y_min = int(cy - rect_height / 2.0 + 0.5)
    raw_y_max = int(cy + rect_height / 2.0 + 0.5)
    if not allow_crop:
        if raw_x_min < 0 or raw_x_max >= map_width or \
           raw_y_min < 0 or raw_y_max >= map_height:
            return np.empty((0, 2), dtype=int)
    
    
    # 직사각형을 그릴 grid의 시작과 끝 좌표를 int 형식으로 얻음
    x_min = max(0, int(cx - rect_width/2 + 0.5))
    x_max = min(map_width - 1, int(cx + rect_width/2 + 0.5))
    y_min = max(0, int(cy - rect_height/2 + 0.5))
    y_max = min(map_height - 1, int(cy + rect_height/2 + 0.5))
    
    if x_min > x_max or y_min > y_max:
        return np.empty((0, 2), dtype=int)
    
    ix_vals = np.arange(x_min, x_max + 1)
    iy_vals = np.arange(y_min, y_max + 1)
    IX, IY = np.meshgrid(ix_vals, iy_vals, indexing='xy')
    
    return np.column_stack((IX.ravel(), IY.ravel()))

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
    
    leg_indices_list = []
    
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
        if theta == 0.0 or leg_size < 10:
            indices = get_rect_indices_for_small_rect(x, y, leg_size, leg_size, 
                                                      map_width, map_height, theta, allow_crop=True)
        else:
            indices = get_rect_indices_for_big_rect(x, y, leg_size, leg_size, 
                                                    map_width, map_height, theta, allow_crop=True)
        
        if len(indices) > 0:
            leg_indices_list.append(indices)
    
    if not leg_indices_list:
        return np.empty((0, 2), dtype=int)
    
    return np.vstack(leg_indices_list)

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


def compute_frontier_map( 
    cleaned_map: np.ndarray, 
    coverable: np.ndarray,
    use_dilation: bool = True
) -> np.ndarray:
    """
    cleaned_map과 coverable 맵을 기반으로 Frontier 맵을 계산.

    Args:
        cleaned_map (np.ndarray): 청소된 횟수/여부 맵 (H, W). 0 초과면 청소된 구역.
        coverable (np.ndarray): 청소 가능 구역 맵 (H, W). 1이면 청소 가능 영역.
        use_dilation (bool): Frontier 영역을 3x3 커널로 Dilation(팽창)시킬지 여부.

    Returns:
        np.ndarray: uint8 형태의 Frontier 맵 (H, W). Frontier 위치는 1, 나머지는 0.
    """
    H, W = cleaned_map.shape
    frontier_map = np.zeros((H, W), dtype=np.uint8)

    # 1. 청소된 영역이 하나도 없으면 Frontier도 없음
    cleaned_mask = (cleaned_map > 0)
    if not np.any(cleaned_mask):
        return frontier_map

    # 2. 유효한 미청소 영역 마스크 (청소 가능하고 아직 청소되지 않은 영역)
    valid_uncleaned_mask = (coverable == 1) & (cleaned_map == 0)

    # 3. 청소된 셀들(cleaned > 0)의 8방향 이웃 좌표 계산
    cy, cx = np.where(cleaned_mask)
    offsets = np.array([
        [-1, -1], [-1, 0], [-1, 1],
        [ 0, -1],          [ 0, 1],
        [ 1, -1], [ 1, 0], [ 1, 1]
    ], dtype=np.int32)

    cleaned_coords = np.column_stack((cx, cy))
    
    # 청소된 셀 주변 8방향 좌표들을 구하고 중복 제거
    roi_coords = np.unique((cleaned_coords[:, None, :] + offsets[None, :, :]).reshape(-1, 2), axis=0)

    # 4. 지도 밖으로 나가는 좌표(In-bounds) 제외
    valid_mask = (roi_coords[:, 0] >= 0) & (roi_coords[:, 0] < W) & \
                 (roi_coords[:, 1] >= 0) & (roi_coords[:, 1] < H)
    valid_coords = roi_coords[valid_mask]
    rx, ry = valid_coords[:, 0], valid_coords[:, 1]

    # 5. Raw Frontier 조건: (청소된 영역의 이웃) AND (유효한 미청소 영역)
    raw_frontier_mask = valid_uncleaned_mask[ry, rx]
    rx, ry = rx[raw_frontier_mask], ry[raw_frontier_mask]

    if len(rx) == 0:
        return frontier_map

    # 6. Raw Frontier 마스크 생성
    raw_frontier_patch = np.zeros((H, W), dtype=np.uint8)
    raw_frontier_patch[ry, rx] = 1

    # 7. 3x3 Kernel Dilation 적용 및 최종 Frontier 산출
    if use_dilation:
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated_frontier = cv2.dilate(raw_frontier_patch, kernel, iterations=1)
        
        # Dilation된 영역 중 (청소 가능 & 아직 청소 안 된) 영역만 최종 1로 설정
        valid_dilated = (dilated_frontier == 1) & valid_uncleaned_mask
        frontier_map[valid_dilated] = 1
    else:
        frontier_map[raw_frontier_patch == 1] = 1

    return frontier_map
