import math
import numpy as np
import cv2
from collections import deque


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