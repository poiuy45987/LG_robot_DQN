import numpy as np
import math
from PIL import Image
import torch

from .map_generator import BLANK

def get_device_info(device: torch.device):
    
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


def set_torch_seed(seed: int):
    
    # Pytorch 난수 생성기의 seed 설정
    torch.manual_seed(seed) # PyTorch CPU 난수 고정
    if torch.cuda.is_available(): # PyTorch GPU 난수 고정
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 멀티 GPU 사용 시
        
    # 결정론적 연산 설정 (속도는 조금 느려질 수 있지만 결과는 항상 동일)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def float_to_int_coord(cx, cy):
    
    # 단일 float 또는 int인 경우
    if isinstance(cx, (int, float, np.floating, np.integer)):
        return math.floor(cx+0.5), math.floor(cy+0.5)
    
    # list나 array인 경우
    return np.floor(np.asarray(cx) + 0.5).astype(int), np.floor(np.asarray(cy) + 0.5).astype(int)


def display_image(img_array: np.ndarray):
    """
    numpy 배열 이미지를 주피터 또는 시스템 뷰어에 최적화된 방식으로 출력합니다.
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
        
def normal_round(x):
    """
    x(단일 숫자 또는 여러 숫자)를 반올림하는 함수.
    모든 경우 X.5를 X+1로 반올림하도록 함.
    기존 round 함수는 X.5를 반올린할 때 가장 가까운 짝수로 반올림함.
    """
    
    # 1. 단일 숫자(float, int)인 경우 -> 순수 파이썬으로 초고속 처리
    if isinstance(x, (int, float, np.floating, np.integer)):
        return int(x + 0.5) if x >= 0 else int(x - 0.5)

    # 2. 리스트나 배열인 경우 -> 넘파이의 벡터화 연산 활용
    x_np = np.asanyarray(x)
    return np.floor(x_np + 0.5).astype(int)

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
        eff_size (tuple[float, float]): 효과적인 맵 크기 (높이, 너비)
        crop_size (int): 실제로 채워버릴 정사각형의 한 변의 길이
        robot_diameter (int): 로봇의 지름 (안전 마진으로 활용할 상하좌우 확장 크기)
        stride (int): 탐색 윈도우의 이동 간격 (crop_size보다 작은 값 가능)
        fill_value (int): 채울 장애물 종류 (기본값: WALL = 1)
    """
    H, W = obs.shape
    if eff_size is not None:
        H, W = eff_size[0], eff_size[1]

    # 1. 기본적으로 모든 영역을 DQN 구역(1)으로 초기화한 '새로운 제어 맵' 생성
    # 검증된 청정 구역만 Heuristic(0)으로 깎아 나가는 방식이 안전합니다.
    mode_map = np.ones((H, W), dtype=np.int8)
    
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
            if np.all(check_window == BLANK):
                
                # --- [4번 요구사항: 실제 기록은 늘리기 전 원래 정사각형 사이즈만] ---
                mode_map[y : y + crop_size, x : x + crop_size] = 0

    return mode_map