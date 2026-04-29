import numpy as np
from PIL import Image
import torch

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

def float_to_int_coord(cx: float, cy: float) -> tuple[int, int]:
    return int(cx+0.5), int(cy+0.5)


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