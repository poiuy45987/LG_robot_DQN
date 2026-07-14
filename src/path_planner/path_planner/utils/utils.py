import numpy as np
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
