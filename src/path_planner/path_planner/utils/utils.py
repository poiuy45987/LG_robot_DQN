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

def print_model_info(model: torch.nn.Module):
    
    # 1. 전체 파라미터 수 및 학습 가능한(Trainable) 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 2. 파라미터 용량 계산 (기본 float32 = 4 bytes 기준)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers()) # BatchNorm 등의 버퍼 용량
    
    total_size_mb = (param_size + buffer_size) / (1024 ** 2)

    print(f"Total Parameters     : {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Trainable Parameters : {trainable_params:,}")
    print(f"Model File Size      : {total_size_mb:.2f} MB")