import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon_in', torch.empty(in_features))
        self.register_buffer('weight_epsilon_out', torch.empty(out_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.sigma_init = sigma_init
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / (self.in_features ** 0.5)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / (self.in_features ** 0.5))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / (self.out_features ** 0.5))
        
    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_epsilon_in.device) # 평균 0, 표준편차 1인 정규분포 (음수/양수 섞임)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        self.weight_epsilon_in.copy_(self._scale_noise(self.in_features))
        self.weight_epsilon_out.copy_(self._scale_noise(self.out_features))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def forward(self, x):
        if self.training:
            # forward 시점에 외적(ger)을 수행하여 일시적으로 행렬 생성
            # 메모리에 저장하지 않으므로 속도와 용량 모두 이득입니다.
            weight = self.weight_mu + self.weight_sigma * torch.ger(self.weight_epsilon_out, self.weight_epsilon_in)
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

class CNN_ReDQN(nn.Module):
    def __init__(self, **kwargs):
        super(CNN_ReDQN, self).__init__()
        
        # kwargs 처리
        self.action_size = kwargs.get('action_size', 4)
        use_noisy = kwargs.get('use_noisy', False)
        self.grid_map_size = kwargs.get('grid_map_size', 51)
        self.do_normalize = kwargs.get('do_normalize', False)
        self.momentum = 0.1
        
        # Normalization을 할 경우, map data의 평균과 표준편차를 저장
        if self.do_normalize:
            # map_data (batch, 3, 51, 51)의 채널별 평균과 표준편차를 저장
            # 초기값은 0과 1로 설정
            self.register_buffer('map_mean', torch.zeros(1, 3, 1, 1))
            self.register_buffer('map_var', torch.ones(1, 3, 1, 1))
            # 첫 번째 배치가 들어왔는지 확인하는 플래그 (0: 미수신, 1: 수신완료)
            self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        
        # Image encoder
        self.map_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), # 51 -> 51
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0), # 51 -> 25
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0), # 25 -> 12
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 512),
            nn.ReLU(),
        )
        
        # Vector encoder
        self.vec_encoder = nn.Sequential(
            nn.Linear(11, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        
        # Q-value network
        self.q_value_net = None
        if use_noisy:
            self.q_value_net = nn.Sequential(
                NoisyLinear(512+64, 256),
                nn.ReLU(),
                NoisyLinear(256, 256),
                nn.ReLU(),
                NoisyLinear(256, self.action_size)
            )
        else:
            self.q_value_net = nn.Sequential(
                nn.Linear(512+64, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, self.action_size)
            )

    def _normalize_map(self, map_data):
        if not self.do_normalize:
            return map_data

        # 첫 번째 배치가 들어왔을 때만 평균/표준편차 계산 및 고정
        if self.training:
            with torch.no_grad():
                # 채널별(dim=1)로 평균과 표준편차 계산
                # (Batch, Width, Height) 차원에 대해 평균을 냄
                batch_mean = map_data.mean(dim=(0, 2, 3), keepdim=True)
                batch_var = map_data.var(dim=(0, 2, 3), keepdim=True)
                    
                if self.initialized == 0:
                    self.map_mean.copy_(batch_mean)
                    self.map_var.copy_(batch_var + 1e-8) # Zero division 방지
                    self.initialized.fill_(1)      # 플래그 업데이트
                    print(f"Map Normalization Initialized! Mean: {self.map_mean.flatten()}")
                else:
                    self.map_mean.copy_(self.momentum * batch_mean + (1 - self.momentum) * self.map_mean)
                    self.map_var.copy_(self.momentum * batch_var + (1 - self.momentum) * self.map_var)

        # 2. 저장된 버퍼 값을 사용하여 정규화 수행
        return (map_data - self.map_mean) / torch.sqrt(self.map_var + 1e-8)
    
    def forward(self, map_data, vec_data, proximity_penalty=None):
        """
        global_x: (batch, 3, 20, 20)
        """
        # Map data normalization
        map_data = self._normalize_map(map_data)
        
        # 각각 encoder에 통과
        feat_map = self.map_encoder(map_data)
        feat_vec = self.vec_encoder(vec_data)
        
        # 특징 결합 (dim=1은 batch 이후의 채널/특징 차원)
        features = torch.cat((feat_map, feat_vec), dim=1)
        
        q_values = self.q_value_net(features)
        
        # # 4. Dynamic Incentive 결합 (논문 기법: 장애물 근처 패널티)
        # # 만약 proximity_penalty가 주어지면 Q-value에서 차감하여 장애물 방향 선택을 억제
        # if proximity_penalty is not None:
        #     q_values = q_values - proximity_penalty
            
        return q_values

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()