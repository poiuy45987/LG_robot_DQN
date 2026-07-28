import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

def get_next_layer_dim(input_dim: int, kernel_size: int, stride: int, padding: int) -> int:
    """2D convolution layer의 output dimension 계산"""
    return (input_dim + 2*padding - kernel_size) // stride + 1

class LSTMPolicyNetwork(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        # kwargs 처리
        ang_diff_diridx = kwargs.get('ang_diff_diridx')
        if ang_diff_diridx is None:
            raise ValueError("PolicyNetwork를 초기화하려면 'ang_diff_diridx' 인자가 반드시 필요합니다")
        ang_diff_tensor = torch.tensor(ang_diff_diridx, dtype=torch.float32)
        raw_actions = torch.stack([torch.cos(ang_diff_tensor), torch.sin(ang_diff_tensor)], dim=1)
        self.register_buffer('raw_actions', raw_actions) # FIXME: (cos, sin) 형태의 모든 action들을 저장. Shape: (action_num, action_dim)
        
        self.action_dim = raw_actions.size(1) # Input으로 받는 각 action의 dimension (Default: [cos theta, sin theta])
        self.action_num = raw_actions.size(0) # Action 선택지 수
        self.action_enc_dim = kwargs.get('action_enc_dim', 128) # Action vector를 encoding하는 dimension (Default: 128)
        
        self.local_view_dim = kwargs.get('local_view_dim', 51) # Local view map의 dimension
        self.stack_steps = kwargs.get('stack_steps', 1) # State에서 local view map의 layer 수
        self.map_feat_dim = kwargs.get('map_feat_dim', 512) # Local view map의 feature dimension
        self.loc_vec_in_dim = kwargs.get('loc_vec_in_dim', 16) # Additional state vector의 dimension
        self.vec_feat_dim = kwargs.get('vec_feat_dim', 128) # Additional state vector의 feature dimension
        
        self.lstm_in_dim = kwargs.get('lstm_in_dim', 5) # LSTM cell block에 들어가는 raw input vector (Default: [x, y, cos theta, sin theta, coverage])
        self.lstm_in_prj_dim = kwargs.get('lstm_in_prj_dim', 32) # LSTM cell block에 직접 들어가는 input vector (Default: 32)
        self.lstm_hid_dim = kwargs.get('lstm_hid_dim', 512)
        
        # 1. Action Encoder: Encoded action은 각 action이 feature vector에서 어떤 element를 봐야하는지 표시함.
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_enc_dim),
        )
        
        # 2. Local view feature extractor: Local view map CNN feature extractor + Ray distance vector feature extractor
        
        # Local view map feature extractor
        next_dim = get_next_layer_dim(self.local_view_dim, kernel_size=3, stride=1, padding=1) # 51 -> 51
        next_dim = get_next_layer_dim(next_dim, kernel_size=3, stride=2, padding=0) # 51 -> 25
        final_dim = get_next_layer_dim(next_dim, kernel_size=3, stride=2, padding=0) # 25 -> 12
        self.map_enc = nn.Sequential(
            nn.Conv2d(4*self.stack_steps, 32, kernel_size=3, stride=1, padding=1), # 51 -> 51
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0), # 51 -> 25
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0), # 25 -> 12
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * final_dim * final_dim, self.map_feat_dim),
        )
        
        # Additional state vector feature extractor
        self.vec_enc = nn.Sequential(
            nn.Linear(self.loc_vec_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.vec_feat_dim),
        )
        
        # 3. LSTM: Input projection + LSTM cell block
        
        self.lstm_inp_prj = nn.Sequential(
            nn.Linear(self.lstm_in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.lstm_in_prj_dim)
        )

        nn.LSTM
        self.lstm = nn.LSTMCell(input_size=self.lstm_in_prj_dim, hidden_size=self.lstm_hid_dim)
        
        # 4. Compress network: Encoded action의 차원과 일치시키기 위해 feature를 compresse하는 network
        total_feat_dim = self.map_feat_dim + self.vec_feat_dim + self.lstm_hid_dim
        self.cmps_net = nn.Linear(total_feat_dim, self.action_enc_dim)
        # self.cmps_net = nn.Sequential(
        #     nn.Linear(total_feat_dim, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, self.action_enc_dim)
        # )

    def forward(self, loc_map_data, loc_vec_data, glob_vec_data, h_t, c_t) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        loc_map_data: 
        local_view: (batch_size, 1, H, W) - Polar rotated local view
        last_action: (batch_size,) - 직전 선택한 액션 인덱스 (필요시 x_t에 concatenate 하거나 활용)
        h_t, c_t: (batch_size, lstm_hidden_dim) - LSTM의 이전 히든/셀 상태
        """
        
        # 1. Local feature 추출
        map_feat = self.map_enc(loc_map_data) # Shape: (B, map_feat_dim=512)
        vec_feat = self.vec_enc(loc_vec_data) # Shape: (B, vec_feat_dim=64)
        loc_feat = torch.cat((map_feat, vec_feat), dim=1) # Shape: (B, local_feat_dim=512+64)
        
        # 2. LSTM output 계산
        lstm_in = self.lstm_inp_prj(glob_vec_data) # Shape: (B, lstm_in_prj_dim=32)
        next_h_t, next_c_t = self.lstm(lstm_in, (h_t, c_t)) # Shape: (B, lstm_hid_dim=512)
        
        # 3. Feature fusion & Compression
        total_feat = torch.cat((loc_feat, next_h_t), dim=1)
        cmps_feat = self.cmps_net(total_feat) # Shape: (B, action_enc_dim=128)
        
        # 4. 각 action의 attention score 계산
        encoded_actions = self.action_encoder(self.raw_actions) # Shape: (action_num=16, action_enc_dim=128)
        attn_scores = torch.matmul(cmps_feat, encoded_actions.t()) # Shape: (B, action_enc_dim=128) @ (action_enc_dim=128, action_num=16) = (B, action_num=16)
        action_probs = F.softmax(attn_scores, dim=1) # Shape: (B, action_num=16)
        
        return action_probs, next_h_t, next_c_t