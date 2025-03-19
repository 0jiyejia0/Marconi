import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SSMLayer(nn.Module):
    def __init__(self, hidden_dim, state_dim=16, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        self.A = nn.Parameter(torch.randn(state_dim, state_dim))
        self.B = nn.Parameter(torch.randn(state_dim, hidden_dim))
        self.C = nn.Parameter(torch.randn(hidden_dim, state_dim))
        
        log_dt = torch.linspace(math.log(dt_min), math.log(dt_max), hidden_dim)
        self.log_dt = nn.Parameter(log_dt)
        
        self._init_parameters()
    
    def _init_parameters(self):
        nn.init.normal_(self.A, mean=0.0, std=0.01)
        nn.init.normal_(self.B, mean=0.0, std=0.01)
        nn.init.normal_(self.C, mean=0.0, std=0.01)
    
    def forward(self, x, state=None):
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        dt = torch.mean(torch.exp(self.log_dt))
        A_discrete = torch.matrix_exp(self.A * dt)
        
        B_discrete = torch.matmul(
            torch.inverse(self.A),
            torch.matmul(A_discrete - torch.eye(self.state_dim, device=device), self.B)
        )
        
        if state is None:
            state = torch.zeros(batch_size, self.state_dim, device=device)
        
        outputs = []
        current_state = state
        
        for t in range(seq_len):
            current_state = torch.bmm(
                current_state.unsqueeze(1), 
                A_discrete.expand(batch_size, -1, -1)
            ).squeeze(1) + torch.matmul(x[:, t, :], B_discrete.t())
            
            output = torch.matmul(current_state, self.C.t())
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        
        return outputs, current_state