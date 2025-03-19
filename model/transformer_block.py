import torch
import torch.nn as nn
import torch.nn.functional as F
from .ssm_layer import SSMLayer
from .attention import SelfAttention

class SSMTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, state_dim=16, dropout=0.1):
        super().__init__()
        self.ssm_layer = SSMLayer(hidden_dim, state_dim)
        self.attention = SelfAttention(hidden_dim, num_heads)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x, ssm_state=None, past_key_values=None, return_key_values=False):
        residual = x
        ssm_out, new_ssm_state = self.ssm_layer(self.norm1(x), ssm_state)
        x = residual + self.dropout(ssm_out)
        
        residual = x
        if return_key_values:
            attn_out, present_key_values = self.attention(
                self.norm2(x), 
                past_key_values=past_key_values, 
                return_key_values=True
            )
        else:
            attn_out = self.attention(
                self.norm2(x), 
                past_key_values=past_key_values
            )
        x = residual + self.dropout(attn_out)
        
        residual = x
        x = residual + self.dropout(self.mlp(self.norm3(x)))
        
        if return_key_values:
            return x, new_ssm_state, present_key_values
        
        return x, new_ssm_state