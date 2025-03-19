import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_block import SSMTransformerBlock

class SSMTransformerModel(nn.Module):
    """Complete SSM-Transformer model with multiple layers"""
    def __init__(self, 
                 vocab_size, 
                 hidden_dim=512, 
                 num_layers=4, 
                 num_heads=8, 
                 state_dim=16, 
                 dropout=0.1,
                 max_seq_len=1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        self.blocks = nn.ModuleList([
            SSMTransformerBlock(hidden_dim, num_heads, state_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, 
                input_ids, 
                ssm_states=None, 
                past_key_values=None, 
                return_key_values=False):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if ssm_states is None:
            ssm_states = [None] * self.num_layers
        
        if past_key_values is None:
            past_key_values = [None] * self.num_layers
        
        past_length = 0
        if past_key_values[0] is not None:
            past_length = past_key_values[0][0].size(2)
        
        position_ids = torch.arange(
            past_length, past_length + seq_len, 
            dtype=torch.long, device=device
        ).unsqueeze(0).expand(batch_size, -1)
        
        x = self.embedding(input_ids) + self.position_embedding(position_ids)
        
        new_ssm_states = []
        present_key_values = [] if return_key_values else None
        
        for i, block in enumerate(self.blocks):
            if return_key_values:
                x, new_state, present_kv = block(
                    x, 
                    ssm_state=ssm_states[i], 
                    past_key_values=past_key_values[i],
                    return_key_values=True
                )
                present_key_values.append(present_kv)
            else:
                x, new_state = block(
                    x, 
                    ssm_state=ssm_states[i], 
                    past_key_values=past_key_values[i]
                )
            
            new_ssm_states.append(new_state)
        
        x = self.norm(x)
        logits = self.head(x)
        
        if return_key_values:
            return logits, new_ssm_states, present_key_values
        
        return logits, new_ssm_states