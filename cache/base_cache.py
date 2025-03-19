from abc import ABC, abstractmethod
import torch

class BaseCacheManager(ABC):
    
    def __init__(self, model):
        self.model = model
    
    @abstractmethod
    def generate(self, input_ids, max_length=50, temperature=1.0, top_k=50):
        pass
    
    def _sample_next_token(self, logits, temperature=1.0, top_k=50):
        next_token_logits = logits[:, -1, :] / temperature
        
        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)

        probs = torch.nn.functional.softmax(top_k_logits, dim=-1)

        next_token_idx = torch.multinomial(probs, num_samples=1)
        next_token = torch.gather(top_k_indices, -1, next_token_idx)
        
        return next_token