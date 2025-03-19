import torch
import time
from .base_cache import BaseCacheManager

class NoCacheManager(BaseCacheManager):
    
    def generate(self, input_ids, max_length=50, temperature=1.0, top_k=50):
        start_time = time.time()
        
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            logits, _ = self.model(input_ids)
        
        for _ in range(max_length):
            with torch.no_grad():
                logits, _ = self.model(generated_ids)
            
            next_token = self._sample_next_token(logits, temperature, top_k)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        
        total_time = time.time() - start_time
        
        return {
            'output': generated_ids,
            'time': total_time,
            'memory_usage': 0  # No cache, so memory usage is 0
        }