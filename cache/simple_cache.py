import torch
import time
from collections import OrderedDict
from .base_cache import BaseCacheManager

class SimpleKVCache:
    def __init__(self, max_entries=100):
        self.max_entries = max_entries
        self.cache = OrderedDict()
        
        self.hit_count = 0
        self.miss_count = 0
        self.memory_usage = 0
    
    def _generate_key(self, tokens):
        return hash(tuple(tokens))
    
    def lookup(self, tokens):
        best_match = None
        best_length = 0
        
        for prefix_length in range(len(tokens), 0, -1):
            prefix = tokens[:prefix_length]
            key = self._generate_key(prefix)
            
            if key in self.cache:
                best_match = self.cache[key]
                best_length = prefix_length
                self.cache.move_to_end(key)
                self.hit_count += 1
                return best_match, best_length
        
        self.miss_count += 1
        return None, 0
    
    def update(self, tokens, kv_caches, ssm_states):
        key = self._generate_key(tokens)
        
        if key in self.cache:
            old_kv, old_ssm = self.cache[key]
            self._subtract_memory_usage(old_kv, old_ssm)
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_entries:
            _, (old_kv, old_ssm) = self.cache.popitem(last=False)
            self._subtract_memory_usage(old_kv, old_ssm)
        
        self._add_memory_usage(kv_caches, ssm_states)
        
        self.cache[key] = (kv_caches, ssm_states)
    
    def _add_memory_usage(self, kv_caches, ssm_states):
        usage = self._calculate_memory_usage(kv_caches, ssm_states)
        self.memory_usage += usage
    
    def _subtract_memory_usage(self, kv_caches, ssm_states):
        usage = self._calculate_memory_usage(kv_caches, ssm_states)
        self.memory_usage -= usage
    
    def _calculate_memory_usage(self, kv_caches, ssm_states):
        usage = 0
        
        for k, v in kv_caches:
            usage += k.numel() * k.element_size()
            usage += v.numel() * v.element_size()
        
        for state in ssm_states:
            usage += state.numel() * state.element_size()
        
        return usage
    
    def get_hit_rate(self):
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def get_memory_usage_mb(self):
        return self.memory_usage / (1024 * 1024)
    
    def clear(self):
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        self.memory_usage = 0


class SimpleKVCacheManager(BaseCacheManager):
    def __init__(self, model, max_entries=100):
        super().__init__(model)
        self.cache = SimpleKVCache(max_entries=max_entries)
    
    def generate(self, input_ids, max_length=50, temperature=1.0, top_k=50):
        start_time = time.time()
        device = input_ids.device
        
        generated_ids = input_ids.clone()
        
        initial_tokens = input_ids[0].cpu().tolist()
        cached_data, match_length = self.cache.lookup(initial_tokens)
        
        if cached_data:
            kv_caches, ssm_states = cached_data
            
            if match_length < len(initial_tokens):
                with torch.no_grad():
                    _, new_ssm_states, new_kv_caches = self.model(
                        input_ids[:, match_length:], 
                        ssm_states=ssm_states,
                        past_key_values=kv_caches,
                        return_key_values=True
                    )
                ssm_states = new_ssm_states
                kv_caches = new_kv_caches
        else:
            with torch.no_grad():
                _, ssm_states, kv_caches = self.model(
                    input_ids, 
                    return_key_values=True
                )
            
            self.cache.update(initial_tokens, kv_caches, ssm_states)
        
        for _ in range(max_length):
            current_input = generated_ids[:, -1].unsqueeze(-1)
            
            with torch.no_grad():
                logits, ssm_states, kv_caches = self.model(
                    current_input,
                    ssm_states=ssm_states, 
                    past_key_values=kv_caches,
                    return_key_values=True
                )
            
            next_token = self._sample_next_token(logits, temperature, top_k)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            if _ % 5 == 0:
                current_tokens = generated_ids[0].cpu().tolist()
                self.cache.update(current_tokens, kv_caches, ssm_states)
        
        total_time = time.time() - start_time
        
        return {
            'output': generated_ids,
            'time': total_time,
            'hit_rate': self.cache.get_hit_rate(),
            'memory_usage': self.cache.get_memory_usage_mb()
        }