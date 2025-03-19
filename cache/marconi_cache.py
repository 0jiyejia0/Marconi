import time
import torch
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from .base_cache import BaseCacheManager

@dataclass
class CacheNode:
    prefix_id: str
    tokens: List[int]
    length: int
    
    kv_caches: List[Tuple[torch.Tensor, torch.Tensor]]
    ssm_states: List[torch.Tensor]
    
    parent: Optional[str] = None
    children: Set[str] = field(default_factory=set)
    
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 0
    
    is_input_prefix: bool = False
    is_dialogue_end: bool = False
    is_partial_dialogue: bool = False
    
    flops_saved: float = 0.0
    memory_consumption: float = 0.0
    utility_score: float = 0.0
    
    def __post_init__(self):
        self._calculate_memory_consumption()
        self.access_count = 1
    
    def _calculate_memory_consumption(self):
        kv_memory = sum(k.numel() * k.element_size() + v.numel() * v.element_size() 
                      for k, v in self.kv_caches)
        
        ssm_memory = sum(state.numel() * state.element_size() for state in self.ssm_states)
        
        self.memory_consumption = kv_memory + ssm_memory
        
    def update_access(self):
        self.last_access_time = time.time()
        self.access_count += 1
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def is_branch_point(self) -> bool:
        return len(self.children) > 1
    
    def has_single_child(self) -> bool:
        return len(self.children) == 1
    
    def get_protection_level(self) -> int:
        """
        Get protection level for eviction decisions
        Higher value = more protected
        
        Level 4: Branch points with dialogue continuation (highest protection)
        Level 3: Branch points or dialogue continuations
        Level 2: Pure input prefixes or dialogue ends
        Level 1: Normal nodes with at least one child
        Level 0: Leaf nodes with no special status (lowest protection)
        """
        if self.is_branch_point() and self.is_partial_dialogue:
            return 4
        elif self.is_branch_point() or self.is_partial_dialogue:
            return 3
        elif self.is_input_prefix or self.is_dialogue_end:
            return 2
        elif len(self.children) > 0:
            return 1
        else:
            return 0


class MarconiCache:
    def __init__(
        self, 
        max_memory: int = 1024 * 1024 * 1024,  # Default 1GB
        alpha: float = 0.5,  # Weight for FLOP efficiency in utility score
        flops_attention_scale: float = 2.0,  # FLOP scaling factor for attention (O(L²))
        flops_ssm_scale: float = 1.0,  # FLOP scaling factor for SSM (O(L))
        flops_mlp_scale: float = 1.0,  # FLOP scaling factor for MLP (O(L))
        max_branch_factor: int = 3,   # Max number of branches before considering over-branched
        checkpoint_every_n: int = 32, # Checkpoint SSM states every N tokens for chunked state passing
        speculative_chunk_size: int = 64,  # Size of chunks for speculative insertion
        model_config: Dict = None,    # Model configuration
        request_tracking_window: int = 100,  # Number of requests to track for alpha tuning
        dialogue_token_ids: List[int] = None,  # Special tokens that often mark dialogue boundaries
    ):
        self.max_memory = max_memory
        self.current_memory = 0
        self.alpha = alpha
        self.alpha_min = 0.1
        self.alpha_max = 0.9
        
        self.flops_attention_scale = flops_attention_scale
        self.flops_ssm_scale = flops_ssm_scale
        self.flops_mlp_scale = flops_mlp_scale
        
        self.model_config = model_config or {
            "num_attention_layers": 4,
            "num_ssm_layers": 4,
            "hidden_dim": 512,
            "head_dim": 64,
            "num_heads": 8,
        }
        
        self.nodes: Dict[str, CacheNode] = {}
        self.root_prefixes: Set[str] = set()
        
        self.max_branch_factor = max_branch_factor
        self.checkpoint_every_n = checkpoint_every_n
        self.speculative_chunk_size = speculative_chunk_size
        self.overbranched_nodes: Set[str] = set()
        
        self.dialogue_token_ids = dialogue_token_ids or [0, 1, 2]
        self.recent_dialogue_ends: List[str] = []
        self.max_recent_dialogues = 5
        
        self.earliest_access_time = float('inf')
        self.latest_access_time = 0.0
        
        self.hit_count = 0
        self.miss_count = 0
        
        self.request_tracking_window = request_tracking_window
        self.request_history: List[Dict] = []
        self.last_alpha_tuning_time = 0
        self.alpha_tuning_interval = 1000
        
        self.is_generation_phase = False
        
        self.prev_request_tokens = None
    
    def _generate_prefix_id(self, tokens: List[int]) -> str:
        return hash(tuple(tokens)).__str__()
    
    def _calculate_flops_saved(self, length: int, parent_length: int = 0) -> float:
        """
        Calculate estimated FLOPs saved by caching this prefix
        
        Args:
            length: Length of current prefix
            parent_length: Length of parent prefix (for incremental FLOP calculation)
            
        Returns:
            Estimated FLOPs saved
        """
        increment_length = length - parent_length
        if increment_length <= 0:
            return 0.0
        
        num_attention_layers = self.model_config["num_attention_layers"]
        num_ssm_layers = self.model_config["num_ssm_layers"]
        hidden_dim = self.model_config["hidden_dim"]
        head_dim = self.model_config["head_dim"]
        num_heads = self.model_config["num_heads"]
        
        attention_flops = 0
        for _ in range(num_attention_layers):
            qk_flops = increment_length * (parent_length + increment_length/2) * head_dim * num_heads
            av_flops = increment_length * (parent_length + increment_length/2) * head_dim * num_heads
            attention_flops += (qk_flops + av_flops) * self.flops_attention_scale
        
        ssm_flops = 0
        for _ in range(num_ssm_layers):
            ssm_flops += increment_length * hidden_dim * self.flops_ssm_scale
        
        mlp_flops = 0
        for _ in range(num_attention_layers + num_ssm_layers):
            mlp_flops += increment_length * hidden_dim * 4 * hidden_dim * 2 * self.flops_mlp_scale
        
        total_flops = attention_flops + ssm_flops + mlp_flops
        return total_flops
    
    def _calculate_recency(self, node: CacheNode) -> float:
        time_range = self.latest_access_time - self.earliest_access_time
        if time_range <= 0:
            return 1.0
        
        return (node.last_access_time - self.earliest_access_time) / time_range
    
    def _calculate_frequency(self, node: CacheNode) -> float:
        max_count = max([n.access_count for n in self.nodes.values()]) if self.nodes else 1
        return node.access_count / max_count
    
    def _calculate_flop_efficiency(self, node: CacheNode) -> float:
        if node.memory_consumption <= 0:
            return 0.0
        
        return node.flops_saved / node.memory_consumption
    
    def _update_utility_score(self, node: CacheNode):
        recency = self._calculate_recency(node)
        frequency = self._calculate_frequency(node)
        flop_efficiency = self._calculate_flop_efficiency(node)
        
        base_score = 0.4 * recency + 0.6 * frequency + self.alpha * flop_efficiency
        
        protection_level = node.get_protection_level()
        protection_multiplier = 1.0 + (protection_level * 0.5)
        
        node.utility_score = base_score * protection_multiplier
    
    def _update_all_utility_scores(self):
        access_times = [node.last_access_time for node in self.nodes.values() if node.access_count > 0]
        if access_times:
            self.earliest_access_time = min(access_times)
            self.latest_access_time = max(access_times)
        
        for node in self.nodes.values():
            self._update_utility_score(node)
    
    def find_best_prefix_match(self, tokens: List[int]) -> Optional[Tuple[str, int]]:
        """
        Find the best prefix match for the given tokens
        
        Args:
            tokens: Input token sequence
            
        Returns:
            (prefix_id, match_length): Matched prefix ID and length, or None if no match
        """
        best_match_id = None
        best_match_length = 0
        
        for prefix_id in self.root_prefixes:
            current_id = prefix_id
            
            while current_id is not None:
                node = self.nodes[current_id]
                node_tokens = node.tokens
                
                if (len(node_tokens) <= len(tokens) and 
                    all(a == b for a, b in zip(node_tokens, tokens))):
                    
                    if len(node_tokens) > best_match_length:
                        best_match_id = current_id
                        best_match_length = len(node_tokens)
                    
                    next_id = None
                    for child_id in node.children:
                        child_node = self.nodes[child_id]
                        child_tokens = child_node.tokens
                        
                        if (len(child_tokens) <= len(tokens) and 
                            all(a == b for a, b in zip(child_tokens, tokens))):
                            next_id = child_id
                            break
                    
                    current_id = next_id
                else:
                    break
        
        if best_match_id is not None:
            return (best_match_id, best_match_length)
        
        return None
    
    def is_dialogue_continuation(self, tokens: List[int]) -> bool:
        if not self.prev_request_tokens:
            return False
            
        if len(self.prev_request_tokens) >= len(tokens):
            return False
            
        if not all(a == b for a, b in zip(self.prev_request_tokens, tokens[:len(self.prev_request_tokens)])):
            return False
            
        new_portion = tokens[len(self.prev_request_tokens):]
        for token in new_portion:
            if token in self.dialogue_token_ids:
                return True
                
        for dialogue_end_id in self.recent_dialogue_ends:
            if dialogue_end_id in self.nodes:
                dialogue_tokens = self.nodes[dialogue_end_id].tokens
                if len(dialogue_tokens) < len(tokens) and all(a == b for a, b in zip(dialogue_tokens, tokens[:len(dialogue_tokens)])):
                    return True
                    
        return len(new_portion) < len(tokens) // 3
    
    def detect_potential_branch_points(self, tokens: List[int]) -> List[int]:
        potential_points = []
        
        for i, token in enumerate(tokens):
            if token in self.dialogue_token_ids:
                potential_points.append(i)
                
        match = self.find_best_prefix_match(tokens)
        if match:
            prefix_id, match_length = match
            node = self.nodes[prefix_id]
            
            if node.is_branch_point():
                potential_points.append(match_length - 1)
                
            if match_length < len(tokens) and node.children:
                next_token = tokens[match_length]
                would_create_branch = True
                
                for child_id in node.children:
                    child_tokens = self.nodes[child_id].tokens
                    if len(child_tokens) > match_length and child_tokens[match_length] == next_token:
                        would_create_branch = False
                        break
                
                if would_create_branch:
                    potential_points.append(match_length)
        
        for i in range(0, len(tokens), self.checkpoint_every_n):
            if i not in potential_points:
                potential_points.append(i)
                
        return sorted(set(potential_points))
    
    def _should_cache_prefix(self, tokens: List[int], is_input_prefix: bool = False, 
                           is_dialogue_end: bool = False, is_partial_dialogue: bool = False) -> bool:
        """
        Implement speculative insertion logic to decide whether to cache a prefix
        
        Args:
            tokens: Token sequence to potentially cache
            is_input_prefix: Whether this is a pure input prefix (no generation)
            is_dialogue_end: Whether this is the end of a dialogue turn
            is_partial_dialogue: Whether this is part output + part input (dialogue continuation)
            
        Returns:
            True if the prefix should be cached, False otherwise
        """
        if is_input_prefix or is_dialogue_end or is_partial_dialogue:
            return True
            
        parent_match = self.find_best_prefix_match(tokens[:-1]) if len(tokens) > 1 else None
        if parent_match:
            parent_id, _ = parent_match
            parent_node = self.nodes[parent_id]
            
            for child_id in parent_node.children:
                child_tokens = self.nodes[child_id].tokens
                if child_tokens[-1] != tokens[-1]:
                    return True
            
            if parent_id in self.overbranched_nodes:
                return False
        
        if self.is_generation_phase:
            for token in tokens[-1:]:
                if token in self.dialogue_token_ids:
                    return True
            
            if len(tokens) % self.checkpoint_every_n == 0:
                return True
                
            token_len_factor = min(len(tokens) / 100, 1.0)
            cache_probability = 0.1 + (0.4 * token_len_factor)
            
            return random.random() < cache_probability
        
        return random.random() < 0.75
    
    def _check_branch_factor(self, parent_id: str):
        if parent_id and parent_id in self.nodes:
            parent_node = self.nodes[parent_id]
            if len(parent_node.children) > self.max_branch_factor:
                self.overbranched_nodes.add(parent_id)
    
    def admit(self, 
              tokens: List[int], 
              kv_caches: List[Tuple[torch.Tensor, torch.Tensor]], 
              ssm_states: List[torch.Tensor],
              is_input_prefix: bool = False,
              is_dialogue_end: bool = False,
              is_partial_dialogue: bool = False,
              force_admit: bool = False) -> Optional[str]:
        prefix_id = self._generate_prefix_id(tokens)
        
        if prefix_id in self.nodes:
            node = self.nodes[prefix_id]
            node.update_access()
            
            if is_input_prefix:
                node.is_input_prefix = True
            if is_dialogue_end:
                node.is_dialogue_end = True
                if prefix_id not in self.recent_dialogue_ends:
                    self.recent_dialogue_ends.append(prefix_id)
                    if len(self.recent_dialogue_ends) > self.max_recent_dialogues:
                        self.recent_dialogue_ends.pop(0)
            if is_partial_dialogue:
                node.is_partial_dialogue = True
                
            self.latest_access_time = max(self.latest_access_time, node.last_access_time)
            
            self._track_request(tokens, hit=True)
            
            return prefix_id
        
        if not force_admit and not self._should_cache_prefix(tokens, is_input_prefix, 
                                                         is_dialogue_end, is_partial_dialogue):
            return None
        
        best_parent_match = self.find_best_prefix_match(tokens[:-1]) if len(tokens) > 1 else None
        parent_id = best_parent_match[0] if best_parent_match else None
        parent_length = best_parent_match[1] if best_parent_match else 0
        
        flops_saved = self._calculate_flops_saved(len(tokens), parent_length)
        
        new_node = CacheNode(
            prefix_id=prefix_id,
            tokens=tokens.copy(),
            length=len(tokens),
            kv_caches=kv_caches,
            ssm_states=ssm_states,
            parent=parent_id,
            flops_saved=flops_saved,
            is_input_prefix=is_input_prefix,
            is_dialogue_end=is_dialogue_end,
            is_partial_dialogue=is_partial_dialogue
        )
        
        if self.current_memory + new_node.memory_consumption > self.max_memory:
            freed_memory = self.evict(new_node.memory_consumption)
            
            if freed_memory < new_node.memory_consumption:
                return None
        
        self.nodes[prefix_id] = new_node
        self.current_memory += new_node.memory_consumption
        
        if parent_id:
            parent_node = self.nodes[parent_id]
            parent_node.children.add(prefix_id)
            
            self._check_branch_factor(parent_id)
        else:
            self.root_prefixes.add(prefix_id)
        
        self.earliest_access_time = min(self.earliest_access_time, new_node.last_access_time)
        self.latest_access_time = max(self.latest_access_time, new_node.last_access_time)
        
        if is_dialogue_end:
            if prefix_id not in self.recent_dialogue_ends:
                self.recent_dialogue_ends.append(prefix_id)
                if len(self.recent_dialogue_ends) > self.max_recent_dialogues:
                    self.recent_dialogue_ends.pop(0)
        
        self._update_all_utility_scores()
        
        self._track_request(tokens, hit=False)
        
        self._maybe_tune_alpha()
        
        return prefix_id
    
    def evict(self, required_memory: int) -> int:
        self._update_all_utility_scores()
        
        nodes_by_protection = {
            0: [],  # Leaf nodes with no special status
            1: [],  # Normal nodes with at least one child
            2: [],  # Input prefixes or dialogue ends
            3: [],  # Branch points or dialogue continuations
            4: []   # Branch points with dialogue continuation (highest protection)
        }
        
        for node in self.nodes.values():
            protection_level = node.get_protection_level()
            nodes_by_protection[protection_level].append(node)
        
        for level in nodes_by_protection:
            nodes_by_protection[level].sort(key=lambda node: node.utility_score)
        
        freed_memory = 0
        nodes_to_evict = []
        
        for level in range(5):
            for node in nodes_by_protection[level]:
                if freed_memory >= required_memory:
                    break
                
                nodes_to_evict.append(node.prefix_id)
                freed_memory += node.memory_consumption
            
            if freed_memory >= required_memory:
                break
        
        for prefix_id in nodes_to_evict:
            if prefix_id in self.recent_dialogue_ends:
                self.recent_dialogue_ends.remove(prefix_id)
                
            self._remove_node(prefix_id)
        
        return freed_memory
    
    def _remove_node(self, prefix_id: str):
        if prefix_id not in self.nodes:
            return
        
        node = self.nodes[prefix_id]
        
        self.overbranched_nodes.discard(prefix_id)
        
        if node.parent and node.parent in self.nodes:
            parent_node = self.nodes[node.parent]
            parent_node.children.remove(prefix_id)
        else:
            self.root_prefixes.discard(prefix_id)
        
        for child_id in list(node.children):
            child_node = self.nodes[child_id]
            
            child_node.parent = node.parent
            
            if node.parent and node.parent in self.nodes:
                parent_node = self.nodes[node.parent]
                parent_node.children.add(child_id)
            else:
                self.root_prefixes.add(child_id)
        
        self.current_memory -= node.memory_consumption
        
        del self.nodes[prefix_id]
    
    def lookup(self, tokens: List[int]) -> Optional[Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor]]]:
        is_dialogue_continuation = self.is_dialogue_continuation(tokens)
        
        match = self.find_best_prefix_match(tokens)
        if not match:
            self.miss_count += 1
            
            self._track_request(tokens, hit=False, is_dialogue_continuation=is_dialogue_continuation)
            
            self.prev_request_tokens = tokens.copy()
            
            return None
        
        prefix_id, match_length = match
        node = self.nodes[prefix_id]
        
        node.update_access()
        self.earliest_access_time = min(self.earliest_access_time, node.last_access_time)
        self.latest_access_time = max(self.latest_access_time, node.last_access_time)
        
        if is_dialogue_continuation and not node.is_partial_dialogue:
            node.is_partial_dialogue = True
        
        self.hit_count += 1
        
        self._track_request(tokens, hit=True, is_dialogue_continuation=is_dialogue_continuation)
        
        self.prev_request_tokens = tokens.copy()
        
        self._maybe_tune_alpha()
        
        return (node.kv_caches, node.ssm_states)
    
    def _track_request(self, tokens: List[int], hit: bool, is_dialogue_continuation: bool = False):
        self.request_history.append({
            'tokens': tokens.copy(),
            'hit': hit,
            'time': time.time(),
            'token_length': len(tokens),
            'is_dialogue_continuation': is_dialogue_continuation
        })
        
        if len(self.request_history) > self.request_tracking_window:
            self.request_history.pop(0)
    
    def _maybe_tune_alpha(self):
        current_time = time.time()
        if current_time - self.last_alpha_tuning_time < self.alpha_tuning_interval:
            return
            
        if len(self.request_history) < min(20, self.request_tracking_window // 2):
            return
            
        self.last_alpha_tuning_time = current_time
        self._tune_alpha()
    
    def _tune_alpha(self):
        if not self.request_history:
            return
            
        current_hits = sum(1 for req in self.request_history if req['hit'])
        current_hit_rate = current_hits / len(self.request_history)
        
        dialogue_requests = [req for req in self.request_history if req['is_dialogue_continuation']]
        if dialogue_requests:
            dialogue_hits = sum(1 for req in dialogue_requests if req['hit'])
            dialogue_hit_rate = dialogue_hits / len(dialogue_requests)
        else:
            dialogue_hit_rate = 0.0
        
        # If hit rate is very low for dialogue continuations, prioritize them
        if dialogue_requests and dialogue_hit_rate < 0.3:
            # Hit rate too low for dialogue continuations - adjust alpha to retain more
            new_alpha = max(self.alpha - 0.05, self.alpha_min)
        elif current_hit_rate > 0.9:
            # Hit rate too high overall - might be wasting memory on easy cases
            # Increase alpha to prioritize FLOP efficiency more
            new_alpha = min(self.alpha + 0.05, self.alpha_max)
        elif current_hit_rate < 0.2:
            # Hit rate too low overall - might be evicting useful prefixes
            # Decrease alpha to prioritize recency/frequency more
            new_alpha = max(self.alpha - 0.05, self.alpha_min)
        else:
            # Try small adjustments and see if they help
            # This is a simplified version of what would be a more complex
            # reinforcement learning or Bayesian optimization approach
            if random.random() < 0.5:
                adjustment = 0.02
            else:
                adjustment = -0.02
                
            new_alpha = max(min(self.alpha + adjustment, self.alpha_max), self.alpha_min)
        
        # Only update if there's a meaningful change
        if abs(new_alpha - self.alpha) > 0.01:
            self.alpha = new_alpha
            self._update_all_utility_scores()
    
    def set_generation_phase(self, is_generation: bool):
        self.is_generation_phase = is_generation
    
    def get_hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def adjust_alpha(self, new_alpha: float):
        self.alpha = max(min(new_alpha, self.alpha_max), self.alpha_min)
        self._update_all_utility_scores()
    
    def clear(self):
        self.nodes.clear()
        self.root_prefixes.clear()
        self.overbranched_nodes.clear()
        self.current_memory = 0
        self.hit_count = 0
        self.miss_count = 0
        self.earliest_access_time = float('inf')
        self.latest_access_time = 0.0
        self.request_history.clear()
        self.recent_dialogue_ends.clear()
        self.prev_request_tokens = None
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "current_memory_mb": self.current_memory / (1024 * 1024),
            "max_memory_mb": self.max_memory / (1024 * 1024),
            "memory_usage_percentage": (self.current_memory / self.max_memory) * 100 if self.max_memory > 0 else 0,
            "num_nodes": len(self.nodes),
            "num_root_prefixes": len(self.root_prefixes),
            "num_branch_points": sum(1 for node in self.nodes.values() if node.is_branch_point()),
            "num_overbranched": len(self.overbranched_nodes),
            "num_dialogue_ends": sum(1 for node in self.nodes.values() if node.is_dialogue_end),
            "num_partial_dialogues": sum(1 for node in self.nodes.values() if node.is_partial_dialogue),
            "hit_rate": self.get_hit_rate(),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "alpha": self.alpha,
        }


class MarconiCacheManager(BaseCacheManager):
    def __init__(
        self, 
        model,
        max_memory: int = 1024 * 1024 * 1024,  # Default 1GB
        alpha: float = 0.5,  # Weight for FLOP efficiency
        max_branch_factor: int = 3,   # Max branches per node
        checkpoint_every_n: int = 32, # Checkpoint SSM states every N tokens
        speculative_chunk_size: int = 64,  # Size of chunks for speculative insertion
        dialogue_token_ids: List[int] = None,  # Special tokens marking dialogue boundaries
    ):
        super().__init__(model)
        
        model_config = {
            "num_attention_layers": 0,
            "num_ssm_layers": 0,
            "hidden_dim": model.hidden_dim,
            "head_dim": model.hidden_dim // model.blocks[0].attention.num_heads,
            "num_heads": model.blocks[0].attention.num_heads,
        }
        
        for block in model.blocks:
            if hasattr(block, 'attention'):
                model_config["num_attention_layers"] += 1
            if hasattr(block, 'ssm_layer'):
                model_config["num_ssm_layers"] += 1
        
        self.cache = MarconiCache(
            max_memory=max_memory,
            alpha=alpha,
            model_config=model_config,
            max_branch_factor=max_branch_factor,
            checkpoint_every_n=checkpoint_every_n,
            speculative_chunk_size=speculative_chunk_size,
            dialogue_token_ids=dialogue_token_ids
        )
        
        self.last_computed_tokens = None
        self.last_computed_kv_caches = None
        self.last_computed_ssm_states = None
    
    def generate(self, input_ids, max_length=50, temperature=1.0, top_k=50):
        start_time = time.time()
        device = input_ids.device
        
        generated_ids = input_ids.clone()
        
        self.cache.set_generation_phase(False)
        
        initial_tokens = input_ids[0].cpu().tolist()
        
        is_dialogue_continuation = self.cache.is_dialogue_continuation(initial_tokens)
        
        cached_data = self.cache.lookup(initial_tokens)
        
        if cached_data:
            kv_caches, ssm_states = cached_data
            match_length = len(initial_tokens)
            
            if is_dialogue_continuation:
                match = self.cache.find_best_prefix_match(initial_tokens)
                if match:
                    prefix_id, _ = match
                    if prefix_id in self.cache.nodes:
                        self.cache.nodes[prefix_id].is_partial_dialogue = True
        else:
            potential_branch_points = self.cache.detect_potential_branch_points(initial_tokens)
            
            if not potential_branch_points and not is_dialogue_continuation:
                with torch.no_grad():
                    _, ssm_states, kv_caches = self.model(
                        input_ids, 
                        return_key_values=True
                    )
                
                self.cache.admit(initial_tokens, kv_caches, ssm_states, is_input_prefix=True)
            else:
                best_match = self.cache.find_best_prefix_match(initial_tokens)
                if best_match:
                    prefix_id, match_length = best_match
                    node = self.cache.nodes[prefix_id]
                    start_kv_caches = node.kv_caches
                    start_ssm_states = node.ssm_states
                    
                    remaining_ids = input_ids[:, match_length:]
                    
                    with torch.no_grad():
                        _, ssm_states, kv_caches = self.model(
                            remaining_ids, 
                            ssm_states=start_ssm_states,
                            past_key_values=start_kv_caches,
                            return_key_values=True
                        )
                else:
                    match_length = 0
                    
                    chunk_size = self.cache.speculative_chunk_size
                    
                    first_chunk_size = min(chunk_size, len(initial_tokens))
                    first_chunk_ids = input_ids[:, :first_chunk_size]
                    
                    with torch.no_grad():
                        _, first_ssm_states, first_kv_caches = self.model(
                            first_chunk_ids, 
                            return_key_values=True
                        )
                    
                    first_chunk_tokens = initial_tokens[:first_chunk_size]
                    self.cache.admit(first_chunk_tokens, first_kv_caches, first_ssm_states, 
                                    is_input_prefix=True)
                    
                    if first_chunk_size < len(initial_tokens):
                        remaining_ids = input_ids[:, first_chunk_size:]
                        
                        with torch.no_grad():
                            _, ssm_states, kv_caches = self.model(
                                remaining_ids, 
                                ssm_states=first_ssm_states,
                                past_key_values=first_kv_caches,
                                return_key_values=True
                            )
                    else:
                        ssm_states = first_ssm_states
                        kv_caches = first_kv_caches
                
                self.cache.admit(initial_tokens, kv_caches, ssm_states, 
                               is_input_prefix=True,
                               is_partial_dialogue=is_dialogue_continuation)
        
        self.last_computed_tokens = initial_tokens.copy()
        self.last_computed_kv_caches = kv_caches
        self.last_computed_ssm_states = ssm_states
        
        self.cache.set_generation_phase(True)
        
        for i in range(max_length):
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
            
            current_tokens = generated_ids[0].cpu().tolist()
            
            should_checkpoint = False
            
            if i % self.cache.checkpoint_every_n == 0:
                should_checkpoint = True
            
            if i == max_length - 1:
                should_checkpoint = True
                
            if next_token.item() in self.cache.dialogue_token_ids:
                should_checkpoint = True
                
            if should_checkpoint:
                is_dialogue_end = (i == max_length - 1)
                
                self.cache.admit(
                    current_tokens, 
                    kv_caches, 
                    ssm_states,
                    is_input_prefix=False,
                    is_dialogue_end=is_dialogue_end
                )
        
        total_time = time.time() - start_time
        
        stats = self.cache.get_stats()
        
        return {
            'output': generated_ids,
            'time': total_time,
            'hit_rate': stats['hit_rate'],
            'memory_usage': stats['current_memory_mb']
        }