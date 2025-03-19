import torch
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from functools import lru_cache

from utils import clear_memory, ensure_dir
from cache import MarconiCacheManager, SimpleKVCacheManager, NoCacheManager
from .visualization import ResultsVisualizer

class TestbenchRunner:
    
    def __init__(self, model, vocab_size=10000, device='cuda' if torch.cuda.is_available() else 'cpu',
                 max_marconi_memory=1024*1024*512, simple_cache_entries=100, 
                 results_dir='cache_test_results', use_parallel=True):
        self.model = model.to(device)
        self.device = device
        self.vocab_size = vocab_size
        self.use_parallel = use_parallel
        
        self.cache_managers = {
            'marconi': MarconiCacheManager(model, max_memory=max_marconi_memory),
            'simple': SimpleKVCacheManager(model, max_entries=simple_cache_entries),
            'no_cache': NoCacheManager(model)
        }
        
        self.performance_data = {cache_type: {metric: [] for metric in 
                               ['time', 'hit_rate', 'memory', 'speedup']}
                              for cache_type in ['marconi', 'simple', 'no_cache']}
        
        self.results_dir = results_dir
        ensure_dir(self.results_dir)
        
        self.visualizer = ResultsVisualizer(self.results_dir)
        
        self._prompt_cache = {}
    
    @lru_cache(maxsize=100)
    def _generate_random_prompt(self, prompt_length=10, batch_size=1, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            
        return torch.randint(0, self.vocab_size, (batch_size, prompt_length), device=self.device)
    
    def _generate_similar_prompts(self, base_prompt, num_prompts=10, similarity=0.8, seed=None):
        cache_key = (base_prompt.shape[1], num_prompts, similarity, seed)
        
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]
        
        if seed is not None:
            torch.manual_seed(seed)
            
        prompts = []
        base_length = base_prompt.shape[1]
        
        for _ in range(num_prompts):
            prompt = base_prompt.clone()
            
            change_positions = torch.rand(base_length) > similarity
            num_changes = change_positions.sum().item()
            
            if num_changes > 0:
                new_tokens = torch.randint(0, self.vocab_size, (1, num_changes), device=self.device)
                prompt[0, change_positions] = new_tokens
            
            prompts.append(prompt)
        
        self._prompt_cache[cache_key] = prompts
        return prompts
    
    def run_single_test(self, prompt, gen_length=30, clear_caches=True):
        results = {}
        
        if clear_caches:
            self.cache_managers['marconi'].cache.clear()
            self.cache_managers['simple'].cache.clear()
        
        for cache_type, manager in self.cache_managers.items():
            try:
                clear_memory()
                
                result = manager.generate(
                    prompt.clone(), max_length=gen_length
                )
                
                time_taken = result['time']
                memory_usage = result.get('memory_usage', 0)
                hit_rate = result.get('hit_rate', 0)
                
                results[cache_type] = {
                    'time': time_taken,
                    'memory': memory_usage,
                    'hit_rate': hit_rate
                }
                
            except Exception as e:
                print(f"Error testing {cache_type}: {str(e)}")
                traceback.print_exc()
                results[cache_type] = {
                    'time': float('inf'),
                    'memory': 0,
                    'hit_rate': 0,
                    'error': str(e)
                }
        
        if 'no_cache' in results and results['no_cache']['time'] != float('inf'):
            no_cache_time = results['no_cache']['time']
            for cache_type in ['marconi', 'simple']:
                if cache_type in results and 'time' in results[cache_type]:
                    results[cache_type]['speedup'] = no_cache_time / results[cache_type]['time']
        
        for cache_type, metrics in results.items():
            for metric, value in metrics.items():
                if metric in self.performance_data[cache_type]:
                    self.performance_data[cache_type][metric].append(value)
        
        self._print_test_results(results)
        
        return results
    
    def _print_test_results(self, results):
        print(f"\n测试结果:")
        
        if 'marconi' in results:
            r = results['marconi']
            print(f"  Marconi缓存: {r['time']:.3f}秒", end="")
            if 'hit_rate' in r:
                print(f", 命中率: {r['hit_rate']:.2f}", end="")
            if 'memory' in r:
                print(f", 内存: {r['memory']:.1f}MB", end="")
            if 'speedup' in r:
                print(f", 加速比: {r['speedup']:.2f}x", end="")
            print()
            
        if 'simple' in results:
            r = results['simple']
            print(f"  简单缓存: {r['time']:.3f}秒", end="")
            if 'hit_rate' in r:
                print(f", 命中率: {r['hit_rate']:.2f}", end="")
            if 'memory' in r:
                print(f", 内存: {r['memory']:.1f}MB", end="")
            if 'speedup' in r:
                print(f", 加速比: {r['speedup']:.2f}x", end="")
            print()
            
        if 'no_cache' in results:
            r = results['no_cache']
            print(f"  无缓存: {r['time']:.3f}秒")
    
    def _run_test_worker(self, prompt, gen_length, clear_caches, idx=None, total=None):
        if idx is not None and total is not None:
            print(f"\n提示 {idx+1}/{total}")
        result = self.run_single_test(prompt, gen_length, clear_caches)
        return result
    
    def run_similarity_test(self, num_prompts=10, prompt_length=10, similarity=0.8, gen_length=30, seed=None):
        print(f"\n=== 运行相似度测试 (相似度: {similarity}) ===")
        
        base_prompt = self._generate_random_prompt(prompt_length, seed=seed)
        
        prompts = self._generate_similar_prompts(base_prompt, num_prompts, similarity, seed=seed)
        
        results = []
        
        if self.use_parallel and num_prompts > 1:
            # 在并行模式下收集测试数据，但不在工作线程中绘图
            with ThreadPoolExecutor() as executor:
                future_to_idx = {}
                for i, prompt in enumerate(prompts):
                    future = executor.submit(
                        self._run_test_worker, 
                        prompt, 
                        gen_length, 
                        i==0,  # 只在第一次测试前清除缓存
                        i, 
                        num_prompts
                    )
                    future_to_idx[future] = i
                
                for future in tqdm(as_completed(future_to_idx), total=len(prompts), desc="测试相似提示"):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"提示 {idx+1} 失败: {str(e)}")
                        traceback.print_exc()
        else:
            for i, prompt in enumerate(tqdm(prompts, desc="测试相似提示")):
                print(f"\n提示 {i+1}/{num_prompts}")
                result = self.run_single_test(prompt, gen_length, i==0)
                results.append(result)
        
        return results
    
    def run_length_test(self, min_length=5, max_length=50, step=5, gen_length=30, seed=None):
        print(f"\n=== 运行长度测试 (长度: {min_length}-{max_length}) ===")
        
        lengths = list(range(min_length, max_length + 1, step))
        results = []
        
        if self.use_parallel and len(lengths) > 1:
            with ThreadPoolExecutor() as executor:
                future_to_length = {}
                for length in lengths:
                    prompt = self._generate_random_prompt(length, seed=seed if seed is None else seed+length)
                    future = executor.submit(self._run_test_worker, prompt, gen_length, True)
                    future_to_length[future] = length
                
                for future in tqdm(as_completed(future_to_length), total=len(lengths), desc="测试不同长度"):
                    length = future_to_length[future]
                    try:
                        result = future.result()
                        print(f"\n提示长度: {length}")
                        result['prompt_length'] = length
                        results.append(result)
                    except Exception as e:
                        print(f"长度 {length} 失败: {str(e)}")
                        traceback.print_exc()
        else:
            for length in tqdm(lengths, desc="测试不同长度"):
                prompt = self._generate_random_prompt(length, seed=seed if seed is None else seed+length)
                print(f"\n提示长度: {length}")
                result = self.run_single_test(prompt, gen_length)
                result['prompt_length'] = length
                results.append(result)
        
        return results
    
    def run_reuse_test(self, num_iterations=5, prompt_length=10, gen_length=30, seed=None):

        print(f"\n=== 运行重复使用测试 ({num_iterations}次) ===")
        
        prompt = self._generate_random_prompt(prompt_length, seed=seed)
        
        results = []
        
        for manager in self.cache_managers.values():
            if hasattr(manager, 'cache') and hasattr(manager.cache, 'clear'):
                manager.cache.clear()
        
        for i in range(num_iterations):
            print(f"\n迭代 {i+1}/{num_iterations}")
            result = self.run_single_test(prompt.clone(), gen_length, clear_caches=False)
            result['iteration'] = i + 1
            results.append(result)
        
        self.reuse_test_results = results
        
        return results
    
    def _run_dialogue_turn(self, cache_type, history, turn_length, gen_length, user_input=None):
        manager = self.cache_managers[cache_type]
        
        clear_memory()
        
        try:
            result = manager.generate(
                history.clone(), max_length=gen_length
            )
            
            if user_input is None:
                user_input = self._generate_random_prompt(turn_length)
                
            updated_history = torch.cat([result['output'], user_input], dim=1)
            
            if 'hit_rate' not in result and cache_type != 'no_cache':
                result['hit_rate'] = 0.0
                
            if 'memory_usage' not in result and cache_type != 'no_cache':
                result['memory_usage'] = 0.0
                
            return result, updated_history
            
        except Exception as e:
            print(f"对话轮次出错 ({cache_type}): {str(e)}")
            import traceback
            traceback.print_exc()
            
            default_result = {
                'time': 0.0,
                'output': history,  # 保持原样
            }
            
            if cache_type != 'no_cache':
                default_result['hit_rate'] = 0.0
                default_result['memory_usage'] = 0.0
                
            return default_result, history  # 保持历史不变
        
    def test_multiturn_dialogue(self, num_turns=5, initial_length=10, turn_length=5, gen_length=20, seed=None):
        print(f"\n=== 测试多轮对话场景 ({num_turns}轮) ===")
        
        results = {
            'turn': [],
            'marconi_time': [],
            'simple_time': [],
            'no_cache_time': [],
            'marconi_hit_rate': [],
            'simple_hit_rate': [],
            'marconi_memory': [],
            'simple_memory': [],
            'history_length': []
        }
        
        initial_prompt = self._generate_random_prompt(initial_length, seed=seed)
        histories = {
            'marconi': initial_prompt.clone(),
            'simple': initial_prompt.clone(),
            'no_cache': initial_prompt.clone()
        }
        
        for cache_type, manager in self.cache_managers.items():
            if hasattr(manager, 'cache') and hasattr(manager.cache, 'clear'):
                manager.cache.clear()
        
        user_inputs = []
        if seed is not None:
            torch.manual_seed(seed + 1000)
        for _ in range(num_turns):
            user_inputs.append(self._generate_random_prompt(turn_length))
        
        for turn in range(1, num_turns + 1):
            print(f"\n对话轮次 {turn}/{num_turns}")
            results['turn'].append(turn)
            results['history_length'].append(histories['marconi'].shape[1])
            
            user_input = user_inputs[turn-1]
            
            for cache_type in ['marconi', 'simple', 'no_cache']:
                result, histories[cache_type] = self._run_dialogue_turn(
                    cache_type, 
                    histories[cache_type], 
                    turn_length, 
                    gen_length, 
                    user_input
                )
                
                results[f'{cache_type}_time'].append(result['time'])
                
                if cache_type != 'no_cache':
                    if 'hit_rate' in result:
                        results[f'{cache_type}_hit_rate'].append(result['hit_rate'])
                    if 'memory_usage' in result:
                        results[f'{cache_type}_memory'].append(result['memory_usage'])
            
            print(f"  对话历史长度: {histories['marconi'].shape[1]} tokens")
            print(f"  Marconi缓存: {results['marconi_time'][-1]:.3f}秒, " + 
                f"命中率: {results['marconi_hit_rate'][-1]:.2f} (如果可用), " + 
                f"内存: {results['marconi_memory'][-1]:.1f}MB (如果可用)")
            print(f"  简单缓存: {results['simple_time'][-1]:.3f}秒, " + 
                f"命中率: {results['simple_hit_rate'][-1]:.2f} (如果可用), " + 
                f"内存: {results['simple_memory'][-1]:.1f}MB (如果可用)")
            print(f"  无缓存: {results['no_cache_time'][-1]:.3f}秒")
        
        self.multiturn_results = results
    
        return results
    
    def test_memory_limited_dialogue(self, memory_limits=None, max_entries=None, num_turns=5, initial_length=10, turn_length=5, gen_length=20):
        if memory_limits is None:
            memory_limits = [32, 64, 128, 256, 512]  # MB
        
        if max_entries is None:
            max_entries = [10, 20, 50, 100, 200]
        
        print(f"\n=== 测试不同内存限制下的多轮对话性能 ({num_turns}轮) ===")
        
        memory_results = {
            'memory_limit': [],  # MB
            'marconi_hit_rate': [],
            'marconi_avg_time': [],
            'marconi_final_memory': [],
            'entry_limit': [],  # entries
            'simple_hit_rate': [],
            'simple_avg_time': [],
            'simple_final_memory': []
        }
        
        for memory_limit in memory_limits:
            print(f"\n--- Marconi缓存内存限制: {memory_limit}MB ---")
            
            marconi_manager = MarconiCacheManager(
                self.model, 
                max_memory=memory_limit * 1024 * 1024  # convert to bytes
            )
            
            initial_prompt = self._generate_random_prompt(initial_length)
            marconi_history = initial_prompt.clone()
            
            marconi_times = []
            marconi_hit_rates = []
            
            for turn in range(1, num_turns + 1):
                print(f"  对话轮次 {turn}/{num_turns}")
                
                clear_memory()
                
                marconi_result = marconi_manager.generate(
                    marconi_history.clone(), max_length=gen_length
                )
                
                marconi_times.append(marconi_result['time'])
                marconi_hit_rates.append(marconi_result['hit_rate'])
                
                user_input = self._generate_random_prompt(turn_length)
                marconi_history = torch.cat([marconi_result['output'], user_input], dim=1)
            
            memory_results['memory_limit'].append(memory_limit)
            memory_results['marconi_hit_rate'].append(np.mean(marconi_hit_rates))
            memory_results['marconi_avg_time'].append(np.mean(marconi_times))
            memory_results['marconi_final_memory'].append(marconi_result['memory_usage'])
            
            print(f"  平均命中率: {np.mean(marconi_hit_rates):.4f}")
            print(f"  平均时间: {np.mean(marconi_times):.4f}秒")
            print(f"  最终内存使用: {marconi_result['memory_usage']:.2f}MB")
        
        for entries in max_entries:
            print(f"\n--- 简单缓存条目限制: {entries}条 ---")
            
            simple_manager = SimpleKVCacheManager(
                self.model, 
                max_entries=entries
            )
            
            initial_prompt = self._generate_random_prompt(initial_length)
            simple_history = initial_prompt.clone()
            
            simple_times = []
            simple_hit_rates = []
            
            for turn in range(1, num_turns + 1):
                print(f"  对话轮次 {turn}/{num_turns}")
                
                clear_memory()
                
                simple_result = simple_manager.generate(
                    simple_history.clone(), max_length=gen_length
                )
                
                simple_times.append(simple_result['time'])
                simple_hit_rates.append(simple_result['hit_rate'])
                
                user_input = self._generate_random_prompt(turn_length)
                simple_history = torch.cat([simple_result['output'], user_input], dim=1)
            
            memory_results['entry_limit'].append(entries)
            memory_results['simple_hit_rate'].append(np.mean(simple_hit_rates))
            memory_results['simple_avg_time'].append(np.mean(simple_times))
            memory_results['simple_final_memory'].append(simple_result['memory_usage'])
            
            print(f"  平均命中率: {np.mean(simple_hit_rates):.4f}")
            print(f"  平均时间: {np.mean(simple_times):.4f}秒")
            print(f"  最终内存使用: {simple_result['memory_usage']:.2f}MB")
        
        # 保存内存限制测试结果但不在这里绘图
        self.memory_limited_results = memory_results
        
        return memory_results
    
    def visualize_results(self):
        """Visualize performance results"""
        print("\n绘制结果图表...")
        
        # 在主线程中绘制所有图表
        summary = self.visualizer.plot_all(self.performance_data)
        
        # 绘制特殊测试的图表（如果有）
        if hasattr(self, 'reuse_test_results'):
            self._plot_reuse_results(self.reuse_test_results)
            
        if hasattr(self, 'multiturn_results'):
            self.visualizer.plot_multiturn_results(self.multiturn_results)
            
        if hasattr(self, 'memory_limited_results'):
            self.visualizer.plot_memory_limited_results(self.memory_limited_results)
        
        print(f"所有图表已保存到 {self.results_dir} 目录")
        return summary
    
    def _plot_reuse_results(self, results):
        """Plot results for reuse test in the main thread"""
        iterations = [r['iteration'] for r in results]
        marconi_times = [r['marconi']['time'] for r in results]
        simple_times = [r['simple']['time'] for r in results]
        no_cache_times = [r['no_cache']['time'] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, marconi_times, 'o-', label='Marconi缓存', linewidth=2)
        plt.plot(iterations, simple_times, 's-', label='简单缓存', linewidth=2)
        plt.plot(iterations, no_cache_times, '^-', label='无缓存', linewidth=2)
        plt.xlabel('迭代次数')
        plt.ylabel('生成时间 (秒)')
        plt.title('重复使用相同提示的性能')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'{self.results_dir}/reuse_test.png', dpi=300)
        plt.close()
    
    def run_all_tests(self, quick=False, seed=None):
        if quick:
            reuse_iterations = 3
            similarity_prompts = 3
            length_min, length_max, length_step = 10, 30, 10
            dialogue_turns = 3
        else:
            reuse_iterations = 5
            similarity_prompts = 10
            length_min, length_max, length_step = 5, 50, 5
            dialogue_turns = 5
        
        self.run_reuse_test(num_iterations=reuse_iterations, seed=seed)
        
        self.run_similarity_test(num_prompts=similarity_prompts, 
                               similarity=0.8, seed=seed)
        
        self.run_length_test(min_length=length_min, max_length=length_max, 
                           step=length_step, seed=seed)
        
        self.test_multiturn_dialogue(num_turns=dialogue_turns, seed=seed)
        
        return self.visualize_results()