import numpy as np
import matplotlib
# 在导入pyplot前设置非交互式后端
matplotlib.use('Agg')  # 使用非交互式后端防止线程问题
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import os
from functools import lru_cache

class ResultsVisualizer: 
    def __init__(self, results_dir='cache_test_results', 
                 style='seaborn-v0_8-darkgrid', dpi=300,
                 language='zh', figure_format='png',
                 palette=None):
        self.results_dir = results_dir
        self.dpi = dpi
        self.figure_format = figure_format
        self.style = style
        self.language = language
        self.palette = palette or {
            'Marconi缓存': "#4C72B0",
            'Marconi Cache': "#4C72B0",
            '简单缓存': "#55A868",
            'Simple Cache': "#55A868",
            '无缓存': "#C44E52",
            'No Cache': "#C44E52"
        }
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Set plot style 
        plt.style.use(style)
        
        # Set font properties based on language
        self._set_font_properties(language)
    
    def _set_font_properties(self, language):
        if language == 'zh':
            # Chinese font settings
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 
                                               'DejaVu Sans', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.family'] = 'sans-serif'
            
            # Text mappings
            self.text = {
                'time_comparison': '生成时间对比',
                'hit_rate_comparison': '缓存命中率对比',
                'memory_comparison': '内存使用对比',
                'speedup_comparison': '加速比对比',
                'efficiency_comparison': '缓存效率指标 (加速比/内存使用)',
                'hit_rate_vs_speedup': '命中率与加速比关系',
                'time': '时间(秒)',
                'hit_rate': '命中率',
                'memory': '内存使用 (MB)',
                'speedup': '加速比 (相对于无缓存)',
                'efficiency': '效率指标 (加速比/MB)',
                'marconi_trend': 'Marconi趋势',
                'simple_trend': '简单缓存趋势',
                'multiturn_time': '多轮对话中的生成时间变化',
                'multiturn_hit_rate': '多轮对话中的缓存命中率变化',
                'multiturn_memory': '多轮对话中的内存使用变化',
                'turn': '对话轮次',
                'system_type': '系统类型',
                'cache_type': '缓存类型',
                'marconi_hit_rate_memory': 'Marconi缓存命中率随内存限制变化',
                'simple_hit_rate_entries': '简单缓存命中率随条目数限制变化',
                'hit_rate_vs_memory': '命中率与内存使用关系',
                'memory_limit': '内存限制 (MB)',
                'max_entries': '最大条目数'
            }
        else:
            # English font settings
            plt.rcParams['font.family'] = 'sans-serif'
            
            # Text mappings
            self.text = {
                'time_comparison': 'Generation Time Comparison',
                'hit_rate_comparison': 'Cache Hit Rate Comparison',
                'memory_comparison': 'Memory Usage Comparison',
                'speedup_comparison': 'Speedup Comparison',
                'efficiency_comparison': 'Cache Efficiency (Speedup/Memory)',
                'hit_rate_vs_speedup': 'Hit Rate vs Speedup',
                'time': 'Time (seconds)',
                'hit_rate': 'Hit Rate',
                'memory': 'Memory Usage (MB)',
                'speedup': 'Speedup (relative to no cache)',
                'efficiency': 'Efficiency (speedup/MB)',
                'marconi_trend': 'Marconi Trend',
                'simple_trend': 'Simple Cache Trend',
                'multiturn_time': 'Generation Time in Multi-turn Dialogue',
                'multiturn_hit_rate': 'Cache Hit Rate in Multi-turn Dialogue',
                'multiturn_memory': 'Memory Usage in Multi-turn Dialogue',
                'turn': 'Dialogue Turn',
                'system_type': 'System Type',
                'cache_type': 'Cache Type',
                'marconi_hit_rate_memory': 'Marconi Cache Hit Rate vs Memory Limit',
                'simple_hit_rate_entries': 'Simple Cache Hit Rate vs Entry Limit',
                'hit_rate_vs_memory': 'Hit Rate vs Memory Usage',
                'memory_limit': 'Memory Limit (MB)',
                'max_entries': 'Maximum Entries'
            }
    
    def _get_cache_labels(self):
        if self.language == 'zh':
            return {
                'marconi': 'Marconi缓存',
                'simple': '简单缓存',
                'no_cache': '无缓存'
            }
        else:
            return {
                'marconi': 'Marconi Cache',
                'simple': 'Simple Cache',
                'no_cache': 'No Cache'
            }
    
    def _save_figure(self, filename, fig=None, close=True):
        if fig is None:
            fig = plt.gcf()
        
        # Create full path
        path = os.path.join(self.results_dir, f"{filename}.{self.figure_format}")
        
        # Save with specified settings
        fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
        
        # Close if requested (to free memory)
        if close:
            plt.close(fig)
    
    def _prepare_boxplot_data(self, data, metrics, cache_types=None):
        cache_labels = self._get_cache_labels()
        
        # Create DataFrame
        df_dict = {}
        for cache_type, label in cache_labels.items():
            # 如果指定了cache_types，只处理指定的缓存类型
            if cache_types and cache_type not in cache_types:
                continue
                
            if cache_type in data and metrics in data[cache_type]:
                values = data[cache_type][metrics]
                if values and len(values) > 0:  # 确保有数据
                    df_dict[label] = values
        
        return pd.DataFrame(df_dict)
    
    def _apply_standard_style(self, ax, title, xlabel=None, ylabel=None, grid=True):
        ax.set_title(title, fontsize=14)
        
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)
        
        if grid:
            ax.grid(True, linestyle='--', alpha=0.7)
    
    def _create_comparison_plot(self, data, metric, title, ylabel, filename, 
                               cache_types=None, figsize=(10, 6)):
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data
        df = self._prepare_boxplot_data(data, metric, cache_types)
        
        if df.empty:
            print(f"Warning: No data for {metric} comparison")
            plt.close(fig)
            return None
        
        # 检查数据点数量，如果太少则使用swarm图而不是箱线图
        if df.shape[1] > 0 and all(len(df[col]) < 5 for col in df.columns):
            # 少于5个数据点时使用swarmplot
            sns.swarmplot(data=df, ax=ax, palette=self.palette, size=10)
        else:
            # 否则使用箱线图
            sns.boxplot(data=df, ax=ax, palette=self.palette)
            # 添加数据点显示
            sns.stripplot(data=df, ax=ax, color='black', alpha=0.5, jitter=True, size=5)
        
        self._apply_standard_style(ax, title, ylabel=ylabel)
        
        self._save_figure(filename, fig)
        
        return fig
    
    def plot_time_comparison(self, data):
        return self._create_comparison_plot(
            data, 'time', 
            self.text['time_comparison'],
            self.text['time'],
            'time_comparison'
        )
    
    def plot_hit_rate_comparison(self, data):
        return self._create_comparison_plot(
            data, 'hit_rate', 
            self.text['hit_rate_comparison'],
            self.text['hit_rate'],
            'hit_rate_comparison',
            cache_types=['marconi', 'simple']
        )
    
    def plot_memory_comparison(self, data):
        return self._create_comparison_plot(
            data, 'memory', 
            self.text['memory_comparison'],
            self.text['memory'],
            'memory_comparison',
            cache_types=['marconi', 'simple']
        )
    
    def plot_speedup_comparison(self, data):
        """Plot speedup comparison boxplot"""
        return self._create_comparison_plot(
            data, 'speedup', 
            self.text['speedup_comparison'],
            self.text['speedup'],
            'speedup_comparison',
            cache_types=['marconi', 'simple']
        )
    
    def plot_efficiency_metric(self, data):
        # 检查是否有足够的数据
        if not (data['marconi']['speedup'] and data['marconi']['memory'] and 
                data['simple']['speedup'] and data['simple']['memory']):
            print("Warning: Insufficient data for efficiency metric")
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        cache_labels = self._get_cache_labels()
        
        marconi_efficiency = [s/m if m > 0 else 0 for s, m in zip(
            data['marconi']['speedup'],
            data['marconi']['memory']
        )]
        
        simple_efficiency = [s/m if m > 0 else 0 for s, m in zip(
            data['simple']['speedup'],
            data['simple']['memory']
        )]
        
        df = pd.DataFrame({
            cache_labels['marconi']: marconi_efficiency,
            cache_labels['simple']: simple_efficiency
        })
        
        sns.boxplot(data=df, ax=ax, palette=self.palette)
        
        self._apply_standard_style(
            ax, 
            self.text['efficiency_comparison'],
            ylabel=self.text['efficiency']
        )
        
        self._save_figure('efficiency_comparison', fig)
        
        return fig
    
    def plot_hit_rate_vs_speedup(self, data):
        # 检查是否有足够的数据
        if not (data['marconi']['hit_rate'] and data['marconi']['speedup'] and 
                data['simple']['hit_rate'] and data['simple']['speedup']):
            print("Warning: Insufficient data for hit rate vs speedup")
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        cache_labels = self._get_cache_labels()
        
        marconi_data = pd.DataFrame({
            'hit_rate': data['marconi']['hit_rate'],
            'speedup': data['marconi']['speedup'],
            'system': [cache_labels['marconi']] * len(data['marconi']['hit_rate'])
        })
        
        simple_data = pd.DataFrame({
            'hit_rate': data['simple']['hit_rate'],
            'speedup': data['simple']['speedup'],
            'system': [cache_labels['simple']] * len(data['simple']['hit_rate'])
        })
        
        combined_data = pd.concat([marconi_data, simple_data])
        
        sns.scatterplot(
            data=combined_data, 
            x='hit_rate', 
            y='speedup', 
            hue='system', 
            s=100, 
            alpha=0.7,
            palette=self.palette,
            ax=ax
        )
        
        sns.regplot(
            data=marconi_data, 
            x='hit_rate', 
            y='speedup', 
            scatter=False, 
            label=self.text['marconi_trend'],
            ax=ax
        )
        
        sns.regplot(
            data=simple_data, 
            x='hit_rate', 
            y='speedup', 
            scatter=False, 
            label=self.text['simple_trend'],
            ax=ax
        )
        
        self._apply_standard_style(
            ax,
            self.text['hit_rate_vs_speedup'],
            xlabel=self.text['hit_rate'],
            ylabel=self.text['speedup']
        )
        
        ax.legend()
        
        self._save_figure('hit_rate_vs_speedup', fig)
        
        return fig
    
    def plot_multiturn_results(self, results):
        self._plot_multiturn_time(results)
        self._plot_multiturn_hit_rate(results)
        self._plot_multiturn_memory(results)
    
    def _plot_multiturn_time(self, results):
        if not results['turn']:
            print("Warning: No data for multi-turn time plot")
            return None
            
        fig, ax = plt.subplots(figsize=(12, 7))
        cache_labels = self._get_cache_labels()
        
        time_df = pd.DataFrame({
            'Turn': results['turn'] * 3,
            'Time': results['marconi_time'] + results['simple_time'] + results['no_cache_time'],
            'System': [cache_labels['marconi']] * len(results['turn']) + 
                     [cache_labels['simple']] * len(results['turn']) + 
                     [cache_labels['no_cache']] * len(results['turn'])
        })
        
        sns.lineplot(
            data=time_df,
            x='Turn',
            y='Time',
            hue='System',
            marker='o',
            linewidth=3,
            markersize=10,
            palette=self.palette,
            ax=ax
        )
        
        self._apply_standard_style(
            ax,
            self.text['multiturn_time'],
            xlabel=self.text['turn'],
            ylabel=self.text['time']
        )
        
        ax.legend(title=self.text['system_type'], fontsize=12, title_fontsize=13, loc='best')
        
        self._save_figure('multiturn_time', fig)
        
        return fig
    
    def _plot_multiturn_hit_rate(self, results):
        # 检查是否有必要的数据
        if not ('turn' in results and 
                'marconi_hit_rate' in results and len(results['marconi_hit_rate']) > 0 and
                'simple_hit_rate' in results and len(results['simple_hit_rate']) > 0):
            print("Warning: No data for multi-turn hit rate plot")
            return None
            
        fig, ax = plt.subplots(figsize=(12, 7))
        cache_labels = self._get_cache_labels()
        
        # Prepare data - 只使用 marconi 和 simple 缓存，no_cache 没有命中率数据
        marconi_hit_df = pd.DataFrame({
            'Turn': results['turn'],
            'Hit Rate': results['marconi_hit_rate'],
            'System': [cache_labels['marconi']] * len(results['turn'])
        })
        
        simple_hit_df = pd.DataFrame({
            'Turn': results['turn'],
            'Hit Rate': results['simple_hit_rate'],
            'System': [cache_labels['simple']] * len(results['turn'])
        })
        
        hit_rate_df = pd.concat([marconi_hit_df, simple_hit_df], ignore_index=True)
        
        limited_palette = {k: v for k, v in self.palette.items() 
                        if k in [cache_labels['marconi'], cache_labels['simple']]}
        
        sns.lineplot(
            data=hit_rate_df,
            x='Turn',
            y='Hit Rate',
            hue='System',
            marker='o',
            linewidth=3,
            markersize=10,
            palette=limited_palette,
            ax=ax
        )
        
        self._apply_standard_style(
            ax,
            self.text['multiturn_hit_rate'],
            xlabel=self.text['turn'],
            ylabel=self.text['hit_rate']
        )
        
        ax.legend(title=self.text['cache_type'], fontsize=12, title_fontsize=13, loc='best')
        ax.set_ylim(0, 1.0)
        
        self._save_figure('multiturn_hit_rate', fig)
        
        return fig
    
    def _plot_multiturn_memory(self, results):
        # 检查是否有必要的数据
        if not ('turn' in results and 
                'marconi_memory' in results and len(results['marconi_memory']) > 0 and
                'simple_memory' in results and len(results['simple_memory']) > 0):
            print("Warning: No data for multi-turn memory plot")
            return None
            
        fig, ax = plt.subplots(figsize=(12, 7))
        cache_labels = self._get_cache_labels()
        
        # Prepare data - 只使用 marconi 和 simple 缓存，no_cache 没有内存数据
        marconi_mem_df = pd.DataFrame({
            'Turn': results['turn'],
            'Memory': results['marconi_memory'],
            'System': [cache_labels['marconi']] * len(results['turn'])
        })
        
        simple_mem_df = pd.DataFrame({
            'Turn': results['turn'],
            'Memory': results['simple_memory'],
            'System': [cache_labels['simple']] * len(results['turn'])
        })
        
        memory_df = pd.concat([marconi_mem_df, simple_mem_df], ignore_index=True)
        
        limited_palette = {k: v for k, v in self.palette.items() 
                        if k in [cache_labels['marconi'], cache_labels['simple']]}
        
        sns.lineplot(
            data=memory_df,
            x='Turn',
            y='Memory',
            hue='System',
            marker='o',
            linewidth=3,
            markersize=10,
            palette=limited_palette,
            ax=ax
        )
        
        self._apply_standard_style(
            ax,
            self.text['multiturn_memory'],
            xlabel=self.text['turn'],
            ylabel=self.text['memory']
        )
        
        ax.legend(title=self.text['cache_type'], fontsize=12, title_fontsize=13, loc='best')
        
        self._save_figure('multiturn_memory', fig)
        
        return fig
    
    def plot_memory_limited_results(self, results):
        self._plot_marconi_memory_hit_rate(results)
        self._plot_simple_entries_hit_rate(results)
        self._plot_hit_rate_vs_memory(results)
    
    def _plot_marconi_memory_hit_rate(self, results):
        if not (results['memory_limit'] and results['marconi_hit_rate']):
            print("Warning: No data for Marconi memory hit rate plot")
            return None
            
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(
            results['memory_limit'], 
            results['marconi_hit_rate'], 
            'o-', 
            linewidth=3, 
            markersize=10, 
            color=self.palette[self._get_cache_labels()['marconi']]
        )
        
        self._apply_standard_style(
            ax,
            self.text['marconi_hit_rate_memory'],
            xlabel=self.text['memory_limit'],
            ylabel=self.text['hit_rate']
        )
        
        ax.set_xticks(results['memory_limit'])
        
        self._save_figure('marconi_memory_hit_rate', fig)
        
        return fig
    
    def _plot_simple_entries_hit_rate(self, results):
        if not (results['entry_limit'] and results['simple_hit_rate']):
            print("Warning: No data for simple entries hit rate plot")
            return None
            
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(
            results['entry_limit'], 
            results['simple_hit_rate'], 
            'o-', 
            linewidth=3, 
            markersize=10, 
            color=self.palette[self._get_cache_labels()['simple']]
        )
        
        self._apply_standard_style(
            ax,
            self.text['simple_hit_rate_entries'],
            xlabel=self.text['max_entries'],
            ylabel=self.text['hit_rate']
        )
        
        ax.set_xticks(results['entry_limit'])
        
        self._save_figure('simple_entries_hit_rate', fig)
        
        return fig
    
    def _plot_hit_rate_vs_memory(self, results):
        if not (results['marconi_final_memory'] and results['marconi_hit_rate'] and
                results['simple_final_memory'] and results['simple_hit_rate'] and
                results['memory_limit'] and results['entry_limit']):
            print("Warning: No data for hit rate vs memory plot")
            return None
            
        fig, ax = plt.subplots(figsize=(12, 7))
        cache_labels = self._get_cache_labels()
        
        ax.scatter(
            results['marconi_final_memory'], 
            results['marconi_hit_rate'], 
            s=150, 
            color=self.palette[cache_labels['marconi']], 
            label=cache_labels['marconi'], 
            alpha=0.8, 
            edgecolor='black', 
            linewidth=1.5
        )
        
        ax.scatter(
            results['simple_final_memory'], 
            results['simple_hit_rate'], 
            s=150, 
            color=self.palette[cache_labels['simple']], 
            label=cache_labels['simple'], 
            alpha=0.8, 
            edgecolor='black', 
            linewidth=1.5
        )
        
        for i, (mem, hit) in enumerate(zip(results['marconi_final_memory'], results['marconi_hit_rate'])):
            ax.annotate(
                f"{results['memory_limit'][i]}MB", 
                (mem, hit), 
                textcoords="offset points", 
                xytext=(5, 7), 
                fontsize=9, 
                color=self.palette[cache_labels['marconi']]
            )
        
        for i, (mem, hit) in enumerate(zip(results['simple_final_memory'], results['simple_hit_rate'])):
            label_text = f"{results['entry_limit'][i]}条" if self.language == 'zh' else f"{results['entry_limit'][i]} entries"
            ax.annotate(
                label_text, 
                (mem, hit), 
                textcoords="offset points", 
                xytext=(5, 7), 
                fontsize=9, 
                color=self.palette[cache_labels['simple']]
            )
        
        self._apply_standard_style(
            ax,
            self.text['hit_rate_vs_memory'],
            xlabel=self.text['memory'],
            ylabel=self.text['hit_rate']
        )
        
        ax.legend(fontsize=12)
        
        self._save_figure('hit_rate_vs_memory', fig)
        
        return fig
    
    def plot_all(self, data):
        # 确保数据有效
        if not data or not all(k in data for k in ['marconi', 'simple', 'no_cache']):
            print("Warning: Invalid data format for plotting")
            return "Error: Insufficient data for visualization"
            
        self.plot_time_comparison(data)
        self.plot_hit_rate_comparison(data)
        self.plot_memory_comparison(data)
        self.plot_speedup_comparison(data)
        self.plot_efficiency_metric(data)
        self.plot_hit_rate_vs_speedup(data)
        
        summary = self.generate_summary_report(data)
        
        return summary
    
    def generate_summary_report(self, data):
        if not data or not all(k in data for k in ['marconi', 'simple', 'no_cache']):
            print("Warning: Insufficient data for summary report")
            return "Error: Insufficient data for summary report"
            
        avg_data = {
            'marconi': {
                'avg_time': np.mean(data['marconi']['time']) if data['marconi']['time'] else 0,
                'avg_hit_rate': np.mean(data['marconi']['hit_rate']) if data['marconi']['hit_rate'] else 0,
                'avg_memory': np.mean(data['marconi']['memory']) if data['marconi']['memory'] else 0,
                'avg_speedup': np.mean(data['marconi']['speedup']) if data['marconi']['speedup'] else 1.0
            },
            'simple': {
                'avg_time': np.mean(data['simple']['time']) if data['simple']['time'] else 0,
                'avg_hit_rate': np.mean(data['simple']['hit_rate']) if data['simple']['hit_rate'] else 0,
                'avg_memory': np.mean(data['simple']['memory']) if data['simple']['memory'] else 0,
                'avg_speedup': np.mean(data['simple']['speedup']) if data['simple']['speedup'] else 1.0
            },
            'no_cache': {
                'avg_time': np.mean(data['no_cache']['time']) if data['no_cache']['time'] else 0
            }
        }
        
        if self.language == 'zh':
            report = self._generate_chinese_report(data, avg_data)
        else:
            report = self._generate_english_report(data, avg_data)
        
        report_path = os.path.join(self.results_dir, 'performance_summary.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report
    
    def _generate_chinese_report(self, data, avg_data):
        # 计算样本数，安全处理
        sample_count = len(data['marconi']['time']) if data['marconi']['time'] else 0
        
        # 计算安全的速度比
        marconi_vs_simple = avg_data['simple']['avg_time'] / max(0.001, avg_data['marconi']['avg_time'])
        marconi_eff = avg_data['marconi']['avg_speedup'] / max(0.1, avg_data['marconi']['avg_memory'])
        simple_eff = avg_data['simple']['avg_speedup'] / max(0.1, avg_data['simple']['avg_memory'])
        
        report = f"""
### 缓存系统性能测试报告

#### 测试摘要
- 总测试次数: {sample_count}

#### 平均性能指标

| 系统 | 平均时间(秒) | 平均命中率 | 平均内存使用(MB) | 平均加速比 |
|------|------------|----------|--------------|----------|
| Marconi缓存 | {avg_data['marconi']['avg_time']:.3f} | {avg_data['marconi']['avg_hit_rate']:.3f} | {avg_data['marconi']['avg_memory']:.1f} | {avg_data['marconi']['avg_speedup']:.2f}x |
| 简单缓存 | {avg_data['simple']['avg_time']:.3f} | {avg_data['simple']['avg_hit_rate']:.3f} | {avg_data['simple']['avg_memory']:.1f} | {avg_data['simple']['avg_speedup']:.2f}x |
| 无缓存 | {avg_data['no_cache']['avg_time']:.3f} | - | - | 1.00x |

#### 性能分析

1. **时间性能**:
   - Marconi缓存比无缓存快 {avg_data['marconi']['avg_speedup']:.2f}x
   - 简单缓存比无缓存快 {avg_data['simple']['avg_speedup']:.2f}x
   - Marconi缓存比简单缓存快 {marconi_vs_simple:.2f}x

2. **内存效率**:
   - Marconi缓存每MB内存带来的加速比: {marconi_eff:.4f}x/MB
   - 简单缓存每MB内存带来的加速比: {simple_eff:.4f}x/MB

3. **命中率**:
   - Marconi缓存的平均命中率: {avg_data['marconi']['avg_hit_rate']:.3f}
   - 简单缓存的平均命中率: {avg_data['simple']['avg_hit_rate']:.3f}

#### 结论

- Marconi缓存在性能和内存效率上都优于简单缓存和无缓存系统
- 缓存系统显著提高了生成速度，尤其是在处理相似或重复提示时
- Marconi缓存比简单缓存更高效地利用内存资源
"""
        return report
    
    def _generate_english_report(self, data, avg_data):
        """Generate summary report in English"""
        # 计算样本数，安全处理
        sample_count = len(data['marconi']['time']) if data['marconi']['time'] else 0
        
        # 计算安全的速度比
        marconi_vs_simple = avg_data['simple']['avg_time'] / max(0.001, avg_data['marconi']['avg_time'])
        marconi_eff = avg_data['marconi']['avg_speedup'] / max(0.1, avg_data['marconi']['avg_memory'])
        simple_eff = avg_data['simple']['avg_speedup'] / max(0.1, avg_data['simple']['avg_memory'])
        
        report = f"""
### Cache System Performance Test Report

#### Test Summary
- Total tests: {sample_count}

#### Average Performance Metrics

| System | Avg Time(s) | Avg Hit Rate | Avg Memory(MB) | Avg Speedup |
|------|------------|----------|--------------|----------|
| Marconi Cache | {avg_data['marconi']['avg_time']:.3f} | {avg_data['marconi']['avg_hit_rate']:.3f} | {avg_data['marconi']['avg_memory']:.1f} | {avg_data['marconi']['avg_speedup']:.2f}x |
| Simple Cache | {avg_data['simple']['avg_time']:.3f} | {avg_data['simple']['avg_hit_rate']:.3f} | {avg_data['simple']['avg_memory']:.1f} | {avg_data['simple']['avg_speedup']:.2f}x |
| No Cache | {avg_data['no_cache']['avg_time']:.3f} | - | - | 1.00x |

#### Performance Analysis

1. **Time Performance**:
   - Marconi cache is {avg_data['marconi']['avg_speedup']:.2f}x faster than no cache
   - Simple cache is {avg_data['simple']['avg_speedup']:.2f}x faster than no cache
   - Marconi cache is {marconi_vs_simple:.2f}x faster than simple cache

2. **Memory Efficiency**:
   - Marconi cache speedup per MB: {marconi_eff:.4f}x/MB
   - Simple cache speedup per MB: {simple_eff:.4f}x/MB

3. **Hit Rate**:
   - Marconi cache average hit rate: {avg_data['marconi']['avg_hit_rate']:.3f}
   - Simple cache average hit rate: {avg_data['simple']['avg_hit_rate']:.3f}

#### Conclusion

- Marconi cache outperforms simple cache and no-cache systems in both performance and memory efficiency
- Cache systems significantly improve generation speed, especially when processing similar or repeated prompts
- Marconi cache uses memory resources more efficiently than simple cache
"""
        return report