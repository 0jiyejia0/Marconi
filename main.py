"""
SSM-Transformer缓存测试主程序

此程序比较不同缓存策略的性能：
- Marconi缓存: 基于前缀树和效用评分的高级缓存策略
- 简单KV缓存: 基于LRU策略的简单前缀缓存
- 无缓存: 每次都重新计算整个序列
"""

import argparse
import torch
import os
import matplotlib
matplotlib.use('Agg')
from utils import create_dummy_model, set_seed, ensure_dir
from benchmarks import TestbenchRunner

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='缓存系统性能测试')
    
    # 测试类型参数
    test_group = parser.add_argument_group('测试类型')
    test_group.add_argument('--all', action='store_true', help='运行所有测试')
    test_group.add_argument('--quick', action='store_true', help='运行快速版本的所有测试')
    test_group.add_argument('--reuse', action='store_true', help='运行重复使用测试')
    test_group.add_argument('--similarity', action='store_true', help='运行相似度测试')
    test_group.add_argument('--length', action='store_true', help='运行长度测试')
    test_group.add_argument('--multiturn', action='store_true', help='运行多轮对话测试')
    test_group.add_argument('--memory', action='store_true', help='运行内存限制测试')
    
    # 通用配置参数
    config_group = parser.add_argument_group('测试配置')
    config_group.add_argument('--seed', type=int, default=42, help='随机种子')
    config_group.add_argument('--no-parallel', action='store_true', help='禁用并行测试')
    config_group.add_argument('--language', choices=['zh', 'en'], default='zh', help='报告和图表语言')
    config_group.add_argument('--output-dir', type=str, default='cache_test_results', help='结果输出目录')
    
    # 模型参数
    model_group = parser.add_argument_group('模型配置')
    model_group.add_argument('--hidden-dim', type=int, default=128, help='隐藏层维度')
    model_group.add_argument('--num-layers', type=int, default=2, help='模型层数')
    
    # 缓存配置参数
    cache_group = parser.add_argument_group('缓存配置')
    cache_group.add_argument('--marconi-memory', type=int, default=512, help='Marconi缓存内存上限(MB)')
    cache_group.add_argument('--simple-entries', type=int, default=100, help='简单缓存最大条目数')
    
    args = parser.parse_args()
    
    # 如果没有指定任何测试，则默认运行所有测试
    if not (args.all or args.quick or args.reuse or args.similarity or args.length or 
            args.multiturn or args.memory):
        args.all = True
    
    print("初始化测试环境...")
    
    # 设置随机种子以保证结果可复现
    set_seed(args.seed)
    
    # 确保输出目录存在
    ensure_dir(args.output_dir)
    
    # 创建测试模型
    model = create_dummy_model(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
    
    # 创建测试运行器
    runner = TestbenchRunner(
        model=model,
        max_marconi_memory=args.marconi_memory * 1024 * 1024,  # 转换为字节
        simple_cache_entries=args.simple_entries,
        results_dir=args.output_dir,
        use_parallel=not args.no_parallel
    )
    
    # 设置可视化器语言
    runner.visualizer.language = args.language
    runner.visualizer._set_font_properties(args.language)
    
    # 通过使用优化后的run_all_tests函数简化代码
    if args.all or args.quick:
        print("\n=== 运行所有测试 ===")
        summary = runner.run_all_tests(quick=args.quick, seed=args.seed)
        print("\n性能摘要:")
        print(summary)
    else:
        # 运行指定的测试
        if args.reuse:
            print("\n=== 测试1: 重复使用相同提示 ===")
            runner.run_reuse_test(num_iterations=3, prompt_length=20, gen_length=20, seed=args.seed)
        
        if args.similarity:
            print("\n=== 测试2: 相似提示测试 ===")
            # 测试不同相似度
            for similarity in [0.8]:
                runner.run_similarity_test(num_prompts=3, prompt_length=20, 
                                        similarity=similarity, gen_length=20, seed=args.seed)
        
        if args.length:
            print("\n=== 测试3: 不同长度的提示 ===")
            runner.run_length_test(min_length=10, max_length=30, step=10, gen_length=20, seed=args.seed)
        
        if args.multiturn:
            print("\n=== 测试4: 多轮对话场景 ===")
            runner.test_multiturn_dialogue(num_turns=5, initial_length=10, 
                                         turn_length=5, gen_length=10, seed=args.seed)
        
        if args.memory:
            print("\n=== 测试5: 内存限制下的多轮对话 ===")
            # 使用较小的值以加快测试速度
            memory_limits = [16, 32, 64, 128]  # MB
            max_entries = [5, 10, 20, 50]  # 条目数
            runner.test_memory_limited_dialogue(
                memory_limits=memory_limits,
                max_entries=max_entries,
                num_turns=4,
                initial_length=10,
                turn_length=5,
                gen_length=10
            )
        
        # 可视化和保存结果
        summary = runner.visualize_results()
        print("\n性能摘要:")
        print(summary)
    
    print("\n测试完成! 所有结果已保存到", args.output_dir)

if __name__ == "__main__":
    main()