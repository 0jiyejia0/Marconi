# SSM-Transformer 缓存性能测试框架

## 项目结构

```
project_root/
├── model/                    # 模型相关代码
│   ├── attention.py          # 自注意力层
│   ├── ssm_layer.py          # 状态空间模型层
│   ├── transformer_block.py  # SSM-Transformer块
│   └── transformer_model.py  # 完整模型
│
├── cache/                    # 缓存系统
│   ├── base_cache.py         # 缓存基类
│   ├── marconi_cache.py      # Marconi缓存实现（高级缓存策略）
│   ├── simple_cache.py       # 简单KV缓存实现（基于LRU）
│   └── no_cache.py           # 无缓存对照组
│
├── benchmarks/               # 基准测试
│   ├── runner.py             # 测试运行器（支持并行测试）
│   └── visualization.py      # 可视化工具（支持多语言输出）
│
├── utils/                    # 工具函数
│   └── common.py             # 通用工具函数
│
├── cache_test_results/       # 测试结果输出目录（自动创建）
│   ├── time_comparison.png   # 时间比较图
│   ├── hit_rate_comparison.png  # 命中率比较图
│   ├── memory_comparison.png    # 内存使用比较图
│   ├── ...                   # 其他图表和结果
│   └── performance_summary.md   # 性能总结报告
│
├── main.py                   # 主程序入口
└── requirements.txt          # 依赖项
```

## 缓存系统

本项目实现了三种缓存策略：

1. **Marconi缓存**：一种高级缓存策略，主要特点：
   - 基于前缀树的缓存存储结构
   - 自适应效用评分机制（结合时间相关度和计算效率）
   - 智能分支检测和对话感知能力
   - 内存限制下的高效驱逐策略
   - 适合长序列和多轮对话场景

2. **简单KV缓存**：一种基础的KV缓存策略：
   - 基于LRU（最近最少使用）淘汰机制
   - 简单的前缀匹配
   - 固定数量的缓存条目
   - 适合简单场景及有限资源环境

3. **无缓存**：作为对照组，每次都重新计算整个序列。

## 运行测试

### 基本用法

```bash
# 运行所有测试
python main.py --all

# 运行快速版本的所有测试（较少样本）
python main.py --quick

# 使用英文界面和输出
python main.py --all --language en
```

### 特定测试

```bash
# 重复使用相同提示测试
python main.py --reuse

# 相似提示测试
python main.py --similarity

# 不同长度的提示测试
python main.py --length

# 多轮对话场景测试
python main.py --multiturn

# 内存限制下的多轮对话测试
python main.py --memory
```

### 高级选项

```bash
# 设置随机种子以保证结果可重现
python main.py --all --seed 123

# 禁用并行测试（如果遇到问题）
python main.py --all --no-parallel

# 自定义结果输出目录
python main.py --all --output-dir my_results

# 配置缓存参数
python main.py --all --marconi-memory 256 --simple-entries 50
```

## 测试场景

1. **重复使用测试**：评估重复使用相同提示时的缓存效率
   - 展示了多次使用相同输入时的加速效果
   - 理想情况下，缓存系统在第二次及之后的运行中会显著加速

2. **相似度测试**：评估处理相似提示时的缓存效率
   - 使用具有不同相似度的提示变体
   - 测试缓存系统识别和利用部分匹配的能力

3. **长度测试**：评估不同长度提示对缓存性能的影响
   - 测试从短到长的多种输入长度
   - 展示缓存系统处理不同输入规模的能力

4. **多轮对话测试**：模拟真实多轮对话场景
   - 随着对话进行，缓存应该显示出逐渐增加的命中率
   - 测试缓存系统在长上下文中的表现

5. **内存限制测试**：评估不同内存约束下的缓存性能
   - 测试各种内存限制下的命中率和性能
   - 评估缓存策略在资源受限环境的效率

## 测试结果与解读

测试完成后，所有结果将保存在 `cache_test_results` 目录（或通过 `--output-dir` 指定的目录）中：

### 生成的图表及含义

1. **生成时间对比** (time_comparison.png)
   - 比较三种缓存策略的生成时间
   - 值越低越好，代表生成速度更快

2. **命中率对比** (hit_rate_comparison.png)
   - 比较不同缓存策略的命中率
   - 值越高越好（接近1.0），表示缓存利用率高

3. **内存使用对比** (memory_comparison.png)
   - 比较不同缓存策略的内存占用
   - 在保证性能的前提下，值越低越好

4. **加速比对比** (speedup_comparison.png)
   - 相对于无缓存的速度提升倍数
   - 值越高越好，代表缓存带来的加速效果更明显

5. **缓存效率指标** (efficiency_comparison.png)
   - 每单位内存带来的加速比(加速比/MB)
   - 值越高越好，代表缓存策略更高效利用内存

6. **多轮对话相关图表**
   - 展示随着对话轮次增加的性能变化趋势
   - 理想情况下，命中率应随轮次增加而上升，生成时间应下降

7. **内存限制测试图表**
   - 展示不同内存限制下的性能权衡
   - 帮助确定最佳内存配置

### 性能报告

系统会生成一个全面的性能报告 (performance_summary.md)，包含：

- 测试样本数量
- 各缓存系统的平均性能指标
- 系统间的直接性能比较
- 内存效率分析
- 总体结论和建议

## 依赖项

- PyTorch >= 1.10.0
- NumPy
- Matplotlib
- Pandas
- Seaborn
- tqdm

可以使用以下命令安装依赖：

```bash
pip install -r requirements.txt
```