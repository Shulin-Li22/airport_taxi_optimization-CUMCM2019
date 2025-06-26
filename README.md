# 机场出租车问题数学建模

[![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Math Modeling](https://img.shields.io/badge/category-mathematical%20modeling-orange.svg)]()

> 2019年高教社杯全国大学生数学建模竞赛C题完整解决方案

## 📖 项目简介

本项目针对机场出租车运营中的关键决策问题，通过数学建模方法提供系统性解决方案。基于深圳宝安国际机场的真实数据，建立了多个数学模型来优化出租车司机决策和机场调度系统。

**核心问题**：出租车司机在机场送客后的选择策略
- **方案A**：排队等待载客返回市区
- **方案B**：直接空载返回市区拉客

## 🎯 解决的四个关键问题

### 问题一：司机决策模型 📊
- **建模方法**：排队论M/M/1模型 + 收益期望分析
- **核心算法**：比较接客收益期望(R_A)与不接客收益期望(R_B)
- **应用价值**：为司机提供不同时段的最优决策建议

### 问题二：实际数据验证 📈
- **数据来源**：深圳宝安国际机场2013年10月22日全天数据
- **数据规模**：出租车GPS轨迹数据 + 航班到达信息
- **验证结果**：模型在真实场景下的有效性和实用性

### 问题三：乘车效率优化 🚗
- **优化目标**：机场"乘车区"上车点配置
- **技术方法**：离散事件仿真(SimPy)
- **最优方案**：15个上车点配置，实现成本与效率最佳平衡

### 问题四：短途车优先权 ⚖️
- **分界标准**：9.27公里作为短途/长途分界点
- **分配策略**：基于净收益的动态插队机制
- **预期效果**：实现短途与长途司机收益均衡

## 🏗️ 项目结构

```
airport-taxi-optimization/
├── README.md                          # 项目说明文档
├── requirements.txt                   # Python依赖包
├── docs/                             # 项目文档
│   ├── competition_problem.pdf       # 竞赛原题
│   └── research_paper.pdf           # 完整论文
├── data/                             # 数据文件
│   ├── raw/                         # 原始数据
│   │   ├── taxi_gps_data.txt       # 出租车GPS数据
│   │   └── flight_data.xlsx        # 航班信息数据
│   └── processed/                   # 处理后数据
│       ├── 出租车到离时间结果.xlsx      # 车辆到离时间
│       ├── 出租车载人筛选结果.xlsx      # 载客车辆数据
│       ├── 机场航班统计结果.xlsx        # 航班统计
│       ├── Decision_Model_Results.xlsx # 决策模型结果
│       ├── Taxi_Trips标记.xlsx        # 行程分类标记
│       └── 短途车插队信息.xlsx          # 优先权分配
├── results/                          # 输出结果
│   └── figures/                     # 图表和可视化
└── src/                             # 源代码
    ├── problem1_decision_model.py   # 问题一：决策模型
    ├── problem2_data_preprocessing.py # 问题二：数据预处理
    ├── problem2_taxi_analysis.py   # 问题二：出租车分析
    ├── problem2_flight_analysis.py # 问题二：航班分析
    ├── problem2_modeling.py        # 问题二：建模验证
    ├── problem2_heatmap.py         # 问题二：热力图
    ├── problem3_optimization.py    # 问题三：效率优化
    ├── problem4_distance_analysis.py # 问题四：距离分析
    ├── problem4_priority_system.py # 问题四：优先权系统
    ├── problem4_heatmap.py         # 问题四：行程热力图
    └── TaxiTripProcessor.java      # Java数据处理工具
```

## 🚀 快速开始

### 环境要求
- Python 3.7+
- Java 8+ (可选，用于数据预处理)

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/your-username/airport-taxi-optimization.git
cd airport-taxi-optimization
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **准备数据**
- 将原始数据文件放入 `data/raw/` 目录
- 确保数据文件路径正确

### 运行程序

**按问题顺序执行：**

```bash
# 问题一：建立决策模型
python src/problem1_decision_model.py

# 问题二：数据处理与模型验证
python src/problem2_data_preprocessing.py  # 数据预处理
python src/problem2_taxi_analysis.py      # 出租车统计
python src/problem2_flight_analysis.py    # 航班分析
python src/problem2_modeling.py           # 决策建模
python src/problem2_heatmap.py           # 生成热力图

# 问题三：乘车效率优化
python src/problem3_optimization.py

# 问题四：优先权分配
python src/problem4_distance_analysis.py  # 距离阈值分析
python src/problem4_priority_system.py    # 优先权系统
python src/problem4_heatmap.py           # 行程可视化
```

**独立运行任意程序：**
```bash
# 每个Python文件都可以独立运行
python src/problem3_optimization.py
```

## 📊 主要成果

### 数学模型
- **排队论模型**：精确计算蓄车池等待时间
- **收益期望模型**：量化不同策略的经济效益
- **仿真优化模型**：寻找最优上车点配置
- **优先权分配模型**：实现收益公平分配

### 关键结果
- **决策策略**：提供24小时分时段决策建议
- **效率提升**：优化后上车点配置显著减少等待时间
- **收益均衡**：短途车优先权机制有效提升整体收益公平性
- **实用价值**：模型可推广至其他机场的出租车管理

### 可视化分析
- 🗺️ 出租车GPS轨迹热力图
- 📈 航班密度时间分布图
- 💰 成本效益对比分析
- 🚕 优先权分配效果展示

## 🛠️ 技术栈

**核心技术：**
- **数学方法**：排队论、概率统计、优化理论
- **仿真技术**：离散事件仿真、蒙特卡罗方法
- **数据分析**：时间序列分析、空间数据处理

**编程工具：**
- **Python**：主要开发语言
- **Java**：数据预处理辅助工具

**核心库：**
```python
pandas      # 数据处理分析
numpy       # 数值计算
matplotlib  # 数据可视化  
seaborn     # 统计图表
simpy       # 离散事件仿真
gmplot      # 地理热力图
scipy       # 科学计算
openpyxl    # Excel文件处理
```

## 📈 模型创新点

1. **多维度决策模型**
   - 综合考虑时间成本、燃油成本、收益期望
   - 动态调整不同时段的决策策略

2. **真实数据驱动**
   - 基于大规模GPS轨迹数据
   - 结合实际航班信息进行验证

3. **仿真优化结合**
   - 离散事件仿真模拟复杂排队过程
   - 多目标优化寻找最佳配置方案

4. **公平性机制设计**
   - 创新的短途车优先权分配算法
   - 实现效率与公平的平衡

## 📝 文件说明

| 文件名 | 功能描述 | 输入 | 输出 |
|--------|----------|------|------|
| `problem1_decision_model.py` | 建立司机决策数学模型 | 参数设置 | 决策策略 |
| `problem2_data_preprocessing.py` | GPS和航班数据预处理 | 原始数据 | 清洗后数据 |
| `problem2_taxi_analysis.py` | 出租车运营统计分析 | GPS数据 | 统计结果 |
| `problem2_flight_analysis.py` | 航班信息处理分析 | 航班数据 | 需求分析 |
| `problem2_modeling.py` | 决策模型实证验证 | 处理数据 | 验证结果 |
| `problem2_heatmap.py` | 生成GPS位置热力图 | 位置数据 | 热力图 |
| `problem3_optimization.py` | 上车点配置优化仿真 | 仿真参数 | 最优配置 |
| `problem4_distance_analysis.py` | 行程距离分析分类 | 行程数据 | 距离阈值 |
| `problem4_priority_system.py` | 优先权分配系统 | 分类数据 | 插队策略 |
| `problem4_heatmap.py` | 行程分布热力图 | 行程数据 | 可视化图 |

## 🎖️ 竞赛背景

**竞赛信息：**
- 竞赛名称：2019年高教社杯全国大学生数学建模竞赛
- 题目编号：C题
- 题目名称：机场的出租车问题
- 研究对象：深圳宝安国际机场

**问题特点：**
- 实际应用导向的优化问题
- 多约束条件下的决策分析
- 大数据背景下的建模验证
- 兼顾理论性与实用性

## 📚 参考文献

1. Kleinrock, L. (1975). *Queueing Systems, Volume 1: Theory*. John Wiley & Sons.
2. Bertsekas, D. P. (2005). *Dynamic Programming and Optimal Control*. Athena Scientific.
3. Banks, J., et al. (2005). *Discrete-Event System Simulation*. Prentice Hall.

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源协议。

## 🙏 致谢

- 感谢**高教社杯全国大学生数学建模竞赛**提供的研究平台
- 感谢开源社区提供的优秀工具和库

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个Star支持一下！⭐**

**关键词：** `数学建模` `排队论` `离散事件仿真` `决策优化` `机场管理` `Python`

Made with ❤️ for Mathematical Modeling Community

</div>