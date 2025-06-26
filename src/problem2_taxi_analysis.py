# -*- coding: utf-8 -*-
"""
问题二：出租车载客统计分析

功能：
1. 分析每小时载客出租车数量
2. 计算时间权重用于后续建模
3. 生成载客数量可视化图表

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


def setup_environment():
    """设置项目环境和中文字体"""
    # 确保工作目录在项目根目录
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # 确保输出目录存在
    os.makedirs('results/figures', exist_ok=True)

    # 设置中文字体和图形风格
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set(style="whitegrid")


def load_gps_data():
    """加载原始GPS数据"""
    print("加载原始出租车GPS数据...")

    file_path = 'data/raw/taxi_gps_data.txt'

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    # 读取TXT文件，假设文件使用逗号分隔
    df = pd.read_csv(file_path, sep=',', header=None)
    print(f"成功加载GPS数据，共 {len(df)} 条记录")

    return df


def process_time_data(df):
    """处理时间数据并提取小时信息"""
    print("处理时间数据...")

    # 将第二列（时间列）转换为datetime对象
    df[1] = pd.to_datetime(df[1], format='%H:%M:%S', errors='coerce')

    # 提取小时信息
    df['小时'] = df[1].dt.hour

    print("时间数据处理完成")
    return df


def analyze_hourly_passenger_taxis(df):
    """分析每小时载客出租车数量"""
    print("分析每小时载客出租车数量...")

    # 筛选载客状态（第五列为1）的数据并按小时分组统计
    passenger_taxis_per_hour = df[df[4] == 1].groupby('小时').size()

    # 计算总数和权重
    total_passenger_taxis = passenger_taxis_per_hour.sum()
    weights = passenger_taxis_per_hour / total_passenger_taxis

    print(f"全天载客出租车总数：{total_passenger_taxis}")

    return passenger_taxis_per_hour, weights, total_passenger_taxis


def print_analysis_results(passenger_taxis_per_hour, weights):
    """打印分析结果"""
    print("\n每小时载客出租车统计：")
    print("=" * 50)
    print(f"{'小时':<6} {'数量':<12} {'权重':<10}")
    print("-" * 50)

    for hour, count in passenger_taxis_per_hour.items():
        weight = weights[hour]
        print(f"{hour:>4}   {count:>10}   {weight:>8.4f}")


def visualize_hourly_data(passenger_taxis_per_hour):
    """可视化每小时载客数据"""
    print("生成载客数量可视化图表...")

    # 创建柱状图
    plt.figure(figsize=(12, 6))
    bars = sns.barplot(
        x=passenger_taxis_per_hour.index,
        y=passenger_taxis_per_hour.values,
        palette='viridis'
    )

    plt.title('每小时载客出租车数量分布', fontsize=16, fontweight='bold')
    plt.xlabel('时间 (小时)', fontsize=12)
    plt.ylabel('载客出租车数量', fontsize=12)
    plt.xticks(range(0, 24))
    plt.grid(True, alpha=0.3)

    # 在柱子顶部添加数值标签
    for i, bar in enumerate(bars.patches):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=8)

    # 保存图片
    plt.tight_layout()
    plt.savefig('results/figures/hourly_passenger_taxis.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    print("可视化图表已保存到: results/figures/hourly_passenger_taxis.png")


def save_analysis_results(passenger_taxis_per_hour, weights):
    """保存分析结果到Excel文件"""
    print("保存分析结果...")

    # 创建结果DataFrame
    results_df = pd.DataFrame({
        '小时': passenger_taxis_per_hour.index,
        '载客出租车数量': passenger_taxis_per_hour.values,
        '时间权重': weights.values
    })

    # 保存到Excel
    output_file = 'data/processed/每小时载客统计.xlsx'
    results_df.to_excel(output_file, index=False)

    print(f"分析结果已保存到: {output_file}")

    return output_file


def main():
    """
    主函数：执行完整的出租车载客分析流程
    """
    print("=" * 60)
    print("问题二：出租车载客统计分析")
    print("=" * 60)

    try:
        # 步骤1：设置环境
        setup_environment()

        # 步骤2：加载GPS数据
        df = load_gps_data()

        # 步骤3：处理时间数据
        df = process_time_data(df)

        # 步骤4：分析每小时载客数量
        passenger_taxis_per_hour, weights, total_taxis = analyze_hourly_passenger_taxis(df)

        # 步骤5：打印分析结果
        print_analysis_results(passenger_taxis_per_hour, weights)

        # 步骤6：可视化数据
        visualize_hourly_data(passenger_taxis_per_hour)

        # 步骤7：保存结果
        output_file = save_analysis_results(passenger_taxis_per_hour, weights)

        # 输出分析摘要
        print("\n" + "=" * 40)
        print("载客分析摘要：")
        print("=" * 40)
        print(f"分析时间段：24小时")
        print(f"全天载客总数：{total_taxis:,}")
        print(f"高峰时段：{passenger_taxis_per_hour.idxmax()}时 ({passenger_taxis_per_hour.max():,}辆)")
        print(f"低谷时段：{passenger_taxis_per_hour.idxmin()}时 ({passenger_taxis_per_hour.min():,}辆)")
        print(f"平均每小时：{passenger_taxis_per_hour.mean():.0f}辆")
        print(f"结果文件：{output_file}")

        print("\n载客统计分析完成！")

    except Exception as e:
        print(f"程序执行出错：{e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)