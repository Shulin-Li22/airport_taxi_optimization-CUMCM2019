# -*- coding: utf-8 -*-
"""
问题四：行程距离分析

功能：
1. 计算出租车行程距离
2. 确定短途/长途分界阈值
3. 绘制距离分布图和累积分布图
4. 标记行程分类

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    计算两点间的Haversine距离

    Args:
        lon1, lat1: 第一个点的经纬度
        lon2, lat2: 第二个点的经纬度

    Returns:
        float: 距离 (公里)
    """
    R = 6371.0  # 地球半径，单位为公里
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    return distance


def calculate_trip_distances():
    """
    计算行程距离并进行分类
    """
    print("开始计算行程距离...")

    # 读取 CSV 文件
    file_path = '../data/processed/Taxi_Trips.csv'
    df = pd.read_csv(file_path)

    print(f"原始数据形状: {df.shape}")
    print("原始列名:", df.columns.tolist())

    df.columns = ['车辆ID', '开始时间', '结束时间', '起始经度', '起始纬度', '终止经度', '终止纬度']

    print("重命名后的列名:", df.columns.tolist())

    # 转换时间列
    df['开始时间'] = pd.to_datetime(df['开始时间'])
    df['结束时间'] = pd.to_datetime(df['结束时间'])

    # 计算距离
    print("正在计算距离...")
    df['距离(km)'] = df.apply(lambda row: haversine_distance(
        row['起始经度'], row['起始纬度'],
        row['终止经度'], row['终止纬度']
    ), axis=1)

    print(f"计算完成，共{len(df)}条行程记录")

    # 显示基本统计信息
    print(f"距离统计:")
    print(f"最小距离: {df['距离(km)'].min():.3f} km")
    print(f"最大距离: {df['距离(km)'].max():.3f} km")
    print(f"平均距离: {df['距离(km)'].mean():.3f} km")
    print(f"中位数距离: {df['距离(km)'].median():.3f} km")

    return df


def analyze_distance_distribution(df):
    """
    分析距离分布并确定阈值
    """
    print("分析距离分布...")

    # 计算分位数
    percentiles = [25, 50, 75, 90]
    distance_percentiles = np.percentile(df['距离(km)'], percentiles)

    print("行程距离分位数分析：")
    for p, value in zip(percentiles, distance_percentiles):
        print(f"第{p}百分位数: {value:.2f} km")

    # 设定阈值（第90百分位数）
    threshold = distance_percentiles[3]  # 90%分位数
    print(f"\n选择的短途/长途分界阈值: {threshold:.2f} km")

    return threshold, distance_percentiles


def classify_trips(df, threshold):
    """
    根据阈值对行程进行分类
    """
    print(f"根据{threshold:.2f}km阈值对行程进行分类...")

    # 根据距离划分短途和长途
    df['分类'] = np.where(df['距离(km)'] <= threshold, '短途', '长途')

    # 统计分类结果
    classification_stats = df['分类'].value_counts()
    print("分类统计结果：")
    for category, count in classification_stats.items():
        percentage = count / len(df) * 100
        print(f"{category}: {count} 条 ({percentage:.1f}%)")

    return df


def visualize_distance_analysis(df, threshold, distance_percentiles):
    """
    可视化距离分析结果
    """
    print("生成距离分析可视化图表...")

    # 创建结果目录
    os.makedirs('../results/figures', exist_ok=True)

    # 使用whitegrid风格
    sns.set(style='whitegrid')

    # 绘制行驶距离的直方图
    plt.figure(figsize=(12, 8))
    plt.hist(df['距离(km)'], bins=50, color='lightblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                label=f'90%分位数阈值: {threshold:.2f} km')

    # 添加分位数线
    colors = ['blue', 'green', 'orange']
    percentile_labels = ['25%分位数', '50%分位数', '75%分位数']
    for i, (perc, color, label) in enumerate(zip(distance_percentiles[:3], colors, percentile_labels)):
        plt.axvline(x=perc, color=color, linestyle=':', alpha=0.7,
                    label=f'{label}: {perc:.2f} km')

    plt.title('出租车行程距离分布图', fontsize=16, fontweight='bold')
    plt.xlabel('行程距离 (km)', fontsize=12)
    plt.ylabel('频次', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # 保存图片
    plt.savefig('../results/figures/trip_distance_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 绘制累积分布曲线
    plt.figure(figsize=(12, 8))
    sorted_distances = np.sort(df['距离(km)'])
    cdf = np.arange(len(sorted_distances)) / float(len(sorted_distances))
    plt.plot(sorted_distances, cdf, color='brown', linewidth=2, label='累积分布曲线')
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                label=f'90%分位数阈值: {threshold:.2f} km')

    # 添加分位数点
    colors = ['blue', 'green', 'orange']
    percentile_labels = ['25%', '50%', '75%']
    for i, (perc, value, color, label) in enumerate(
            zip([25, 50, 75], distance_percentiles[:3], colors, percentile_labels)):
        plt.plot(value, perc / 100, 'o', markersize=8, color=color,
                 label=f'{label}分位数: {value:.2f} km')

    plt.title('出租车行程距离累积分布图', fontsize=16, fontweight='bold')
    plt.xlabel('行程距离 (km)', fontsize=12)
    plt.ylabel('累积频率', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # 保存图片
    plt.savefig('../results/figures/trip_distance_cdf.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_results(df):
    """
    保存分析结果
    """
    print("保存分析结果...")

    # 创建输出目录
    os.makedirs('../data/processed', exist_ok=True)

    # 保存结果到新的 Excel 文件
    output_file = '../data/processed/Taxi_Trips_marked.xlsx'
    df.to_excel(output_file, index=False)

    print(f'车辆已经根据长途和短途划分，并保存到 {output_file}')

    # 同时保存为CSV格式
    csv_output_file = '../data/processed/Taxi_Trips_marked.csv'
    df.to_csv(csv_output_file, index=False, encoding='utf-8')
    print(f'同时保存CSV格式到 {csv_output_file}')

    return output_file


def main():
    """
    主函数：执行完整的距离分析流程
    """
    print("=" * 60)
    print("问题四：行程距离分析")
    print("=" * 60)

    try:
        # 步骤1：计算行程距离
        df = calculate_trip_distances()

        # 步骤2：分析距离分布，确定阈值
        threshold, distance_percentiles = analyze_distance_distribution(df)

        # 步骤3：根据阈值分类行程
        df_classified = classify_trips(df, threshold)

        # 步骤4：可视化分析结果
        visualize_distance_analysis(df_classified, threshold, distance_percentiles)

        # 步骤5：保存结果
        output_file = save_results(df_classified)

        # 输出总结信息
        print("\n" + "=" * 40)
        print("距离分析总结：")
        print("=" * 40)
        print(f"总行程数：{len(df_classified)}")
        print(f"短途/长途分界阈值：{threshold:.2f} km")
        short_trips = len(df_classified[df_classified['分类'] == '短途'])
        long_trips = len(df_classified[df_classified['分类'] == '长途'])
        print(f"短途行程：{short_trips} 条 ({short_trips / len(df_classified) * 100:.1f}%)")
        print(f"长途行程：{long_trips} 条 ({long_trips / len(df_classified) * 100:.1f}%)")
        print(f"结果已保存到：{output_file}")

        print("\n距离分析完成！")

    except FileNotFoundError:
        print("错误：找不到 'Taxi_Trips.csv' 文件，请确保文件在当前目录下")
    except Exception as e:
        print(f"程序执行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()