#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题四：短途车优先权分配系统
合并了优先权计算和可视化功能

功能：
1. 计算各行程的净收益
2. 确定短途车在长途车队列中的插队位置
3. 生成优先权分配方案
4. 可视化插队位置分布

原始文件：Question4_Decision.py + Question4_Decision2.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    """
    加载数据文件
    """
    print("加载数据文件...")

    # 读取车辆行程数据
    trips_file = 'data/processed/Taxi_Trips标记.xlsx'  # 原来是 'E:\新建文件夹\Taxi_Trips标记 的副本.xlsx'
    df_trips = pd.read_excel(trips_file, engine='openpyxl')

    # 读取等待时间成本数据
    cost_file = 'data/processed/Decision_Model_Results(3).xlsx'  # 原来是 'E:\\新建文件夹\\Decision_Model_Results(3).xlsx'
    df_cost = pd.read_excel(cost_file, engine='openpyxl')

    print(f"加载行程数据：{len(df_trips)} 条记录")
    print(f"加载成本数据：{len(df_cost)} 条记录")

    return df_trips, df_cost


def setup_parameters():
    """
    设置模型参数
    """
    print("设置模型参数...")

    # 设定燃料费和费率
    params = {
        'fuel_cost_per_km': 7.76 / 12.5,  # 每公里燃料成本
        'day_start_fare': 10,  # 日间起步价
        'day_fare_per_km': 2.7,  # 日间每公里费率
        'night_start_fare': 13,  # 夜间起步价
        'night_fare_per_km': 3.51  # 夜间每公里费率
    }

    return params


def parse_time_string(time_str):
    """
    自定义函数处理时间字符串，去除秒部分
    """
    return datetime.strptime(time_str.strip(), '%H:%M:%S').time()


def parse_time_interval(time_interval_str):
    """
    将时间段解析为开始时间和结束时间
    """
    start_time_str, end_time_str = time_interval_str.split(' - ')
    start_time = parse_time_string(start_time_str)
    end_time = parse_time_string(end_time_str)
    return start_time, end_time


def prepare_cost_data(df_cost):
    """
    准备成本数据
    """
    print("处理时间成本数据...")

    # 解析时间段并生成新的列
    df_cost[['Start Time', 'End Time']] = df_cost['Time Interval'].apply(
        parse_time_interval).apply(pd.Series)

    return df_cost


def calculate_net_profit(row, df_cost, params):
    """
    计算时间段内的净收益
    """
    distance = row['距离(km)']
    time_of_day = row['离开时间'].time()

    # 计算燃料费
    fuel_cost = distance * params['fuel_cost_per_km']

    # 判断是白天还是夜间
    day_start = datetime.strptime('06:00:00', '%H:%M:%S').time()
    day_end = datetime.strptime('23:00:00', '%H:%M:%S').time()

    if day_start <= time_of_day <= day_end:
        start_fare = params['day_start_fare']
        fare_per_km = params['day_fare_per_km']
    else:
        start_fare = params['night_start_fare']
        fare_per_km = params['night_fare_per_km']

    # 计算收益
    if distance <= 2:
        revenue = start_fare
    else:
        revenue = start_fare + (distance - 2) * fare_per_km

    # 根据时间段获取等待时间成本
    time_cost_matches = df_cost.loc[
        (df_cost['Start Time'] <= time_of_day) &
        (df_cost['End Time'] >= time_of_day), 'Time Cost C_w'
    ].values

    if len(time_cost_matches) > 0:
        time_cost = time_cost_matches[0]
    else:
        time_cost = 0  # 如果没有找到匹配的时间段，设为0

    # 计算净收益
    net_profit = revenue - fuel_cost - time_cost
    return net_profit


def process_trip_data(df_trips, df_cost, params):
    """
    处理行程数据，计算净收益
    """
    print("计算各行程净收益...")

    # 计算净收益并插入新列
    df_trips['净收益'] = df_trips.apply(
        lambda row: calculate_net_profit(row, df_cost, params), axis=1)

    # 区分短途和长途
    df_short_trips = df_trips[df_trips['分类'] == '短途'].copy()
    df_long_trips = df_trips[df_trips['分类'] == '长途'].copy()

    print(f"短途行程：{len(df_short_trips)} 条")
    print(f"长途行程：{len(df_long_trips)} 条")

    return df_short_trips, df_long_trips


def calculate_insertion_positions(df_short_trips, df_long_trips):
    """
    计算短途车的插队位置
    """
    print("计算短途车插队位置...")

    # 对长途行程按净收益排序，计算排序后的队列位置
    df_long_trips_sorted = df_long_trips.sort_values(
        by='净收益', ascending=False).reset_index(drop=True)

    def get_insertion_position(short_trip_net_profit):
        """计算插队位置"""
        # 计算插队位置
        df_long_trips_sorted['插队位置'] = df_long_trips_sorted.index + 1
        df_long_trips_sorted['相对净收益'] = df_long_trips_sorted['净收益'] - short_trip_net_profit
        insert_position = df_long_trips_sorted[df_long_trips_sorted['相对净收益'] < 0].shape[0] + 1

        # 根据位置分类
        total_long_trips = len(df_long_trips_sorted)
        if insert_position <= total_long_trips * 0.3:
            return '前'
        elif insert_position <= total_long_trips * 0.7:
            return '中'
        else:
            return '后'

    def determine_insertion_positions(short_trips):
        """确定所有短途车的插队位置"""
        results = []
        for _, row in short_trips.iterrows():
            net_profit = row['净收益']
            distance = row['距离(km)']
            position = get_insertion_position(net_profit)
            results.append({
                '距离(km)': distance,
                '净收益': net_profit,
                '插队位置': position
            })
        return results

    # 获取短途车的插队信息
    insertion_positions = determine_insertion_positions(df_short_trips)

    # 转换为DataFrame
    insertion_df = pd.DataFrame(insertion_positions)

    return insertion_df, df_long_trips_sorted


def save_insertion_results(insertion_df):
    """
    保存插队结果
    """
    print("保存插队分析结果...")

    # 保存结果到新文件
    output_file = 'data/processed/短途车插队信息.xlsx'  # 原来是 'E:\新建文件夹\短途车插队信息.xlsx'
    insertion_df.to_excel(output_file, index=False)

    print(f"短途车插队信息已保存到 {output_file}")

    return output_file


def visualize_insertion_analysis(insertion_df, df_long_trips_sorted):
    """
    可视化插队分析结果
    """
    print("生成插队分析可视化图表...")

    # 绘图
    plt.figure(figsize=(12, 8))

    # 绘制长途车队列位置（黑色散点图）
    plt.subplot(2, 1, 1)
    plt.scatter(df_long_trips_sorted.index, [0] * len(df_long_trips_sorted),
                color='black', label='长途', alpha=0.5, s=30)
    plt.title('长途车队列位置')
    plt.xlabel('队列位置')
    plt.yticks([])
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 绘制短途车插队位置（彩色散点图）
    plt.subplot(2, 1, 2)
    colors = {'前': 'orange', '中': 'green', '后': 'red'}

    for position, color in colors.items():
        subset = insertion_df[insertion_df['插队位置'] == position]
        if not subset.empty:
            plt.scatter(subset['距离(km)'], [0] * len(subset),
                        color=color, label=f'{position}部插队', s=100, alpha=0.7)

    plt.title('短途车插队位置分布')
    plt.xlabel('返回距离 (km)')
    plt.yticks([])
    plt.legend(title='插队位置')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    plt.savefig('results/figures/priority_queue_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_insertion_results(insertion_df):
    """
    打印插队结果摘要
    """
    print("\n短途车插队位置分析结果：")
    print("=" * 50)

    # 按距离排序显示结果
    for _, row in insertion_df.iterrows():
        print(f"短途车返回距离为 {row['距离(km)']:.2f} km 时，"
              f"净收益 {row['净收益']:.2f} 元，"
              f"可插队到长途车队的 {row['插队位置']} 位置")

    # 统计各位置的分布
    print("\n插队位置统计：")
    position_stats = insertion_df['插队位置'].value_counts()
    for position, count in position_stats.items():
        percentage = count / len(insertion_df) * 100
        print(f"{position}部插队：{count} 辆车 ({percentage:.1f}%)")


def main():
    """
    主函数：执行完整的优先权分配分析
    """
    print("=" * 60)
    print("问题四：短途车优先权分配系统")
    print("=" * 60)

    try:
        # 步骤1：加载数据
        df_trips, df_cost = load_data()

        # 步骤2：设置参数
        params = setup_parameters()

        # 步骤3：准备成本数据
        df_cost = prepare_cost_data(df_cost)

        # 步骤4：计算净收益，分类行程
        df_short_trips, df_long_trips = process_trip_data(df_trips, df_cost, params)

        # 步骤5：计算插队位置
        insertion_df, df_long_trips_sorted = calculate_insertion_positions(
            df_short_trips, df_long_trips)

        # 步骤6：保存结果
        output_file = save_insertion_results(insertion_df)

        # 步骤7：可视化分析
        visualize_insertion_analysis(insertion_df, df_long_trips_sorted)

        # 步骤8：输出结果摘要
        print_insertion_results(insertion_df)

        # 输出总结信息
        print("\n=" * 50)
        print("优先权分配系统总结：")
        print("=" * 50)
        print(f"短途行程数量：{len(df_short_trips)}")
        print(f"长途行程数量：{len(df_long_trips)}")
        print(f"平均短途净收益：{df_short_trips['净收益'].mean():.2f} 元")
        print(f"平均长途净收益：{df_long_trips['净收益'].mean():.2f} 元")
        print(f"结果已保存到：{output_file}")

        print("\n短途车优先权分配系统分析完成！")

    except Exception as e:
        print(f"程序执行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()