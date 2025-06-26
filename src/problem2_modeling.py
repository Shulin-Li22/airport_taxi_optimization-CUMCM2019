# -*- coding: utf-8 -*-
"""
问题二：出租车司机决策建模与验证

功能：
1. 基于M/M/1排队论模型计算等待时间
2. 构建收益期望模型
3. 验证司机决策策略的有效性
4. 生成决策分析结果和可视化

"""

import pandas as pd
import numpy as np
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
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    # 设置中文字体和图形风格
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set(style="whitegrid")


def load_processed_data():
    """加载预处理后的GPS和航班数据"""
    print("加载预处理后的数据...")

    # 加载GPS数据
    gps_data_path = 'data/processed/出租车到离时间结果.xlsx'
    if not os.path.exists(gps_data_path):
        raise FileNotFoundError(f"GPS数据文件不存在: {gps_data_path}")

    gps_data = pd.read_excel(gps_data_path)

    # 加载航班数据
    flight_data_path = 'data/processed/机场航班统计结果.xlsx'
    if not os.path.exists(flight_data_path):
        raise FileNotFoundError(f"航班数据文件不存在: {flight_data_path}")

    flight_data = pd.read_excel(flight_data_path)

    print(f"GPS数据: {len(gps_data)} 条记录")
    print(f"航班数据: {len(flight_data)} 条记录")

    return gps_data, flight_data


def preprocess_datetime_columns(gps_data, flight_data):
    """预处理日期时间列"""
    print("处理日期时间格式...")

    # 确保日期列是datetime格式
    if 'Japanese Standard Time' in str(gps_data.columns) or '日期' in gps_data.columns:
        date_col = '日期' if '日期' in gps_data.columns else gps_data.columns[1]
        gps_data[date_col] = pd.to_datetime(gps_data[date_col])

    flight_data['计划到达时间'] = pd.to_datetime(flight_data['计划到达时间'])

    return gps_data, flight_data


def detect_arrival_departure_times(group):
    """
    检测到达和离开时间的函数

    Args:
        group: 按车牌分组的GPS数据

    Returns:
        DataFrame: 包含车牌、到达时间、离开时间的数据
    """
    arrival_times = []
    departure_times = []

    in_airport = False
    last_time = None

    date_col = '日期' if '日期' in group.columns else group.columns[1]
    license_col = '车牌' if '车牌' in group.columns else group.columns[0]

    for i in range(len(group)):
        current_time = group.iloc[i][date_col]
        if not in_airport:
            arrival_times.append(current_time)
            in_airport = True
        elif (current_time - last_time).total_seconds() > 1200:  # 超过20分钟
            departure_times.append(last_time)
            in_airport = False

        last_time = current_time

    if in_airport:
        departure_times.append(last_time)

    return pd.DataFrame({
        '车牌': group[license_col].iloc[0],
        '到达时间': arrival_times,
        '离开时间': departure_times
    })


def calculate_arrival_departure_data(gps_data):
    """计算到达离开数据"""
    print("计算车辆到达和离开时间...")

    # 按车牌分组并检测每辆车的到达和离开时间
    license_col = '车牌' if '车牌' in gps_data.columns else gps_data.columns[0]
    date_col = '日期' if '日期' in gps_data.columns else gps_data.columns[1]

    gps_data_sorted = gps_data.sort_values(by=[license_col, date_col])
    arrival_departure_times = gps_data_sorted.groupby(license_col).apply(
        detect_arrival_departure_times
    ).reset_index(drop=True)

    print(f"检测到 {len(arrival_departure_times)} 条到达离开记录")

    return arrival_departure_times


def setup_model_parameters():
    """设置模型参数"""
    print("设置模型参数...")

    # 蓄车池车辆数Q（基于唯一车牌数估算）
    Q = 100  # 可以根据实际数据调整

    # 假设每个航班的平均乘客数量P
    P = 200

    # 每小时的载客出租车数量（用于计算时间成本C_w）
    hourly_taxi_data = {
        0: 729369, 1: 522271, 2: 408140, 3: 310274, 4: 240980, 5: 246297,
        6: 336588, 7: 629110, 8: 1003028, 9: 939582, 10: 1015902, 11: 941575,
        12: 811618, 13: 903500, 14: 1129489, 15: 1131097, 16: 1024277, 17: 1086384,
        18: 989066, 19: 1122078, 20: 1135057, 21: 1162701, 22: 1173174, 23: 1009961
    }

    total_taxis = sum(hourly_taxi_data.values())

    # 收益参数
    R_s = 31.74  # 市区载客平均收益（元/小时）
    C_v = 21.73  # 空载返回市区成本（元）

    params = {
        'Q': Q,
        'P': P,
        'hourly_taxi_data': hourly_taxi_data,
        'total_taxis': total_taxis,
        'R_s': R_s,
        'C_v': C_v
    }

    return params


def calculate_decision_model(arrival_departure_times, flight_data, params):
    """计算决策模型结果"""
    print("计算决策模型...")

    # 时间间隔设置（60分钟）
    start_time = pd.Timestamp(year=2013, month=10, day=22, hour=0, minute=0)
    end_time = pd.Timestamp(year=2013, month=10, day=22, hour=23, minute=59)
    time_delta = pd.Timedelta(minutes=60)

    results = []
    lambda_values = []
    mu_values = []
    time_intervals = []

    current_time = start_time
    while current_time <= end_time:
        next_time = current_time + time_delta

        print(f"处理时间段: {current_time.time()} - {next_time.time()}")

        # 获取当前时间段内到达和离开的车辆数
        arrivals_in_interval = arrival_departure_times[
            (arrival_departure_times['到达时间'] >= current_time) &
            (arrival_departure_times['到达时间'] < next_time)
            ]
        departures_in_interval = arrival_departure_times[
            (arrival_departure_times['离开时间'] >= current_time) &
            (arrival_departure_times['离开时间'] < next_time)
            ]

        # 计算到达率λ和服务率μ
        lambda_rate = arrivals_in_interval.shape[0]
        mu_rate = departures_in_interval.shape[0]

        # 计算乘客需求量D
        flights_in_interval = flight_data[
            (flight_data['计划到达时间'] >= current_time) &
            (flight_data['计划到达时间'] < next_time)
            ]
        D_interval = flights_in_interval['航班数量'].sum() * params['P']

        # 确定接客收益R_c（日间/夜间）
        if 6 <= current_time.hour < 23:  # 日间
            R_c = 99.1
        else:  # 夜间
            R_c = 128.83

        # 计算时间成本C_w
        hour = current_time.hour
        weight = params['hourly_taxi_data'].get(hour, 0) / params['total_taxis']
        time_value_per_hour = weight * 30  # 司机的时间价值

        if mu_rate > lambda_rate:
            T_w_interval = 1 / (mu_rate - lambda_rate)
            C_w_interval = T_w_interval * time_value_per_hour
        else:
            T_w_interval = np.inf
            C_w_interval = np.inf

        # 计算收益期望
        if D_interval + params['Q'] > 0:
            R_A_interval = (D_interval / (D_interval + params['Q'])) * R_c - C_w_interval
        else:
            R_A_interval = -C_w_interval

        R_B_interval = params['R_s'] * T_w_interval - params['C_v']

        # 司机决策
        if R_A_interval > R_B_interval:
            decision = "机场排队"
        else:
            decision = "返回市区"

        # 存储结果
        lambda_values.append(lambda_rate)
        mu_values.append(mu_rate)
        time_intervals.append(f"{current_time.time()} - {next_time.time()}")

        results.append({
            "时间段": f"{current_time.time()} - {next_time.time()}",
            "到达率λ": lambda_rate,
            "服务率μ": mu_rate,
            "需求量D": D_interval,
            "平均等待时间T_w": T_w_interval if T_w_interval != np.inf else "无穷大",
            "时间成本C_w": C_w_interval if C_w_interval != np.inf else "无穷大",
            "接客收益期望R_A": R_A_interval,
            "不接客收益期望R_B": R_B_interval,
            "司机决策": decision
        })

        current_time = next_time

    return results, lambda_values, mu_values, time_intervals


def save_decision_results(results):
    """保存决策结果"""
    print("保存决策模型结果...")

    results_df = pd.DataFrame(results)
    output_file_path = 'data/processed/Decision_Model_Results.xlsx'
    results_df.to_excel(output_file_path, index=False)

    print(f"决策结果已保存到: {output_file_path}")

    return output_file_path


def visualize_decision_analysis(lambda_values, mu_values, time_intervals):
    """可视化决策分析结果"""
    print("生成决策分析可视化...")

    # 计算μ - λ的差值
    mu_lambda_diff = [mu - lam for mu, lam in zip(mu_values, lambda_values)]

    plt.figure(figsize=(14, 8))

    plt.plot(time_intervals, mu_lambda_diff, marker='o',
             label='μ - λ (2013年10月22日)', color='blue', linewidth=2, markersize=6)

    # 添加零线
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='平衡线 (μ = λ)')

    plt.xlabel('时间段', fontsize=12)
    plt.ylabel('μ - λ', fontsize=12)
    plt.title('服务率与到达率差值分析', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 保存图片
    plt.tight_layout()
    plt.savefig('results/figures/service_arrival_rate_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    print("可视化图表已保存到: results/figures/service_arrival_rate_analysis.png")


def analyze_decision_patterns(results):
    """分析决策模式"""
    print("分析决策模式...")

    results_df = pd.DataFrame(results)

    # 统计决策分布
    decision_counts = results_df['司机决策'].value_counts()

    print("\n决策策略分布：")
    for decision, count in decision_counts.items():
        percentage = count / len(results_df) * 100
        print(f"  {decision}: {count} 个时段 ({percentage:.1f}%)")

    # 分析高峰时段决策
    print("\n按时间段分析：")
    for _, row in results_df.iterrows():
        hour = int(row['时间段'].split(':')[0])
        if hour in [8, 9, 17, 18, 19, 20]:  # 高峰时段
            print(f"  {row['时间段']}: {row['司机决策']} "
                  f"(R_A={row['接客收益期望R_A']:.2f}, R_B={row['不接客收益期望R_B']:.2f})")


def main():
    """
    主函数：执行完整的决策建模流程
    """
    print("=" * 60)
    print("问题二：出租车司机决策建模与验证")
    print("=" * 60)

    try:
        # 步骤1：设置环境
        setup_environment()

        # 步骤2：加载预处理数据
        gps_data, flight_data = load_processed_data()

        # 步骤3：预处理日期时间
        gps_data, flight_data = preprocess_datetime_columns(gps_data, flight_data)

        # 步骤4：计算到达离开数据
        arrival_departure_times = calculate_arrival_departure_data(gps_data)

        # 步骤5：设置模型参数
        params = setup_model_parameters()

        # 步骤6：计算决策模型
        results, lambda_values, mu_values, time_intervals = calculate_decision_model(
            arrival_departure_times, flight_data, params)

        # 步骤7：保存结果
        output_file = save_decision_results(results)

        # 步骤8：可视化分析
        visualize_decision_analysis(lambda_values, mu_values, time_intervals)

        # 步骤9：分析决策模式
        analyze_decision_patterns(results)

        # 输出建模摘要
        print("\n" + "=" * 50)
        print("决策建模摘要：")
        print("=" * 50)
        print(f"分析时间段数：{len(results)}")
        print(f"建模基础数据：{len(arrival_departure_times)} 条到达离开记录")
        print(f"结果文件：{output_file}")
        print("模型基于M/M/1排队论和收益期望理论")

        print("\n决策建模与验证完成！")

    except Exception as e:
        print(f"程序执行出错：{e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)