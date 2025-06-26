# -*- coding: utf-8 -*-
"""
问题四：短途车优先权分配系统
合并了优先权计算和可视化功能

功能：
1. 计算各行程的净收益
2. 确定短途车在长途车队列中的插队位置
3. 生成优先权分配方案
4. 可视化插队位置分布

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from pathlib import Path

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def setup_environment():
    """设置项目环境"""

    # 确保输出目录存在
    os.makedirs('../data/processed', exist_ok=True)
    os.makedirs('../results/figures', exist_ok=True)


def load_data():
    """
    加载数据文件
    """
    print("加载数据文件...")

    # 读取车辆行程数据
    trips_file = '../data/processed/Taxi_Trips_marked.xlsx'
    if not os.path.exists(trips_file):
        raise FileNotFoundError(f"行程数据文件不存在: {trips_file}")

    df_trips = pd.read_excel(trips_file, engine='openpyxl')

    # 读取等待时间成本数据
    cost_file = '../data/processed/Decision_Model_Results.xlsx'
    if not os.path.exists(cost_file):
        print(f"警告: 成本数据文件不存在: {cost_file}")
        print("将使用默认时间成本参数")
        df_cost = create_default_cost_data()
    else:
        try:
            df_cost = pd.read_excel(cost_file, engine='openpyxl')
            print(f"成功加载成本数据文件")
        except Exception as e:
            print(f"加载成本数据文件失败: {e}")
            print("将使用默认时间成本参数")
            df_cost = create_default_cost_data()

    print(f"加载行程数据：{len(df_trips)} 条记录")
    print(f"加载成本数据：{len(df_cost)} 条记录")

    return df_trips, df_cost


def create_default_cost_data():
    """创建默认的时间成本数据"""
    print("创建默认时间成本数据...")

    # 创建24小时的时间段数据
    time_intervals = []
    time_costs = []

    for hour in range(24):
        start_time = f"{hour:02d}:00:00"
        end_time = f"{(hour + 1) % 24:02d}:00:00"
        time_interval = f"{start_time} - {end_time}"
        time_intervals.append(time_interval)

        # 基于小时设置不同的时间成本
        if 6 <= hour < 23:  # 日间
            base_cost = 15.0
        else:  # 夜间
            base_cost = 10.0

        # 高峰时段成本更高
        if hour in [8, 9, 17, 18, 19]:
            time_cost = base_cost * 1.5
        else:
            time_cost = base_cost

        time_costs.append(time_cost)

    df_cost = pd.DataFrame({
        'Time Interval': time_intervals,
        'Time Cost C_w': time_costs
    })

    return df_cost


def check_cost_data_columns(df_cost):
    """检查成本数据的列名并进行调整"""
    print("检查成本数据列名...")
    print(f"成本数据列名: {list(df_cost.columns)}")

    # 寻找时间间隔列
    time_interval_col = None
    for col in df_cost.columns:
        if any(keyword in str(col).lower() for keyword in ['time', 'interval', '时间', '间隔']):
            time_interval_col = col
            break

    if time_interval_col is None:
        print("未找到时间间隔列，使用第一列作为时间间隔")
        time_interval_col = df_cost.columns[0]

    # 寻找时间成本列
    time_cost_col = None
    for col in df_cost.columns:
        if any(keyword in str(col).lower() for keyword in ['cost', 'c_w', '成本']):
            time_cost_col = col
            break

    if time_cost_col is None:
        print("未找到时间成本列，使用最后一列作为时间成本")
        time_cost_col = df_cost.columns[-1]

    # 重命名列以保持一致性
    df_cost = df_cost.rename(columns={
        time_interval_col: 'Time Interval',
        time_cost_col: 'Time Cost C_w'
    })

    print(f"使用列: '{time_interval_col}' 作为时间间隔")
    print(f"使用列: '{time_cost_col}' 作为时间成本")

    return df_cost


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
    try:
        return datetime.strptime(str(time_str).strip(), '%H:%M:%S').time()
    except:
        try:
            return datetime.strptime(str(time_str).strip(), '%H:%M').time()
        except:
            print(f"无法解析时间字符串: {time_str}")
            return datetime.strptime('00:00:00', '%H:%M:%S').time()


def parse_time_interval(time_interval_str):
    """
    将时间段解析为开始时间和结束时间
    """
    try:
        if ' - ' in str(time_interval_str):
            start_time_str, end_time_str = str(time_interval_str).split(' - ')
        elif '-' in str(time_interval_str):
            start_time_str, end_time_str = str(time_interval_str).split('-')
        else:
            # 如果无法解析，返回默认值
            return datetime.strptime('00:00:00', '%H:%M:%S').time(), datetime.strptime('01:00:00', '%H:%M:%S').time()

        start_time = parse_time_string(start_time_str)
        end_time = parse_time_string(end_time_str)
        return start_time, end_time
    except Exception as e:
        print(f"解析时间间隔失败: {time_interval_str}, 错误: {e}")
        return datetime.strptime('00:00:00', '%H:%M:%S').time(), datetime.strptime('01:00:00', '%H:%M:%S').time()


def prepare_cost_data(df_cost):
    """
    准备成本数据
    """
    print("处理时间成本数据...")

    # 检查并调整列名
    df_cost = check_cost_data_columns(df_cost)

    # 确保有必要的列
    if 'Time Interval' not in df_cost.columns:
        raise ValueError("成本数据中缺少时间间隔信息")

    if 'Time Cost C_w' not in df_cost.columns:
        print("成本数据中缺少时间成本信息，使用默认值")
        df_cost['Time Cost C_w'] = 15.0
    else:
        # 确保时间成本列是数值类型
        def convert_to_numeric(value):
            try:
                if pd.isna(value):
                    return 15.0
                elif isinstance(value, str):
                    # 如果是字符串，尝试提取数值
                    import re
                    numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', value)
                    return float(numbers[0]) if numbers else 15.0
                else:
                    return float(value)
            except (ValueError, TypeError):
                return 15.0

        df_cost['Time Cost C_w'] = df_cost['Time Cost C_w'].apply(convert_to_numeric)

    # 解析时间段并生成新的列
    try:
        parsed_times = df_cost['Time Interval'].apply(parse_time_interval)
        df_cost['Start Time'] = [t[0] for t in parsed_times]
        df_cost['End Time'] = [t[1] for t in parsed_times]
    except Exception as e:
        print(f"解析时间段失败: {e}")
        # 创建默认时间段
        df_cost['Start Time'] = [datetime.strptime(f"{i:02d}:00:00", '%H:%M:%S').time() for i in range(len(df_cost))]
        df_cost['End Time'] = [datetime.strptime(f"{(i + 1) % 24:02d}:00:00", '%H:%M:%S').time() for i in
                               range(len(df_cost))]

    return df_cost


def check_trips_data_columns(df_trips):
    """检查行程数据的必要列"""
    print("检查行程数据列名...")
    print(f"行程数据列名: {list(df_trips.columns)}")

    required_columns = ['距离(km)', '离开时间', '分类']
    missing_columns = []

    for col in required_columns:
        if col not in df_trips.columns:
            missing_columns.append(col)

    if missing_columns:
        print(f"警告: 行程数据缺少以下列: {missing_columns}")

        # 尝试寻找相似的列名
        for missing_col in missing_columns:
            if missing_col == '距离(km)':
                # 寻找距离相关的列
                for col in df_trips.columns:
                    if any(keyword in str(col).lower() for keyword in ['距离', 'distance', 'km']):
                        df_trips = df_trips.rename(columns={col: '距离(km)'})
                        print(f"找到距离列: '{col}' -> '距离(km)'")
                        break

            elif missing_col == '离开时间':
                # 寻找时间相关的列
                for col in df_trips.columns:
                    if any(keyword in str(col).lower() for keyword in ['时间', 'time', '离开', '结束', '开始']):
                        df_trips = df_trips.rename(columns={col: '离开时间'})
                        print(f"找到时间列: '{col}' -> '离开时间'")
                        break

            elif missing_col == '分类':
                # 寻找分类相关的列
                for col in df_trips.columns:
                    if any(keyword in str(col).lower() for keyword in ['分类', 'class', 'category', '类型']):
                        df_trips = df_trips.rename(columns={col: '分类'})
                        print(f"找到分类列: '{col}' -> '分类'")
                        break

    # 如果仍然缺少必要列，创建默认值
    if '距离(km)' not in df_trips.columns:
        print("创建默认距离列")
        df_trips['距离(km)'] = np.random.uniform(1, 20, len(df_trips))  # 随机距离1-20km

    if '离开时间' not in df_trips.columns:
        print("创建默认时间列")
        df_trips['离开时间'] = pd.to_datetime('2013-10-22 12:00:00')  # 默认时间

    if '分类' not in df_trips.columns:
        print("创建默认分类列")
        # 基于距离进行分类
        df_trips['分类'] = df_trips['距离(km)'].apply(lambda x: '短途' if x <= 9.27 else '长途')

    # 确保数据类型正确
    try:
        # 确保距离列是数值类型
        df_trips['距离(km)'] = pd.to_numeric(df_trips['距离(km)'], errors='coerce')
        df_trips['距离(km)'] = df_trips['距离(km)'].fillna(5.0)  # 用默认值填充NaN

        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df_trips['离开时间']):
            df_trips['离开时间'] = pd.to_datetime(df_trips['离开时间'], errors='coerce')
            df_trips['离开时间'] = df_trips['离开时间'].fillna(pd.to_datetime('2013-10-22 12:00:00'))

        # 确保分类列是字符串类型
        df_trips['分类'] = df_trips['分类'].astype(str)

        print(f"数据类型检查完成:")
        print(f"  距离列类型: {df_trips['距离(km)'].dtype}")
        print(f"  时间列类型: {df_trips['离开时间'].dtype}")
        print(f"  分类列类型: {df_trips['分类'].dtype}")

    except Exception as e:
        print(f"数据类型转换失败: {e}")

    return df_trips


def calculate_net_profit(row, df_cost, params):
    """
    计算时间段内的净收益
    """
    try:
        # 确保距离是数值类型
        distance = float(row['距离(km)']) if not pd.isna(row['距离(km)']) else 5.0
    except:
        distance = 5.0  # 默认距离

    # 处理时间数据
    if pd.isna(row['离开时间']):
        time_of_day = datetime.strptime('12:00:00', '%H:%M:%S').time()
    else:
        if isinstance(row['离开时间'], str):
            try:
                time_of_day = datetime.strptime(row['离开时间'], '%H:%M:%S').time()
            except:
                try:
                    time_of_day = pd.to_datetime(row['离开时间']).time()
                except:
                    time_of_day = datetime.strptime('12:00:00', '%H:%M:%S').time()
        else:
            try:
                time_of_day = row['离开时间'].time()
            except:
                time_of_day = datetime.strptime('12:00:00', '%H:%M:%S').time()

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
        time_cost_raw = time_cost_matches[0]
        # 确保时间成本是数值类型
        try:
            time_cost = float(time_cost_raw) if not pd.isna(time_cost_raw) else 15.0
        except (ValueError, TypeError):
            # 如果是字符串，尝试提取数值
            if isinstance(time_cost_raw, str):
                import re
                numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', time_cost_raw)
                time_cost = float(numbers[0]) if numbers else 15.0
            else:
                time_cost = 15.0
    else:
        time_cost = 15.0  # 默认时间成本

    # 确保所有变量都是数值类型
    try:
        revenue = float(revenue)
        fuel_cost = float(fuel_cost)
        time_cost = float(time_cost)

        # 计算净收益
        net_profit = revenue - fuel_cost - time_cost
        return float(net_profit)
    except (ValueError, TypeError) as e:
        print(f"计算净收益时出错: {e}")
        print(f"revenue: {revenue}, fuel_cost: {fuel_cost}, time_cost: {time_cost}")
        return 0.0  # 返回默认值


def process_trip_data(df_trips, df_cost, params):
    """
    处理行程数据，计算净收益
    """
    print("计算各行程净收益...")

    # 检查行程数据列
    df_trips = check_trips_data_columns(df_trips)

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

    if len(df_long_trips) == 0:
        print("警告: 没有长途行程数据")
        insertion_df = pd.DataFrame({
            '距离(km)': df_short_trips['距离(km)'],
            '净收益': df_short_trips['净收益'],
            '插队位置': ['前'] * len(df_short_trips)
        })
        return insertion_df, pd.DataFrame()

    # 对长途行程按净收益排序，计算排序后的队列位置
    df_long_trips_sorted = df_long_trips.sort_values(
        by='净收益', ascending=False).reset_index(drop=True)

    def get_insertion_position(short_trip_net_profit):
        """计算插队位置"""
        if len(df_long_trips_sorted) == 0:
            return '前'

        # 计算插队位置
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
    output_file = 'data/processed/短途车插队信息.xlsx'
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

    if len(df_long_trips_sorted) > 0:
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
    else:
        plt.subplot(1, 1, 1)

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
        # 步骤1：设置环境
        setup_environment()

        # 步骤2：加载数据
        df_trips, df_cost = load_data()

        # 步骤3：设置参数
        params = setup_parameters()

        # 步骤4：准备成本数据
        df_cost = prepare_cost_data(df_cost)

        # 步骤5：计算净收益，分类行程
        df_short_trips, df_long_trips = process_trip_data(df_trips, df_cost, params)

        # 步骤6：计算插队位置
        insertion_df, df_long_trips_sorted = calculate_insertion_positions(
            df_short_trips, df_long_trips)

        # 步骤7：保存结果
        output_file = save_insertion_results(insertion_df)

        # 步骤8：可视化分析
        visualize_insertion_analysis(insertion_df, df_long_trips_sorted)

        # 步骤9：输出结果摘要
        print_insertion_results(insertion_df)

        # 输出总结信息
        print("\n" + "=" * 50)
        print("优先权分配系统总结：")
        print("=" * 50)
        print(f"短途行程数量：{len(df_short_trips)}")
        print(f"长途行程数量：{len(df_long_trips)}")
        if len(df_short_trips) > 0:
            print(f"平均短途净收益：{df_short_trips['净收益'].mean():.2f} 元")
        if len(df_long_trips) > 0:
            print(f"平均长途净收益：{df_long_trips['净收益'].mean():.2f} 元")
        print(f"结果已保存到：{output_file}")

        print("\n短途车优先权分配系统分析完成！")

    except Exception as e:
        print(f"程序执行出错：{e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)