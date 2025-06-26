# -*- coding: utf-8 -*-
"""
问题二：出租车GPS数据预处理

功能：
1. 读取原始出租车GPS数据
2. 筛选机场范围内的数据
3. 检测车辆到达和离开机场的时间
4. 生成预处理后的数据文件

"""

import pandas as pd
import os
from pathlib import Path


def setup_environment():
    """设置项目环境"""
    # 确保工作目录在项目根目录
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # 确保输出目录存在
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)


def load_raw_data():
    """加载原始GPS数据"""
    print("加载原始出租车GPS数据...")

    # 原始数据文件路径
    file_path = 'data/raw/taxi_gps_data.txt'

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    # 读取TXT文件，无标题行
    df = pd.read_csv(file_path, sep=',', header=None)
    print(f"成功加载数据，共 {len(df)} 条记录")

    return df


def preprocess_data(df):
    """预处理GPS数据"""
    print("开始数据预处理...")

    # 定义要添加的日期部分（对应研究日期：2013年10月22日）
    date_str = '2013-10-22'

    # 将第二列（时间数据）转换为datetime对象，并添加日期
    df[1] = pd.to_datetime(date_str + ' ' + df[1], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # 转换经纬度列的数据类型为浮点数，并处理无效值
    df[2] = pd.to_numeric(df[2], errors='coerce')  # 经度列
    df[3] = pd.to_numeric(df[3], errors='coerce')  # 纬度列

    # 删除经纬度或时间中包含无效值的行
    original_count = len(df)
    df = df.dropna(subset=[1, 2, 3])
    print(f"删除无效数据 {original_count - len(df)} 条，剩余 {len(df)} 条")

    return df


def filter_airport_area(df):
    """筛选机场范围内的数据"""
    print("筛选机场范围内的GPS数据...")

    # 深圳宝安国际机场经纬度范围
    airport_bounds = {
        'lon_min': 113.77003,
        'lon_max': 113.83039,
        'lat_min': 22.61773,
        'lat_max': 22.66807
    }

    # 筛选出经纬度在指定范围内的样本
    df_filtered = df[
        (df[2] >= airport_bounds['lon_min']) &
        (df[2] <= airport_bounds['lon_max']) &
        (df[3] >= airport_bounds['lat_min']) &
        (df[3] <= airport_bounds['lat_max'])
        ]

    print(f"机场范围内数据：{len(df_filtered)} 条")

    return df_filtered


def filter_passenger_status(df_filtered):
    """筛选载客状态的数据"""
    print("筛选载客状态数据...")

    # 筛选出第五列为1的样本（载客状态）
    df_filtered = df_filtered[df_filtered[4] == 1]
    print(f"载客状态数据：{len(df_filtered)} 条")

    # 按车牌号分组并按时间排序
    df_filtered = df_filtered.sort_values(by=[0, 1])

    return df_filtered


def detect_arrival_departure(group):
    """
    检测车辆的进入和离开时间

    Args:
        group: 按车牌分组的数据

    Returns:
        DataFrame: 包含到达和离开时间的数据
    """
    arrival_times = []
    departure_times = []

    in_airport = False
    last_time = None
    last_index = None

    for index, row in group.iterrows():
        current_time = row[1]

        if not in_airport:  # 检测到达机场
            arrival_times.append(index)
            in_airport = True
        elif in_airport and (current_time - last_time).seconds > 1200:  # 检测离开机场（20分钟无活动）
            departure_times.append(last_index)
            arrival_times.append(index)  # 记录下一次的到达时间

        last_time = current_time
        last_index = index

    if in_airport:  # 如果最后一条记录仍然在机场范围内
        departure_times.append(last_index)

    # 选择到达和离开时间对应的记录
    selected_rows = group.loc[arrival_times + departure_times].sort_index()
    return selected_rows


def process_arrival_departure(df_filtered):
    """处理到达离开时间"""
    print("检测车辆到达和离开时间...")

    # 应用函数到每个分组
    df_results = df_filtered.groupby(0).apply(detect_arrival_departure).reset_index(drop=True)

    print(f"生成到达离开数据：{len(df_results)} 条记录")

    return df_results


def save_results(df_results):
    """保存处理结果"""
    print("保存处理结果...")

    # 保存筛选后的数据到新的Excel文件
    output_file_path = 'data/processed/出租车到离时间结果.xlsx'
    df_results.to_excel(output_file_path, index=False)

    print(f"数据预处理完成，结果已保存到: {output_file_path}")

    return output_file_path


def main():
    """
    主函数：执行完整的数据预处理流程
    """
    print("=" * 60)
    print("问题二：出租车GPS数据预处理")
    print("=" * 60)

    try:
        # 步骤1：设置环境
        setup_environment()

        # 步骤2：加载原始数据
        df = load_raw_data()

        # 步骤3：数据预处理
        df = preprocess_data(df)

        # 步骤4：筛选机场范围数据
        df_filtered = filter_airport_area(df)

        # 步骤5：筛选载客状态数据
        df_filtered = filter_passenger_status(df_filtered)

        # 步骤6：处理到达离开时间
        df_results = process_arrival_departure(df_filtered)

        # 步骤7：保存结果
        output_file = save_results(df_results)

        # 输出处理摘要
        print("\n" + "=" * 40)
        print("数据预处理摘要：")
        print("=" * 40)
        print(f"输入文件：data/raw/taxi_gps_data.txt")
        print(f"输出文件：{output_file}")
        print(f"最终数据条数：{len(df_results)}")
        print(f"涉及车辆数：{df_results[0].nunique()}")

        print("\n数据预处理完成！")

    except Exception as e:
        print(f"程序执行出错：{e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)