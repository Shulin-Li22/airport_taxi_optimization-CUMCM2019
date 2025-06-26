# -*- coding: utf-8 -*-
"""
问题二：载客出租车数据筛选

功能：
1. 从原始GPS数据中筛选载客状态的出租车
2. 筛选机场范围内的GPS数据
3. 生成载客出租车位置数据供热力图使用

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


def load_raw_gps_data():
    """加载原始GPS数据"""
    print("加载原始出租车GPS数据...")

    file_path = 'data/raw/taxi_gps_data.txt'

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    # 读取TXT文件，无标题行
    df = pd.read_csv(file_path, sep=',', header=None)
    print(f"成功加载数据，共 {len(df)} 条记录")

    return df


def preprocess_gps_data(df):
    """预处理GPS数据"""
    print("预处理GPS数据...")

    # 定义要添加的日期部分
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
    """筛选载客状态的出租车"""
    print("筛选载客状态的出租车...")

    # 筛选出第五列为1的样本（载客状态）
    passenger_taxis = df_filtered[df_filtered[4] == 1]

    print(f"载客状态数据：{len(passenger_taxis)} 条")
    print(f"涉及车辆：{passenger_taxis[0].nunique()} 辆")

    return passenger_taxis


def format_output_data(passenger_taxis):
    """格式化输出数据"""
    print("格式化输出数据...")

    # 重新排列列并设置标准列名
    # 原始列：[车牌, 时间, 经度, 纬度, 载客状态, 速度]
    output_data = passenger_taxis.copy()

    # 如果有第6列（速度），保留；如果没有，设为0
    if len(output_data.columns) < 6:
        output_data[5] = 0  # 添加速度列，默认为0

    return output_data


def generate_statistics(passenger_taxis):
    """生成统计信息"""
    print("生成载客数据统计...")

    stats = {
        '总记录数': len(passenger_taxis),
        '唯一车辆数': passenger_taxis[0].nunique(),
        '时间范围': {
            '开始时间': passenger_taxis[1].min(),
            '结束时间': passenger_taxis[1].max()
        },
        '空间分布': {
            '经度范围': f"{passenger_taxis[2].min():.6f} ~ {passenger_taxis[2].max():.6f}",
            '纬度范围': f"{passenger_taxis[3].min():.6f} ~ {passenger_taxis[3].max():.6f}"
        }
    }

    return stats


def save_results(output_data):
    """保存筛选结果"""
    print("保存载客出租车筛选结果...")

    # 保存筛选后的数据到Excel文件
    output_file_path = 'data/processed/出租车载人筛选结果.xlsx'
    output_data.to_excel(output_file_path, index=False, header=False)

    print(f"载客出租车数据已保存到: {output_file_path}")

    return output_file_path


def print_statistics(stats):
    """打印统计信息"""
    print("\n" + "=" * 50)
    print("载客出租车数据统计摘要：")
    print("=" * 50)
    print(f"总GPS记录数：{stats['总记录数']:,}")
    print(f"涉及车辆数：{stats['唯一车辆数']:,}")
    print(f"时间范围：{stats['时间范围']['开始时间']} ~ {stats['时间范围']['结束时间']}")
    print(f"经度范围：{stats['空间分布']['经度范围']}")
    print(f"纬度范围：{stats['空间分布']['纬度范围']}")


def main():
    """
    主函数：执行完整的载客出租车筛选流程
    """
    print("=" * 60)
    print("问题二：载客出租车数据筛选")
    print("=" * 60)

    try:
        # 步骤1：设置环境
        setup_environment()

        # 步骤2：加载原始GPS数据
        df = load_raw_gps_data()

        # 步骤3：预处理GPS数据
        df = preprocess_gps_data(df)

        # 步骤4：筛选机场范围数据
        df_airport = filter_airport_area(df)

        # 步骤5：筛选载客状态数据
        passenger_taxis = filter_passenger_status(df_airport)

        # 步骤6：格式化输出数据
        output_data = format_output_data(passenger_taxis)

        # 步骤7：生成统计信息
        stats = generate_statistics(passenger_taxis)
        print_statistics(stats)

        # 步骤8：保存结果
        output_file = save_results(output_data)

        print("\n" + "=" * 40)
        print("载客出租车筛选完成！")
        print("=" * 40)
        print(f"输入：data/raw/taxi_gps_data.txt")
        print(f"输出：{output_file}")
        print("此文件可用于热力图生成")

        print("\n载客出租车筛选完成！")

    except Exception as e:
        print(f"程序执行出错：{e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)