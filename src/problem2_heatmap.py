# -*- coding: utf-8 -*-
"""
问题二：出租车GPS位置热力图生成

功能：
1. 读取载客出租车GPS数据
2. 生成机场区域GPS位置热力图
3. 可视化出租车活动密度分布

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
    os.makedirs('results/figures', exist_ok=True)


def load_passenger_taxi_data():
    """加载载客出租车筛选结果数据"""
    print("加载载客出租车GPS数据...")

    # 数据文件路径
    file_path = 'data/processed/出租车载人筛选结果.xlsx'

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"数据文件不存在: {file_path}\n"
            f"请先运行 problem2_data_preprocessing.py 生成此文件"
        )

    # 读取Excel文件
    df_filtered = pd.read_excel(file_path, header=None)

    # 设置列名
    df_filtered.columns = [
        'License Plate',  # 车牌号
        'Timestamp',  # 时间戳
        'Longitude',  # 经度
        'Latitude',  # 纬度
        'Passenger Status',  # 载客状态
        'Speed'  # 速度
    ]

    print(f"成功加载数据，共 {len(df_filtered)} 条GPS记录")

    return df_filtered


def validate_gps_data(df_filtered):
    """验证GPS数据的有效性"""
    print("验证GPS数据有效性...")

    # 检查经纬度范围（深圳宝安机场附近）
    lon_range = (df_filtered['Longitude'].min(), df_filtered['Longitude'].max())
    lat_range = (df_filtered['Latitude'].min(), df_filtered['Latitude'].max())

    print(f"经度范围: {lon_range[0]:.6f} ~ {lon_range[1]:.6f}")
    print(f"纬度范围: {lat_range[0]:.6f} ~ {lat_range[1]:.6f}")

    # 检查数据完整性
    missing_coords = df_filtered[['Longitude', 'Latitude']].isnull().sum().sum()
    if missing_coords > 0:
        print(f"警告：发现 {missing_coords} 个缺失的坐标值")
        df_filtered = df_filtered.dropna(subset=['Longitude', 'Latitude'])
        print(f"清理后剩余 {len(df_filtered)} 条记录")

    return df_filtered


def generate_heatmap(df_filtered):
    """生成GPS热力图"""
    print("生成GPS位置热力图...")

    try:
        from gmplot import gmplot
    except ImportError:
        raise ImportError(
            "gmplot库未安装。请运行: pip install gmplot"
        )

    # 计算地图中心点（深圳宝安国际机场）
    center_lat = 22.6434  # 机场中心纬度
    center_lon = 113.8  # 机场中心经度
    zoom_level = 13  # 地图缩放级别

    print(f"地图中心: ({center_lat}, {center_lon})")
    print(f"缩放级别: {zoom_level}")

    # 创建gmplot对象
    gmap = gmplot.GoogleMapPlotter(center_lat, center_lon, zoom_level)

    # 提取经纬度数据
    latitudes = df_filtered['Latitude'].tolist()
    longitudes = df_filtered['Longitude'].tolist()

    print(f"生成热力图，包含 {len(latitudes)} 个GPS点")

    # 绘制热力图
    gmap.heatmap(latitudes, longitudes, radius=20, opacity=0.6)

    # 保存热力图到HTML文件
    output_file = 'results/figures/taxi_gps_heatmap.html'
    gmap.draw(output_file)

    print(f"GPS热力图已保存到: {output_file}")

    return output_file


def generate_statistics(df_filtered):
    """生成统计信息"""
    print("生成GPS数据统计信息...")

    stats = {
        '总GPS记录数': len(df_filtered),
        '唯一车辆数': df_filtered['License Plate'].nunique(),
        '时间跨度': {
            '开始时间': df_filtered['Timestamp'].min(),
            '结束时间': df_filtered['Timestamp'].max()
        },
        '空间分布': {
            '经度范围': f"{df_filtered['Longitude'].min():.6f} ~ {df_filtered['Longitude'].max():.6f}",
            '纬度范围': f"{df_filtered['Latitude'].min():.6f} ~ {df_filtered['Latitude'].max():.6f}"
        }
    }

    return stats


def print_statistics(stats):
    """打印统计信息"""
    print("\n" + "=" * 40)
    print("GPS数据统计摘要：")
    print("=" * 40)
    print(f"总GPS记录数：{stats['总GPS记录数']:,}")
    print(f"涉及车辆数：{stats['唯一车辆数']:,}")
    print(f"时间跨度：{stats['时间跨度']['开始时间']} ~ {stats['时间跨度']['结束时间']}")
    print(f"经度范围：{stats['空间分布']['经度范围']}")
    print(f"纬度范围：{stats['空间分布']['纬度范围']}")


def main():
    """
    主函数：执行完整的热力图生成流程
    """
    print("=" * 60)
    print("问题二：出租车GPS位置热力图生成")
    print("=" * 60)

    try:
        # 步骤1：设置环境
        setup_environment()

        # 步骤2：加载载客出租车数据
        df_filtered = load_passenger_taxi_data()

        # 步骤3：验证数据有效性
        df_filtered = validate_gps_data(df_filtered)

        # 步骤4：生成统计信息
        stats = generate_statistics(df_filtered)
        print_statistics(stats)

        # 步骤5：生成热力图
        output_file = generate_heatmap(df_filtered)

        print("\n" + "=" * 40)
        print("热力图生成完成！")
        print("=" * 40)
        print(f"输出文件：{output_file}")
        print("请在浏览器中打开HTML文件查看热力图")

        print("\nGPS热力图生成完成！")

    except Exception as e:
        print(f"程序执行出错：{e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)