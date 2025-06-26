# -*- coding: utf-8 -*-
"""
问题四：出租车行程热力图生成

功能：
1. 读取出租车行程起终点数据
2. 生成行程分布热力图
3. 可视化出租车服务覆盖范围

"""

import pandas as pd
import os
import math
from pathlib import Path


def setup_environment():
    """设置项目环境 - 简化版本，不改变工作目录"""
    # 确保输出目录存在
    os.makedirs('../results/figures', exist_ok=True)
    print("输出目录已准备就绪")


def load_trip_data():
    """加载出租车行程数据"""
    print("加载出租车行程数据...")

    # 数据文件路径
    file_path = '../data/processed/Taxi_Trips.csv'

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"数据文件不存在: {file_path}\n"
            f"请确保数据文件位置正确"
        )

    # 读取CSV文件，指定列名（因为你的文件没有表头）
    column_names = ['出租车ID', '开始时间', '结束时间', '起始经度', '起始纬度', '终止经度', '终止纬度']
    trip_df = pd.read_csv(file_path, names=column_names, header=None)

    print(f"成功加载行程数据，共 {len(trip_df)} 条行程记录")
    print(f"数据列名: {list(trip_df.columns)}")

    # 显示前几行数据
    print("数据示例:")
    print(trip_df.head())

    return trip_df


def validate_trip_data(trip_df):
    """验证行程数据的有效性"""
    print("验证行程数据有效性...")

    required_columns = ['起始经度', '起始纬度', '终止经度', '终止纬度']
    missing_columns = [col for col in required_columns if col not in trip_df.columns]

    if missing_columns:
        raise ValueError(f"数据文件缺少必要列: {missing_columns}")

    # 检查坐标范围
    coord_stats = {
        '起始经度': (trip_df['起始经度'].min(), trip_df['起始经度'].max()),
        '起始纬度': (trip_df['起始纬度'].min(), trip_df['起始纬度'].max()),
        '终止经度': (trip_df['终止经度'].min(), trip_df['终止经度'].max()),
        '终止纬度': (trip_df['终止纬度'].min(), trip_df['终止纬度'].max())
    }

    print("坐标范围统计：")
    for coord_type, (min_val, max_val) in coord_stats.items():
        print(f"{coord_type}: {min_val:.6f} ~ {max_val:.6f}")

    # 检查缺失值
    missing_coords = trip_df[required_columns].isnull().sum().sum()
    if missing_coords > 0:
        print(f"警告：发现 {missing_coords} 个缺失的坐标值")
        trip_df = trip_df.dropna(subset=required_columns)
        print(f"清理后剩余 {len(trip_df)} 条记录")

    # 检查坐标是否合理（基于珠海地区的大致范围）
    # 珠海经度大约在113-114之间，纬度大约在22-23之间
    valid_coords = (
            (trip_df['起始经度'] > 110) & (trip_df['起始经度'] < 120) &
            (trip_df['起始纬度'] > 20) & (trip_df['起始纬度'] < 25) &
            (trip_df['终止经度'] > 110) & (trip_df['终止经度'] < 120) &
            (trip_df['终止纬度'] > 20) & (trip_df['终止纬度'] < 25)
    )

    invalid_count = len(trip_df) - valid_coords.sum()
    if invalid_count > 0:
        print(f"警告：发现 {invalid_count} 条可能异常的坐标记录")
        print("保留所有数据用于分析")

    return trip_df


def extract_coordinates(trip_df):
    """提取所有坐标点（起点和终点）"""
    print("提取行程起终点坐标...")

    # 提取起点和终点的经纬度数据
    start_points = trip_df[['起始纬度', '起始经度']].values
    end_points = trip_df[['终止纬度', '终止经度']].values

    # 合并所有坐标点
    latitudes = trip_df['起始纬度'].tolist() + trip_df['终止纬度'].tolist()
    longitudes = trip_df['起始经度'].tolist() + trip_df['终止经度'].tolist()

    print(f"提取到 {len(latitudes)} 个坐标点（包含起点和终点）")

    return latitudes, longitudes, start_points, end_points


def generate_trip_heatmap(latitudes, longitudes):
    """生成行程热力图"""
    print("生成行程分布热力图...")

    try:
        from gmplot import gmplot
    except ImportError:
        raise ImportError(
            "gmplot库未安装。请运行: pip install gmplot"
        )

    # 计算地图中心点（基于数据的中心位置）
    center_lat = sum(latitudes) / len(latitudes)
    center_lon = sum(longitudes) / len(longitudes)
    zoom_level = 11  # 适合城市级别的缩放

    print(f"地图中心: ({center_lat:.6f}, {center_lon:.6f})")
    print(f"缩放级别: {zoom_level}")

    # 创建gmplot对象
    gmap = gmplot.GoogleMapPlotter(center_lat, center_lon, zoom_level)

    # 绘制热力图
    # 设置热力图参数：半径、透明度
    gmap.heatmap(latitudes, longitudes, radius=25, opacity=0.7)

    # 保存热力图到HTML文件
    output_file = '../results/figures/trip_distribution_heatmap.html'
    gmap.draw(output_file)

    print(f"行程热力图已保存到: {output_file}")

    return output_file


def haversine_distance(lat1, lon1, lat2, lon2):
    """计算两点间的直线距离（公里）"""
    R = 6371  # 地球半径（公里）

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    return R * c


def generate_trip_statistics(trip_df, latitudes, longitudes):
    """生成行程统计信息"""
    print("生成行程统计信息...")

    # 计算每个行程的距离
    distances = []
    for _, row in trip_df.iterrows():
        dist = haversine_distance(
            row['起始纬度'], row['起始经度'],
            row['终止纬度'], row['终止经度']
        )
        distances.append(dist)

    trip_df['距离(km)'] = distances

    stats = {
        '总行程数': len(trip_df),
        '总坐标点数': len(latitudes),
        '服务范围': {
            '经度范围': f"{min(longitudes):.6f} ~ {max(longitudes):.6f}",
            '纬度范围': f"{min(latitudes):.6f} ~ {max(latitudes):.6f}",
            '经度跨度': f"{max(longitudes) - min(longitudes):.6f}度",
            '纬度跨度': f"{max(latitudes) - min(latitudes):.6f}度"
        },
        '行程距离统计': {
            '平均距离': f"{trip_df['距离(km)'].mean():.2f} km",
            '最短距离': f"{trip_df['距离(km)'].min():.2f} km",
            '最长距离': f"{trip_df['距离(km)'].max():.2f} km",
            '距离中位数': f"{trip_df['距离(km)'].median():.2f} km"
        }
    }

    # 出租车统计
    if '出租车ID' in trip_df.columns:
        unique_taxis = trip_df['出租车ID'].nunique()
        trips_per_taxi = trip_df['出租车ID'].value_counts()
        stats['出租车统计'] = {
            '总出租车数': unique_taxis,
            '平均每车行程数': f"{len(trip_df) / unique_taxis:.1f}",
            '最多行程数': trips_per_taxi.max(),
            '最少行程数': trips_per_taxi.min()
        }

    return stats


def print_statistics(stats):
    """打印统计信息"""
    print("\n" + "=" * 50)
    print("行程数据统计摘要：")
    print("=" * 50)

    print(f"总行程数：{stats['总行程数']:,}")
    print(f"总坐标点数：{stats['总坐标点数']:,}")

    print(f"\n服务范围：")
    for key, value in stats['服务范围'].items():
        print(f"  {key}：{value}")

    if '行程距离统计' in stats:
        print(f"\n行程距离统计：")
        for key, value in stats['行程距离统计'].items():
            print(f"  {key}：{value}")

    if '出租车统计' in stats:
        print(f"\n出租车统计：")
        for key, value in stats['出租车统计'].items():
            print(f"  {key}：{value}")


def main():
    """
    主函数：执行完整的行程热力图生成流程
    """
    print("=" * 60)
    print("问题四：出租车行程热力图生成")
    print("=" * 60)

    try:
        # 步骤1：设置环境
        setup_environment()

        # 步骤2：加载行程数据
        trip_df = load_trip_data()

        # 步骤3：验证数据有效性
        trip_df = validate_trip_data(trip_df)

        # 步骤4：提取坐标数据
        latitudes, longitudes, start_points, end_points = extract_coordinates(trip_df)

        # 步骤5：生成统计信息
        stats = generate_trip_statistics(trip_df, latitudes, longitudes)
        print_statistics(stats)

        # 步骤6：生成热力图
        output_file = generate_trip_heatmap(latitudes, longitudes)

        print("\n" + "=" * 50)
        print("行程热力图生成完成！")
        print("=" * 50)
        print(f"输出文件：{output_file}")
        print("请在浏览器中打开HTML文件查看热力图")
        print("热力图显示了出租车服务的空间分布密度")

        print("\n行程热力图生成完成！")

    except Exception as e:
        print(f"程序执行出错：{e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)