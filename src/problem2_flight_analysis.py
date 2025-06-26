
# -*- coding: utf-8 -*-
"""
问题二：航班数据分析

功能：
1. 筛选和统计航班数据
2. 绘制航班密度分布图
3. 分析航班随时间变化趋势

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def process_flight_data():
    """
    处理航班数据：筛选、清洗和统计
    """
    print("开始处理航班数据...")

    # 读取Excel文件
    file_path = '../data/raw/flight_data.xlsx'
    df = pd.read_excel(file_path)

    # 设置日期前缀（对应2013年10月22日的数据）
    date_prefix = '2013-10-22 '

    # 删除'计划到达时间'列中包含缺失值的行
    df = df.dropna(subset=['计划到达时间'])

    # 将时间列转换为完整的日期时间格式
    df['计划到达时间'] = pd.to_datetime(date_prefix + df['计划到达时间'].astype(str),
                                        format='%Y-%m-%d %H:%M:%S')

    # 同时删除出发地/经停点列中包含缺失值的行
    df = df.dropna(subset=['出发地/经停点'])

    # 删除重复的行（同一时间段共享航班的情况）
    df_unique = df.drop_duplicates(subset=['计划到达时间', '出发地/经停点'])

    # 将数据按10分钟间隔进行重采样，并统计每个间隔内的航班数量
    flight_count = df_unique.set_index('计划到达时间').resample('10T').size().reset_index(name='航班数量')

    # 保存结果到新的Excel文件
    output_file_path = '../data/processed/机场航班统计结果.xlsx'
    flight_count.to_excel(output_file_path, index=False)

    print(f"航班统计结果已保存到 {output_file_path}")

    return flight_count


def visualize_flight_data(flight_count):
    """
    可视化航班数据
    """
    print("开始生成航班数据可视化图表...")

    # 设置图形风格
    sns.set(style="whitegrid")

    # 创建图形并设置大小
    plt.figure(figsize=(14, 8))

    # 绘制密度图与直方图结合的可视化
    sns.histplot(flight_count['航班数量'], kde=True, color='blue', bins=20, label='Flight Count Distribution')

    # 设置标签和标题
    plt.xlabel('Number of Flights')
    plt.ylabel('Density')
    plt.title('Density and Histogram of Flights Every 10 Minutes')
    plt.legend()

    # 保存图片
    plt.savefig('../results/figures/flight_density_histogram.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    # 创建时间段的索引（Time Interval 1, 2, 3, ...）
    time_intervals = range(1, len(flight_count) + 1)

    # 打印 time interval 对应的时间段
    print("\n时间间隔对应表：")
    for interval, time in zip(time_intervals, flight_count['计划到达时间']):
        print(f"Time Interval {interval}: {time}")

    # 绘制折线图（不包括5-point-moving-average线）
    plt.figure(figsize=(24, 8))
    plt.plot(time_intervals, flight_count['航班数量'], marker='o', color='green', label='Flight Count (Line)')

    # 设置x轴为 Time Interval
    plt.xticks(time_intervals, labels=time_intervals, rotation=45)

    # 设置标签和标题
    plt.xlabel('Time Interval')
    plt.ylabel('Number of Flights')
    plt.title('Number of Flights Over Time')
    plt.legend()

    # 保存图片
    plt.savefig('../results/figures/flight_time_series.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    print("航班数据可视化完成")


def main():
    """
    主函数：执行完整的航班数据分析流程
    """
    print("=" * 60)
    print("问题二：航班数据分析")
    print("=" * 60)

    try:
        # 步骤1：处理航班数据
        flight_count = process_flight_data()

        # 步骤2：数据可视化
        visualize_flight_data(flight_count)

        # 输出基本统计信息
        print("\n航班统计摘要：")
        print(f"总时间间隔数：{len(flight_count)}")
        print(f"平均每10分钟航班数：{flight_count['航班数量'].mean():.2f}")
        print(f"最大航班密度：{flight_count['航班数量'].max()} 班次/10分钟")
        print(f"最小航班密度：{flight_count['航班数量'].min()} 班次/10分钟")

        print("\n航班数据分析完成！")

    except Exception as e:
        print(f"程序执行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()