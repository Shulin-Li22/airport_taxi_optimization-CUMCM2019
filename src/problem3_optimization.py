# -*- coding: utf-8 -*-
"""
问题三：机场上车点配置优化

功能：
1. 基于离散事件仿真建模乘客和出租车到达过程
2. 计算不同上车点数量下的总成本
3. 寻找最优上车点配置方案
4. 可视化成本分析结果

"""

import simpy
import random
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
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    # 设置中文字体和图形风格
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set(style="whitegrid")


def setup_cost_parameters():
    """设置成本参数"""
    print("设置仿真成本参数...")

    params = {
        'taxi_waiting_cost_per_hour': 31.7,  # 出租车单位时间等待成本（元/小时）
        'pickup_point_cost_per_hour': 13.87,  # 每个上车点的单位时间使用成本（元/小时）
        'service_time': 5,  # 服务时间（分钟）
        'simulation_time': 60,  # 仿真时间（分钟）
        'max_pickup_points': 20  # 最大上车点数量
    }

    print(f"出租车等待成本: {params['taxi_waiting_cost_per_hour']} 元/小时")
    print(f"上车点使用成本: {params['pickup_point_cost_per_hour']} 元/小时")

    return params


class PickupPointState:
    """上车点状态类"""

    def __init__(self):
        self.is_busy = False


def passenger_process(env, name, pickup_point, service_time):
    """
    乘客到达和服务过程

    Args:
        env: SimPy环境
        name: 乘客名称
        pickup_point: 上车点资源
        service_time: 服务时间
    """
    arrival_time = env.now
    with pickup_point.request() as request:
        yield request
        wait_time = env.now - arrival_time
        yield env.timeout(service_time)


def taxi_process(env, name, pickup_points, service_time, pickup_state):
    """
    出租车到达和服务过程

    Args:
        env: SimPy环境
        name: 出租车名称
        pickup_points: 上车点资源列表
        service_time: 服务时间
        pickup_state: 上车点状态列表
    """
    arrival_time = env.now

    # 寻找空闲的上车点
    for i in range(len(pickup_state)):
        if not pickup_state[i].is_busy:
            with pickup_points[i].request() as request:
                yield request
                pickup_state[i].is_busy = True
                wait_time = env.now - arrival_time
                yield env.timeout(service_time)
                pickup_state[i].is_busy = False
                return

    # 如果所有上车点都忙，使用最后一个
    with pickup_points[-1].request() as request:
        yield request
        wait_time = env.now - arrival_time
        yield env.timeout(service_time)


def setup_simulation(env, num_pickup_points, passenger_arrival_rate, taxi_arrival_rate, service_time):
    """
    设置仿真环境

    Args:
        env: SimPy环境
        num_pickup_points: 上车点数量
        passenger_arrival_rate: 乘客到达率
        taxi_arrival_rate: 出租车到达率
        service_time: 服务时间

    Returns:
        list: 上车点资源列表
    """
    # 创建上车点资源和状态
    pickup_points = [simpy.Resource(env, 1) for _ in range(num_pickup_points)]
    pickup_state = [PickupPointState() for _ in range(num_pickup_points)]

    counter = 0

    def passenger_arrival_generator():
        """乘客到达生成器"""
        nonlocal counter
        while True:
            yield env.timeout(random.expovariate(passenger_arrival_rate))
            counter += 1
            env.process(passenger_process(
                env, f'乘客{counter}',
                pickup_points[random.randint(0, num_pickup_points - 1)],
                service_time
            ))

    def taxi_arrival_generator():
        """出租车到达生成器"""
        nonlocal counter
        while True:
            yield env.timeout(random.expovariate(taxi_arrival_rate))
            counter += 1
            env.process(taxi_process(
                env, f'出租车{counter}',
                pickup_points, service_time, pickup_state
            ))

    # 启动到达过程
    env.process(passenger_arrival_generator())
    env.process(taxi_arrival_generator())

    return pickup_points


def run_single_simulation(num_pickup_points, passenger_arrival_rate, taxi_arrival_rate,
                          service_time, simulation_time, cost_params):
    """
    运行单次仿真

    Args:
        num_pickup_points: 上车点数量
        passenger_arrival_rate: 乘客到达率
        taxi_arrival_rate: 出租车到达率
        service_time: 服务时间
        simulation_time: 仿真时间
        cost_params: 成本参数

    Returns:
        float: 总成本
    """
    env = simpy.Environment()
    pickup_points = setup_simulation(env, num_pickup_points, passenger_arrival_rate,
                                     taxi_arrival_rate, service_time)
    env.run(until=simulation_time)

    # 计算等待成本
    total_count = sum(pickup_point.count for pickup_point in pickup_points)
    total_queue = sum(len(pickup_point.queue) for pickup_point in pickup_points)

    avg_wait_time = total_queue / total_count if total_count > 0 else 0

    # 转换为小时并计算成本
    waiting_cost = (cost_params['taxi_waiting_cost_per_hour'] / 60) * avg_wait_time * simulation_time
    infrastructure_cost = num_pickup_points * cost_params['pickup_point_cost_per_hour'] * (simulation_time / 60)

    total_cost = waiting_cost + infrastructure_cost

    return total_cost


def load_simulation_data():
    """加载仿真所需的参数数据"""
    print("加载仿真参数数据...")

    # 尝试加载决策模型结果文件
    file_path = 'data/processed/Decision_Model_Results(3).xlsx'

    if not os.path.exists(file_path):
        print(f"警告: 参数文件 {file_path} 不存在")
        print("使用默认参数进行仿真...")

        # 创建默认参数数据
        default_data = {
            'passenger_arrival_rate': [0.1] * 24,  # 默认乘客到达率
            'taxi_arrival_rate': [0.08] * 24  # 默认出租车到达率
        }
        return pd.DataFrame(default_data)

    try:
        df = pd.read_excel(file_path)
        print(f"成功加载参数数据，共 {len(df)} 个时间段")
        return df
    except Exception as e:
        print(f"加载参数文件失败: {e}")
        print("使用默认参数...")

        default_data = {
            'passenger_arrival_rate': [0.1] * 24,
            'taxi_arrival_rate': [0.08] * 24
        }
        return pd.DataFrame(default_data)


def optimize_pickup_points(df, cost_params):
    """
    优化上车点数量

    Args:
        df: 包含到达率参数的数据框
        cost_params: 成本参数

    Returns:
        dict: 不同上车点数量对应的总成本
    """
    print("开始上车点优化仿真...")

    overall_costs = {}

    # 遍历所有可能的上车点数量
    for num_pickup_points in range(1, cost_params['max_pickup_points'] + 1):
        print(f"仿真 {num_pickup_points} 个上车点...")

        total_cost = 0

        # 遍历所有时间段
        for _, row in df.iterrows():
            # 获取到达率参数（根据数据文件结构调整）
            if len(row) >= 10:
                passenger_arrival_rate = row.iloc[9]  # 第10列
                taxi_arrival_rate = row.iloc[10]  # 第11列
            else:
                passenger_arrival_rate = 0.1  # 默认值
                taxi_arrival_rate = 0.08  # 默认值

            # 运行仿真
            period_cost = run_single_simulation(
                num_pickup_points, passenger_arrival_rate, taxi_arrival_rate,
                cost_params['service_time'], cost_params['simulation_time'], cost_params
            )

            total_cost += period_cost

        overall_costs[num_pickup_points] = total_cost
        print(f"  {num_pickup_points} 个上车点总成本: {total_cost:.2f} 元")

    return overall_costs


def find_optimal_solution(overall_costs):
    """
    寻找最优解

    Args:
        overall_costs: 成本字典

    Returns:
        tuple: (最优上车点数量, 最小总成本)
    """
    optimal_pickup_points = min(overall_costs, key=overall_costs.get)
    min_total_cost = overall_costs[optimal_pickup_points]

    return optimal_pickup_points, min_total_cost


def visualize_optimization_results(overall_costs, optimal_pickup_points, min_total_cost):
    """可视化优化结果"""
    print("生成优化结果可视化...")

    plt.figure(figsize=(12, 8))

    # 创建柱状图
    bars = plt.bar(overall_costs.keys(), overall_costs.values(),
                   color='brown', alpha=0.7, edgecolor='black')

    # 标记最优点
    plt.bar(optimal_pickup_points, min_total_cost,
            color='red', alpha=0.9, label=f'最优解: {optimal_pickup_points}个上车点')

    plt.xlabel('上车点数量', fontsize=12)
    plt.ylabel('总成本 (元)', fontsize=12)
    plt.title('不同上车点数量的总成本分析', fontsize=16, fontweight='bold')
    plt.xticks(range(1, len(overall_costs) + 1))
    plt.grid(axis='y', alpha=0.3)
    plt.legend()

    # 在柱子顶部添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.0f}', ha='center', va='bottom',
                 rotation=45, fontsize=8)

    # 保存图片
    plt.tight_layout()
    plt.savefig('results/figures/pickup_points_optimization.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    print("优化结果图表已保存到: results/figures/pickup_points_optimization.png")


def save_optimization_results(overall_costs, optimal_pickup_points, min_total_cost):
    """保存优化结果"""
    print("保存优化结果...")

    # 创建结果数据框
    results_df = pd.DataFrame({
        '上车点数量': list(overall_costs.keys()),
        '总成本(元)': list(overall_costs.values())
    })

    # 添加最优解标记
    results_df['是否最优'] = results_df['上车点数量'] == optimal_pickup_points

    # 保存到Excel
    output_file = 'data/processed/上车点优化结果.xlsx'
    results_df.to_excel(output_file, index=False)

    print(f"优化结果已保存到: {output_file}")

    return output_file


def print_optimization_summary(overall_costs, optimal_pickup_points, min_total_cost, cost_params):
    """打印优化摘要"""
    print("\n" + "=" * 50)
    print("上车点优化摘要：")
    print("=" * 50)

    print(f"仿真参数：")
    print(f"  服务时间: {cost_params['service_time']} 分钟")
    print(f"  仿真时长: {cost_params['simulation_time']} 分钟")
    print(f"  出租车等待成本: {cost_params['taxi_waiting_cost_per_hour']} 元/小时")
    print(f"  上车点使用成本: {cost_params['pickup_point_cost_per_hour']} 元/小时")

    print(f"\n优化结果：")
    print(f"  最优上车点数量: {optimal_pickup_points} 个")
    print(f"  最小总成本: {min_total_cost:.2f} 元")

    # 成本范围分析
    max_cost = max(overall_costs.values())
    cost_reduction = max_cost - min_total_cost
    reduction_percentage = (cost_reduction / max_cost) * 100

    print(f"\n成本分析：")
    print(f"  最高成本: {max_cost:.2f} 元 (1个上车点)")
    print(f"  成本降低: {cost_reduction:.2f} 元 ({reduction_percentage:.1f}%)")

    # 显示邻近解的成本
    print(f"\n邻近解比较：")
    for points in [optimal_pickup_points - 1, optimal_pickup_points, optimal_pickup_points + 1]:
        if points in overall_costs:
            cost = overall_costs[points]
            if points == optimal_pickup_points:
                print(f"  {points} 个上车点: {cost:.2f} 元 ← 最优解")
            else:
                diff = cost - min_total_cost
                print(f"  {points} 个上车点: {cost:.2f} 元 (+{diff:.2f})")


def analyze_cost_sensitivity(overall_costs):
    """分析成本敏感性"""
    print("\n成本敏感性分析：")

    costs_list = list(overall_costs.values())
    points_list = list(overall_costs.keys())

    # 计算成本变化率
    for i in range(1, len(costs_list)):
        cost_change = costs_list[i] - costs_list[i - 1]
        points_change = points_list[i] - points_list[i - 1]
        rate = cost_change / points_change

        if rate < 0:
            print(f"  增加第{points_list[i]}个上车点: 节省 {abs(rate):.2f} 元")
        else:
            print(f"  增加第{points_list[i]}个上车点: 增加 {rate:.2f} 元")


def main():
    """
    主函数：执行完整的上车点优化流程
    """
    print("=" * 60)
    print("问题三：机场上车点配置优化")
    print("=" * 60)

    try:
        # 步骤1：设置环境
        setup_environment()

        # 步骤2：设置成本参数
        cost_params = setup_cost_parameters()

        # 步骤3：加载仿真数据
        df = load_simulation_data()

        # 步骤4：运行优化仿真
        overall_costs = optimize_pickup_points(df, cost_params)

        # 步骤5：寻找最优解
        optimal_pickup_points, min_total_cost = find_optimal_solution(overall_costs)

        # 步骤6：可视化结果
        visualize_optimization_results(overall_costs, optimal_pickup_points, min_total_cost)

        # 步骤7：保存结果
        output_file = save_optimization_results(overall_costs, optimal_pickup_points, min_total_cost)

        # 步骤8：打印详细摘要
        print_optimization_summary(overall_costs, optimal_pickup_points, min_total_cost, cost_params)

        # 步骤9：敏感性分析
        analyze_cost_sensitivity(overall_costs)

        print(f"\n结果文件: {output_file}")
        print("\n上车点配置优化完成！")

        # 返回结果供其他程序使用
        return {
            'optimal_points': optimal_pickup_points,
            'min_cost': min_total_cost,
            'all_costs': overall_costs
        }

    except Exception as e:
        print(f"程序执行出错：{e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    result = main()
    if isinstance(result, dict):
        print(f"\n✅ 优化成功完成！最优配置：{result['optimal_points']} 个上车点")
        exit(0)
    else:
        exit(result)