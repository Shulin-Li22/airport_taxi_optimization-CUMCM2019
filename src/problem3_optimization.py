import simpy
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 成本参数
C1_per_hour = 31.7  # 出租车单位时间等待成本（元/小时）
C2_per_hour = 13.87  # 每个上车点的单位时间使用成本（元/小时）


class PickupPointState:
    def __init__(self):
        self.is_busy = False


def passenger(env, name, pickup_point, service_time):
    arrival_time = env.now
    with pickup_point.request() as request:
        yield request
        wait_time = env.now - arrival_time
        yield env.timeout(service_time)


def taxi(env, name, pickup_points, service_time, pickup_state):
    arrival_time = env.now
    for i in range(len(pickup_state)):
        if not pickup_state[i].is_busy:
            with pickup_points[i].request() as request:
                yield request
                pickup_state[i].is_busy = True
                wait_time = env.now - arrival_time
                yield env.timeout(service_time)
                pickup_state[i].is_busy = False
                return
    with pickup_points[-1].request() as request:
        yield request
        wait_time = env.now - arrival_time
        yield env.timeout(service_time)


def setup(env, num_pickup_points, passenger_arrival_rate, taxi_arrival_rate, service_time):
    pickup_points = [simpy.Resource(env, 1) for _ in range(num_pickup_points)]
    pickup_state = [PickupPointState() for _ in range(num_pickup_points)]
    i = 0

    def passenger_arrival():
        nonlocal i
        while True:
            yield env.timeout(random.expovariate(passenger_arrival_rate))
            i += 1
            env.process(
                passenger(env, f'乘客 {i}', pickup_points[random.randint(0, num_pickup_points - 1)], service_time))

    def taxi_arrival():
        nonlocal i
        while True:
            yield env.timeout(random.expovariate(taxi_arrival_rate))
            i += 1
            env.process(taxi(env, f'出租车 {i}', pickup_points, service_time, pickup_state))

    env.process(passenger_arrival())
    env.process(taxi_arrival())

    return pickup_points


def run_simulation(num_pickup_points, passenger_arrival_rate, taxi_arrival_rate, service_time, simulation_time):
    env = simpy.Environment()
    pickup_points = setup(env, num_pickup_points, passenger_arrival_rate, taxi_arrival_rate, service_time)
    env.run(until=simulation_time)

    total_count = sum(pickup_point.count for pickup_point in pickup_points)
    total_queue = sum(len(pickup_point.queue) for pickup_point in pickup_points)

    avg_wait_time = total_queue / total_count if total_count > 0 else float('inf')
    C1_total = (C1_per_hour / 60) * avg_wait_time * simulation_time  # 每分钟成本
    total_cost = C1_total + (num_pickup_points * C2_per_hour * simulation_time)

    return total_cost


def process_excel(file_path):
    df = pd.read_excel(file_path)

    service_time = 5  # 服务时间（分钟）
    simulation_time = 60  # 仿真时间（分钟）

    overall_costs = {}

    # 遍历所有上车点数量
    for num_pickup_points in range(1, 21):
        total_cost = 0

        # 遍历所有时间段
        for _, row in df.iterrows():
            passenger_arrival_rate = row.iloc[9]  # 第10列
            taxi_arrival_rate = row.iloc[10]  # 第11列

            # 计算每个时间段的成本
            time_period_cost = run_simulation(num_pickup_points, passenger_arrival_rate, taxi_arrival_rate,
                                              service_time, simulation_time)
            total_cost += time_period_cost

        overall_costs[num_pickup_points] = total_cost

    return overall_costs


def overall_optimal(overall_costs):
    optimal_pickup_points = min(overall_costs, key=overall_costs.get)
    min_total_cost = overall_costs[optimal_pickup_points]

    return optimal_pickup_points, min_total_cost


# 读取 Excel 文件
file_path = '~/Downloads/Decision_Model_Results(3).xlsx'  # 替换为你的 Excel 文件路径
overall_costs = process_excel(file_path)

# 输出每个上车点数量对应的总成本
print("不同上车点数量对应的总成本:")
for num_pickup_points, cost in overall_costs.items():
    print(f'上车点数量: {num_pickup_points}, 总成本: {cost:.2f}')

# 计算总体最佳上车点数量和总成本
optimal_pickup_points, min_total_cost = overall_optimal(overall_costs)
print(f'\n总体最佳上车点数量: {optimal_pickup_points}')
print(f'总体最小总成本: {min_total_cost:.2f}')

# 设置图形风格
sns.set(style="whitegrid")

# 可视化不同上车点数量和对应的总成本
plt.figure(figsize=(12, 8))
bars = plt.bar(overall_costs.keys(), overall_costs.values(), color='Brown')
plt.xlabel('Number of Pickup Points')
plt.ylabel('Total Cost (Yuan)')
plt.title('Total Cost vs. Number of Pickup Points')
plt.xticks(range(1, 21))
plt.grid(axis='y')

# 标注条形顶部的成本值并旋转45度
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height,
             f'{height:.2f}', ha='center', va='bottom', rotation=45)

plt.show()
