import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置图形风格
sns.set(style="whitegrid")

# 读取处理后的GPS数据
gps_data_path = '~/Downloads/出租车到离时间结果.xlsx'
gps_data = pd.read_excel(gps_data_path)

# 读取航班数据
flight_data_path = '~/Downloads/8月19日机场航班统计结果.xlsx'
flight_data = pd.read_excel(flight_data_path)

# 确保日期列是datetime格式
gps_data['日期'] = pd.to_datetime(gps_data['日期'])
flight_data['计划到达时间'] = pd.to_datetime(flight_data['计划到达时间'])


# 定义检测到达和离开时间的函数
def detect_arrival_departure_times(group):
    arrival_times = []
    departure_times = []

    in_airport = False
    last_time = None

    for i in range(len(group)):
        current_time = group.iloc[i]['日期']
        if not in_airport:
            arrival_times.append(current_time)
            in_airport = True
        elif (current_time - last_time).total_seconds() > 1200:  # 超过20分钟
            departure_times.append(last_time)
            in_airport = False

        last_time = current_time

    if in_airport:
        departure_times.append(last_time)

    return pd.DataFrame({'车牌': group['车牌'].iloc[0], '到达时间': arrival_times, '离开时间': departure_times})


# 按车牌分组并检测每辆车的到达和离开时间
gps_data_sorted = gps_data.sort_values(by=['车牌', '日期'])
arrival_departure_times = gps_data_sorted.groupby('车牌').apply(detect_arrival_departure_times).reset_index(drop=True)

# 计算蓄车池车辆数 Q
Q = arrival_departure_times['车牌'].nunique()  # 统计唯一车牌数，即为蓄车池中的车辆数

# 假设每个航班的平均乘客数量 P
P = 200  # 这个值可以根据具体的需求调整

# 创建列表存储每20分钟的结果
lambda_values = []
mu_values = []
time_intervals = []
results = []

# 定义20分钟的时间间隔
start_time = pd.Timestamp(year=2013, month=10, day=22, hour=0, minute=0)
end_time = pd.Timestamp(year=2013, month=10, day=22, hour=23, minute=59)
time_delta = pd.Timedelta(minutes=60)

# 加载每小时的载客出租车数量，用于计算时间成本C_w
hourly_taxi_data = {
    0: 729369, 1: 522271, 2: 408140, 3: 310274, 4: 240980, 5: 246297,
    6: 336588, 7: 629110, 8: 1003028, 9: 939582, 10: 1015902, 11: 941575,
    12: 811618, 13: 903500, 14: 1129489, 15: 1131097, 16: 1024277, 17: 1086384,
    18: 989066, 19: 1122078, 20: 1135057, 21: 1162701, 22: 1173174, 23: 1009961
}
total_taxis = sum(hourly_taxi_data.values())  # 计算总出租车数

# 逐步处理每60分钟的时间段
current_time = start_time
while current_time <= end_time:
    next_time = current_time + time_delta
    print(f"--- Time Interval: {current_time} to {next_time} ---")

    # 获取当前时间段内到达和离开的车辆数
    arrivals_in_interval = arrival_departure_times[
        (arrival_departure_times['到达时间'] >= current_time) & (arrival_departure_times['到达时间'] < next_time)]
    departures_in_interval = arrival_departure_times[
        (arrival_departure_times['离开时间'] >= current_time) & (arrival_departure_times['离开时间'] < next_time)]

    # 计算当前时间段内的到达率 λ 和服务率 μ
    lambda_rate_interval = arrivals_in_interval.shape[0]  # 当前时间段内到达车辆数
    mu_rate_interval = departures_in_interval.shape[0]  # 当前时间段内离开车辆数

    # 计算乘客需求量 D
    flights_in_interval = flight_data[
        (flight_data['计划到达时间'] >= current_time) & (flight_data['计划到达时间'] < next_time)]
    D_interval = flights_in_interval['航班数量'].sum() * P  # 航班数量乘以每航班的平均乘客数量

    # 确定接客收益 R_c 根据深圳出租车行业规定的日间和夜间时间段
    if 6 <= current_time.hour < 23:  # 日间
        R_c = 99.1
    else:  # 夜间
        R_c = 128.83

    # 计算时间成本 C_w（基于每小时载客出租车数量的权重）
    hour = current_time.hour
    weight = hourly_taxi_data.get(hour, 0) / total_taxis
    time_value_per_hour = weight * 30  # 司机的时间价值（每小时收入）
    if mu_rate_interval > lambda_rate_interval:
        T_w_interval = 1 / (mu_rate_interval - lambda_rate_interval)
        C_w_interval = T_w_interval * time_value_per_hour
    else:
        T_w_interval = np.inf
        C_w_interval = np.inf

    # 应用收益期望模型
    R_s = 31.74  # 市区载客平均收益（每小时）
    C_v = 21.73  # 空载返回市区成本

    R_A_interval = (D_interval / (D_interval + Q)) * R_c - C_w_interval
    R_B_interval = R_s * T_w_interval - C_v

    # 司机的选择策略
    if R_A_interval > R_B_interval:
        decision = "Suggest queuing at the airport"
    else:
        decision = "Suggest returning to the city"

    # 打印 λ, μ, 和 D 的值
    print(f"Arrival rate λ: {lambda_rate_interval}, Service rate μ: {mu_rate_interval}, Demand D: {D_interval}")
    print(f"Average waiting time T_w: {T_w_interval} hours")
    print(f"Time cost C_w: {C_w_interval} USD")
    print(f"Expected revenue R_A: {R_A_interval} USD")
    print(f"Expected revenue R_B: {R_B_interval} USD")
    print(f"Driver's decision: {decision}\n")

    # 存储结果
    lambda_values.append(lambda_rate_interval)
    mu_values.append(mu_rate_interval)
    time_intervals.append(f"{current_time.time()} - {next_time.time()}")

    # 存储到Excel的数据
    results.append({
        "Time Interval": f"{current_time.time()} - {next_time.time()}",
        "Arrival Rate λ": lambda_rate_interval,
        "Service Rate μ": mu_rate_interval,
        "Demand D": D_interval,
        "Average Waiting Time T_w": T_w_interval,
        "Time Cost C_w": C_w_interval,
        "Expected Revenue R_A": R_A_interval,
        "Expected Revenue R_B": R_B_interval,
        "Driver's Decision": decision
    })

    # 更新当前时间
    current_time = next_time

# 保存结果到Excel
results_df = pd.DataFrame(results)
output_file_path = '~/Downloads/Decision_Model_Results.xlsx'
results_df.to_excel(output_file_path, index=False)
print(f"Results saved to {output_file_path}")

# 可视化 μ - λ 的分析结果
plt.figure(figsize=(14, 8))

# 绘制 μ - λ 的差值
mu_lambda_diff = [mu - lam for mu, lam in zip(mu_values, lambda_values)]

plt.plot(time_intervals, mu_lambda_diff, marker='o', label='μ - λ (Oct 22, 2013)', color='blue')

# 添加图例和标签
plt.xlabel('Time Interval')
plt.ylabel('μ - λ')
plt.title('Difference Between Service Rate and Arrival Rate on October 22, 2013')
plt.xticks(rotation=45)
plt.legend()

# 显示图表
plt.tight_layout()
plt.show()
