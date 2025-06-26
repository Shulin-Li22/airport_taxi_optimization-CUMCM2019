import pandas as pd

# 读取TXT文件，无标题行
file_path = '~/Downloads/1.TaxiData.txt'  # 请替换为你的文件路径
df = pd.read_csv(file_path, sep=',', header=None)  # 如果分隔符是逗号或空格，可以改为 ',' 或 ' '

# 定义要添加的日期部分
date_str = '2013-10-22'

# 将第二列（时间数据）转换为 datetime 对象，并添加日期
df[1] = pd.to_datetime(date_str + ' ' + df[1], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# 转换经纬度列的数据类型为浮点数，并处理无效值
df[2] = pd.to_numeric(df[2], errors='coerce')  # 经度列
df[3] = pd.to_numeric(df[3], errors='coerce')  # 纬度列

# 删除经纬度或时间中包含无效值的行
df = df.dropna(subset=[1, 2, 3])

# 设定经纬度范围
lon_min, lon_max = 113.77003, 113.83039
lat_min, lat_max = 22.61773, 22.66807

# 筛选出经纬度在指定范围内的样本
df_filtered = df[(df[2] >= lon_min) & (df[2] <= lon_max) & (df[3] >= lat_min) & (df[3] <= lat_max)]

# 筛选出第五列为0的样本（即载人状态）
df_filtered = df_filtered[df_filtered[4] == 1]

# 按车牌号分组并按时间排序
df_filtered = df_filtered.sort_values(by=[0, 1])


# 定义一个函数来检测车辆的进入和离开时间
def detect_arrival_departure(group):
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


# 应用函数到每个分组
df_results = df_filtered.groupby(0).apply(detect_arrival_departure).reset_index(drop=True)

# 保存筛选后的数据到新的Excel文件
output_file_path = '~/Downloads/出租车到离时间结果.xlsx'  # 请替换为你想要保存的文件路径
df_results.to_excel(output_file_path, index=False)

print(f"筛选后的数据已保存到 {output_file_path}")
