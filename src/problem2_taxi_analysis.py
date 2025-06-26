import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置图形风格
sns.set(style="whitegrid")

# 读取TXT文件，无标题行，假设文件使用逗号（,）分隔
file_path = '~/Downloads/1.TaxiData.txt'  # 请替换为你的文件路径
df = pd.read_csv(file_path, sep=',', header=None)

# 将第二列（时间列）转换为datetime对象
df[1] = pd.to_datetime(df[1], format='%H:%M:%S', errors='coerce')

# 提取小时信息，并过滤“是否空载值为0”的数据
df['小时'] = df[1].dt.hour
empty_taxis_per_hour = df[df[4] == 1].groupby('小时').size()

# 打印每小时空载值为0的出租车数量
print(empty_taxis_per_hour)

# 计算每小时的权重（每小时数量占全天总数的比例）
total_empty_taxis = empty_taxis_per_hour.sum()
weights = empty_taxis_per_hour / total_empty_taxis


# 打印每小时的空载值为0的数量和对应的权重
for hour, count in empty_taxis_per_hour.items():
    print(f"Hour: {hour}, Count: {count}, Weight: {weights[hour]:.4f}")

# 可视化“是否空载值为0”的出租车数量
plt.figure(figsize=(12, 6))
sns.barplot(x=empty_taxis_per_hour.index, y=empty_taxis_per_hour.values, palette='viridis')
plt.title('Number of Taxis with passengers per Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Taxis')
plt.xticks(range(0, 24))
plt.grid(True)

# 显示图表
plt.tight_layout()
plt.show()
