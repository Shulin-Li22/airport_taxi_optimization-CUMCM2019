import pandas as pd
from gmplot import gmplot

# 读取数据
trip_data_path = '~/Downloads/Taxi_Trips999.xlsx'
trip_df = pd.read_excel(trip_data_path)

# 提取经纬度数据
latitudes = trip_df['起始纬度'].tolist() + trip_df['终止纬度'].tolist()
longitudes = trip_df['起始经度'].tolist() + trip_df['终止经度'].tolist()

# 创建 gmplot 对象
gmap = gmplot.GoogleMapPlotter(22.6379, 113.8005, 13)  # 设置中心点为深圳机场附近的经纬度

# 绘制热力图
gmap.heatmap(latitudes, longitudes, radius=20)

# 保存生成的热力图
gmap.draw('/Users/Henry/Downloads/taxi2_heatmap_gmplot.html')

print(f"热力图已保存")
