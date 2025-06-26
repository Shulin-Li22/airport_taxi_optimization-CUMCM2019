import pandas as pd
from gmplot import gmplot

# 读取筛选后的数据
df_filtered = pd.read_excel('~/Downloads/出租车载人筛选结果.xlsx', header=None)
df_filtered.columns = ['License Plate', 'Timestamp', 'Longitude', 'Latitude', 'Empty or not', 'Speed']

# 设定地图中心位置
gmap = gmplot.GoogleMapPlotter(22.6434, 113.8, 13)

# 绘制热力图
gmap.heatmap(df_filtered['Latitude'], df_filtered['Longitude'])

# 保存到HTML文件
gmap.draw('/Users/Henry/Downloads/taxi_heatmap_gmplot.html')
