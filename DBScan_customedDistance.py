# DBscanCluster-testing
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from sklearn.cluster import DBSCAN
import time


# region -----------------------Step 1 画3D颜色图fucntion---------------------------------
def voxelFun(d1, colname, TTname):

    # 首先排除空值
    d1 = np.transpose(d1, (1, 2, 0))  # 把时间放到最后一维
    data = ~np.isnan(d1)

    # 归一化数据到0-1范围
    norm = mcolors.Normalize(vmin=np.nanmin(d1), vmax=np.nanmax(d1))
    normalized_data = norm(d1)

    if (len(np.unique(d1[~np.isnan(d1)])) == 1):
        normalized_data = d1

    # 使用颜色映射将归一化的数据映射到红色到绿色的范围
    color_map = plt.get_cmap(colname)
    mapped_colors = color_map(normalized_data)  # 空值返回0,0,0,0
    # 根据归一化的值调整透明度
    alpha_values = 40 - normalized_data * 30  # 将归一化值从范围10-80映射到透明度
    # alpha_values = np.ones_like(normalized_data) * 80
    # 在alpha_values中将空值的格点透明度置为0
    zero_alpha_indices = np.where(mapped_colors[:, :, :, 3] == 0)
    alpha_values[zero_alpha_indices] = 0
    mapped_colors[..., 3] = alpha_values / 255.0  # 因为在颜色映射中，透明度应该是0-1范围的
    # 修改为色号

    def rgba_to_hex(rgba):
        if rgba[3] == 0:  # 检查rgba中a是否为0值
            return '#00000000'
        return '#' + ''.join(['{:02X}'.format(int(x * 255)) for x in rgba])
    hex_colors = np.vectorize(
        rgba_to_hex, signature='(n)->()')(mapped_colors)

    def rgba_to_hex(rgba):
        if rgba[3] == 0:  # 检查rgba中a是否为0值
            return '#00000000'
        return '#' + ''.join(['{:02X}'.format(int(x * 255)) for x in rgba])
    hex_colors = np.vectorize(
        rgba_to_hex, signature='(n)->()')(mapped_colors)

    def explode(data):
        shape_arr = np.array(data.shape)
        size = shape_arr[:3]*2 - 1
        exploded = np.zeros(np.concatenate(
            [size, shape_arr[3:]]), dtype=data.dtype)
        exploded[::2, ::2, ::2] = data
        return exploded

    def expand_coordinates(indices):
        x, y, z = indices
        x[1::2, :, :] += 1
        y[:, 1::2, :] += 1
        z[:, :, 1::2] += 1
        return x, y, z

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([data.shape[0], data.shape[1], data.shape[2]])
    colors = hex_colors
    colors = explode(colors)
    filled = explode(np.ones((d1.shape[0], d1.shape[1], d1.shape[2])))
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))
    ax.voxels(x/2, y/2, z/2, filled,
              facecolors=colors)

    # 设置轴标签
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Time')
    ax.set_title(TTname)

    # 添加颜色轴
    sm = plt.cm.ScalarMappable(cmap=color_map,
                               norm=plt.Normalize(vmin=np.nanmin(d1),
                                                  vmax=np.nanmax(d1)))
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical',
                        fraction=0.03, pad=0.05)
    cbar.set_label('SPI Value')

    plt.show()

    # from plotly.offline import plot_mpl # 使用plotly库将Matplotlib图转换为HTML文件
    # plot_mpl(fig, filename="test.html")
    # mpld3.save_html(fig, "test.html")
# endregion END

# region -----------------------Step 2 读npz文件---------------------------------
# file_name = 'H:/ERA5Land_SPI_China_19691231_20221231.npz'
# data = np.load(file_name)
# a = data['arr1']
# spi_array_80 = a[14450:14500,:,:]
# del data, a
# # # 保存数组a为npz文件到当前目录
# np.savez('SPI2DBscantest.npz', spi_array_80=spi_array_80)

# 加载npz文件中的数组
loaded_data = np.load("SPI2DBscantest.npz")
# 从loaded_data中获取数组a
spi_array_80 = loaded_data['spi_array_80']
spi_array_80 = spi_array_80[0:40, 30:40, 40:50] #取小块测试
# # 只画出低于阈值（-1）的SPI值,测试看看
# drev = spi_array_80.copy()
# drev[drev > (-1)] = np.nan
# voxelFun(drev, "winter", "SPI<-1")  # 低于-1的

# endregion END


# region -----------------------Step 3 DBcluster---------------------------------
# 计算两点之间的距离-自定义
def custom_distance(point1, point2):
    px1, py1, pz1, pv1 = point1[0], point1[1], point1[2], point1[3]
    px2, py2, pz2, pv2 = point2[0], point2[1], point2[2], point2[3]
    pysdis = np.sqrt((px1-px2)**2+(py1-py2)**2)
    dpv = np.where((pv1 < (-1) and pv2 < (-1)), 0, pv1-pv2)  # 若值为正，返回0
    customDist = np.sqrt((pysdis)**2+(pz1-pz2)**2+(dpv)**2)
    return customDist

z, x, y = np.where(~np.isnan(spi_array_80))
data_points = np.column_stack([z, x, y, spi_array_80[z, x, y]])  # 获取数据点

start_time = time.time()
db = DBSCAN(eps=np.sqrt(2), min_samples=10, metric=custom_distance)
db.fit(data_points)
end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)
# endregion END

#region -----------------------Step 4 统计结果并画图---------------------------------
# 创建一个与 array_3d 具有相同形状的空数组，值全为0
labels_array = np.empty_like(spi_array_80)
core_labels_array = np.empty_like(spi_array_80)
# 把核心点都置入标签
dbcorelabels = np.full(data_points.shape[0], -1)
dbcorelabels[db.core_sample_indices_] = db.labels_[db.core_sample_indices_]
# 使用 z, x, y 数组来索引新数组，并将相应的值设置为 data_points 的第四列的值
labels_array[z, x, y] = db.labels_
core_labels_array[z, x, y] = dbcorelabels
labels_array[labels_array == -1] = np.nan
core_labels_array[core_labels_array == -1] = np.nan

np.savez('SPI2DBscantest_labels_array.npz', labels_array=labels_array) #所有分类
np.savez('SPI2DBscantest_core_labels_array.npz', core_labels_array=core_labels_array) #核心

labelwithV1 = spi_array_80.copy()
labelwithV2 = spi_array_80.copy()
labelcorewithV1 = spi_array_80.copy()
labelcorewithV2 = spi_array_80.copy()

labelwithV1[labels_array != 0.0] = np.nan #第1类 即 第一场干旱
labelcorewithV1[core_labels_array != 0.0] = np.nan
labelwithV2[labels_array != 1.0] = np.nan #第2类 即 第二场干旱
labelcorewithV2[core_labels_array != 1.0] = np.nan

#画图
voxelFun(labelwithV1, "winter", "ALL Clusters")
voxelFun(labelcorewithV1, "winter", "Cluster cores")

# voxelFun(labelwithV2, "winter", "ALL Clusters")
# voxelFun(labelcorewithV2, "winter", "Cluster cores")

print("end")

#endregion END


# continue：源代码中设定v>-1不能做核心点
