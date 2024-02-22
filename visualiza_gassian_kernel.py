from mayavi import mlab

import numpy as np

# 加载.npz文件
data = np.load(r'E:\GuidedResearchProject\nnUNet\nnUNet_preprocessed\Dataset027_Aorta\nnUNetPlans_3d_fullres\arota_024_key.npz')

# .npz文件中可能包含多个数组，使用关键字来访问特定的数组
# 假设我们想要加载名为'gaussian_kernel'的数组

non_zero_indices = np.where(data['key'] > 0)
gaussian_kernel = data['key'][0][2:]
print(gaussian_kernel.shape)



# mask = (gaussian_kernel == 0)
# gaussian_kernel[mask] = 0.2
#
# # 创建一个新的场景
# fig = mlab.figure(bgcolor=(1, 1, 1))
#
# # 为数据创建一个体积渲染对象
# src = mlab.pipeline.scalar_field(gaussian_kernel)
# vol = mlab.pipeline.volume(src, vmin=gaussian_kernel.min()-0.2, vmax=gaussian_kernel.max())
#
# # 定义颜色映射
# new_ctf = np.zeros((256, 4), dtype=np.uint8)
# new_ctf[0, :] = [255, 255, 100, 255]  # 浅黄色，对应数据值0
# new_ctf[1:64, :] = [255, 255, 100, 255]   # 浅黄色，对应数据值接近0
# new_ctf[64:128, :] = [0, 0, 255, 255]    # 蓝色，对应数据值1到2
# new_ctf[128:192, :] = [255, 0, 0, 255]   # 红色，对应数据值2到3
# new_ctf[192:256, :] = [0, 255, 0, 255]   # 绿色，对应数据值3到4
#
# # 应用新的颜色映射表
# vol.module_manager.scalar_lut_manager.lut.table = new_ctf
#
# # 更新颜色传输函数和体积渲染
# vol.update_ctf = True
# vol._volume_property.modified()
# vol._ctf.modified()
# vol._otf.modified()
#
# # 显示图形
# mlab.show()
#
# # 显示图形
# mlab.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import os
# import PySide2
#
# # dirname = os.path.dirname(PySide2.__file__)
# # # plugin_path = os.path.join(dirname, 'plugins', 'platforms')
# # # os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
# # os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = 'C:/Users/38408/anaconda3/envs/gr/lib/site-packages/PySide2/plugins/platforms'
#
#
# # 创建一个示例三维数组
# data = np.load(r'E:\GuidedResearchProject\nnUNet\nnUNet_preprocessed\Dataset027_Aorta\nnUNetPlans_3d_fullres\arota_024_key.npz')['key'][1][2:]
#
#
# colors = np.empty(data.shape, dtype=object)  # 使用dtype=object以存储颜色字符串
# colors[(1 < data) & (data <= 2)] = 'blue'
# colors[(2 < data) & (data <= 3)] = 'red'
# colors[(3 < data) & (data <= 4)] = 'green'
#
# # 创建一个图形和一个三维子图
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 绘制每个点
# for i in range(data.shape[0]):
#     for j in range(data.shape[1]):
#         for k in range(data.shape[2]):
#             if data[i, j, k] > 1:  # 只绘制大于1的值
#                 ax.scatter(i, j, k, color=colors[i, j, k])
#
# # 设置图形的标签和标题
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Z Axis')
# ax.set_title('3D Array Visualization')
#
# # 显示图形
# plt.show()