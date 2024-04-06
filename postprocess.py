from skimage import morphology
from skimage.morphology import ball
import numpy as np
import SimpleITK as sitk
import os
# from mayavi import mlab
from matplotlib import pyplot as plt
# from vmtk import vmtkscripts
# import vtk
# from vtk.util import numpy_support
# from vtk.util.numpy_support import numpy_to_vtk
# import numpy as np
# import vmtk.vmtkcenterlines as centerlines
# import vmtk.vmtksurfaceviewer as surfaceviewer
# import itk
def compute(infile, outfile):
    """
    Calls VMTK routine for centerline extraction.

    Parameters
    ----------
    infile : string
        Path to input mesh with open inlet/outlets (.stl format).
    outfile : string
        Path to output centerline (.vtp format).

    Returns
    -------
    None

    """

    # # read surface
    # centerlineReader = vmtkscripts.vmtkSurfaceReader()
    # centerlineReader.InputFileName = infile
    # centerlineReader.Execute()

    # centerline
    centerline = vmtkscripts.vmtkCenterlines()
    centerline.Surface = infile
    centerline.SeedSelectorName = ''
    centerline.AppendEndPoints = 1
    centerline.Execute()

    print('1111')

    # # extract branches
    # branchExtractor = vmtkscripts.vmtkBranchExtractor()
    # branchExtractor.Centerlines = surface
    # branchExtractor.Execute()
    #
    # # merge centerlines
    # centerlineMerge = vmtkscripts.vmtkCenterlineMerge()
    # centerlineMerge.Centerlines = branchExtractor.Centerlines
    # centerlineMerge.Execute()
    #
    # print('222')
    #
    # # write surface
    # centerlineWriter = vmtkscripts.vmtkSurfaceWriter()
    # centerlineWriter.OutputFileName = outfile
    # centerlineWriter.Surface = centerlineMerge.Centerlines
    # centerlineWriter.Execute()

    return centerline.Centerlines

def transfer_vtk(np_array,sitk_image):
    # 将 NumPy 数组转换为 VTK 图像
    vtk_image = vtk.vtkImageData()
    depth, height, width = np_array.shape
    vtk_image.SetDimensions(width, height, depth)
    vtk_image.SetSpacing(sitk_image.GetSpacing()[::-1])
    vtk_image.SetOrigin(sitk_image.GetOrigin()[::-1])

    # 转换 NumPy 数组的数据类型为 VTK 可用的类型
    vtk_array = numpy_to_vtk(num_array=np_array.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    vtk_array.SetNumberOfComponents(1)
    # 将转换后的数据设置为 VTK 图像的像素数据
    vtk_image.GetPointData().SetScalars(vtk_array)
    return vtk_image

def morphological_closing(image, radius):
    """ 执行形态学闭运算 """
    selem = ball(radius)
    return morphology.closing(image, selem)

def skeletonize_3d(image_array):
    """ 使用 SimpleITK 进行 3D 骨架化 """
    print('get image!')
    skeleton = sitk.BinaryThinning(sitk.GetImageFromArray(image_array.astype(int)))
    # filter = sitk.BinaryThinningImageFilter()
    # skeleton = filter.Execute(image_array)
    print('finished')
    return skeleton

def find_skeleton_points(skeleton):
    """ 查找骨架点的坐标 """
    return np.argwhere(skeleton==1)

def find_branch_points(skeleton):
    """ 寻找分支点 """
    branch_points = []
    start_ps = []
    points = find_skeleton_points(skeleton)
    for point in points:
        x, y, z = point
        min_p = np.maximum([x-1, y-1, z-1], 0)
        max_p = np.minimum([x+1, y+1, z+1], np.array(skeleton.shape) - 1)
        neighbourhood = skeleton[min_p[0]:max_p[0]+1, min_p[1]:max_p[1]+1, min_p[2]:max_p[2]+1]
        degree = np.sum(neighbourhood) - 1 # 减去中心点本身
        if degree == 1:
            start_ps.append((x, y, z))
        if degree >= 3: # 分支点至少有3个连接点
            branch_points.append((x, y, z))
    return np.array(branch_points)


DEBUG = True
CACHE = True

# 读取血管图像
if DEBUG:
    file_path = '/home/yiwen/guidedresearch/nnUNet/nnUNet_results/Dataset027_Aorta/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/validation/arota_003.nii.gz'
    cache_dir = '/home/yiwen/guidedresearch/nnUNet/nnUNet_results/Dataset027_Aorta/cache'

else:
    file_path = r'E:\GuidedResearchProject\nnUNet\nnUNet_results\Dataset027_Aorta\nnUNetTrainer__nnUNetPlans__3dfull\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_4\validation\arota_003.nii.gz'
    cache_dir = r'E:\GuidedResearchProject\nnUNet\nnUNet_results\Dataset027_Aorta\cache'

original_image = sitk.ReadImage(file_path)

image_array = sitk.GetArrayFromImage(original_image)

#检查输出图像是否是二值化的，检查完是二值化的
# unique_values = np.unique(image_array)
# if len(unique_values) == 2:
#     print("图像是二值化的")
# else:
#     print("图像不是二值化的")

# 获取图像的体素尺寸，以毫米为单位
spacing = original_image.GetSpacing()  # (x_spacing, y_spacing, z_spacing)

# # 如果图像不是二值化的，需要先进行二值化处理
# # 可以使用 SimpleITK 的 Otsu 阈值来进行二值化
# otsu_filter = sitk.OtsuThresholdImageFilter()
# otsu_filter.SetInsideValue(0)
# otsu_filter.SetOutsideValue(1)
# binary_image_sitk = otsu_filter.Execute(original_image)
# binary_image = sitk.GetArrayFromImage(binary_image_sitk)

dilate_filter = sitk.BinaryDilateImageFilter()
dilate_filter.SetKernelType(sitk.sitkBall)
dilate_filter.SetKernelRadius(2)  # 设置膨胀核半径

# 应用膨胀
dilated_image = dilate_filter.Execute(original_image)

# # 调整 disk 的半径以适应你的图像
erode_filter = sitk.BinaryErodeImageFilter()
erode_filter.SetKernelType(sitk.sitkBall)
erode_filter.SetKernelRadius(2)  # 设置腐蚀核半径

# 应用腐蚀
eroded_image = erode_filter.Execute(dilated_image)

# 执行连通区域分析，即为将整个分割分成多个块
connected_component_image = sitk.ConnectedComponent(eroded_image)
# 使用LabelShapeStatisticsImageFilter计算形态学特征，Execute 更新形态学特征计算，对每个块进行形态学（大小，label）的分析
original_shape_analysis = sitk.LabelShapeStatisticsImageFilter()
original_shape_analysis.Execute(connected_component_image)

# 新建的图像
# 对图片属性层级的操作：
# .GetSize()，.GetSpacing()，.GetDirection()，.GetOrigin()
# https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1Image.html
filtered_image = sitk.Image(connected_component_image.GetSize(), connected_component_image.GetPixelIDValue())
filtered_image.SetSpacing(connected_component_image.GetSpacing())
filtered_image.SetDirection(connected_component_image.GetDirection())
filtered_image.SetOrigin(connected_component_image.GetOrigin())

# 设置z坐标的阈值
z_threshold = 120  # 替换为适合您数据的值
labels = original_shape_analysis.GetLabels()
volumes = {label: original_shape_analysis.GetPhysicalSize(label) for label in labels}

# 获取每个连通组件的体积
# 形态学特征.GetPhysicalSize(), .GetLabels(), .GetCentroid()
# https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1LabelShapeStatisticsImageFilter.html#details

# 遍历每个连通区域
for label in original_shape_analysis.GetLabels():
    # 获取连通区域的中心点坐标
    centroid = original_shape_analysis.GetCentroid(label) #是RAS还是IJK？
    print(centroid)
    # 判断中心点的z坐标是否低于阈值
    if centroid[2] < z_threshold:
        # 如果低于阈值，跳过当前连通区域
        continue
    # 生成当前连通区域的二值图像
    binary_image = connected_component_image == label
    binary_image_cast = sitk.Cast(binary_image, connected_component_image.GetPixelIDValue())

    # 设置重采样器
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(connected_component_image.GetSize())
    resampler.SetOutputSpacing(connected_component_image.GetSpacing())
    resampler.SetOutputOrigin(connected_component_image.GetOrigin())
    resampler.SetOutputDirection(connected_component_image.GetDirection())

    # 设置插值方法
    # 对于二值图像，通常使用最近邻插值
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    # 重采样 binary_image_cast
    resampled_binary_image = resampler.Execute(binary_image_cast)
    # 将当前连通区域添加到过滤后的图像中
    filtered_image = sitk.Add(filtered_image, resampled_binary_image)


spacing = filtered_image.GetSpacing()
# 执行连通区域分析
filer_component_image = sitk.ConnectedComponent(filtered_image)
filter_shape_analysis = sitk.LabelShapeStatisticsImageFilter()
filter_shape_analysis.Execute(filer_component_image)
# 获取每个连通组件的标签
labels = filter_shape_analysis.GetLabels()
# 计算每个连通组件的实际体积
volumes = {label: filter_shape_analysis.GetPhysicalSize(label) for label in labels}

# 对连通组件按体积排序并选择第二大的
sorted_labels = sorted(volumes, key=volumes.get, reverse=True)
second_largest_label = sorted_labels[1] if len(sorted_labels) > 1 else None


# 如果你只想保留最大的连通组件（假设它是主血管）
largest_label = max(volumes, key=volumes.get)
largest_component_image = sitk.BinaryThreshold(filer_component_image,
                                               lowerThreshold=largest_label,
                                               upperThreshold=largest_label,
                                               insideValue=1, outsideValue=0)

largest_component_array = sitk.GetArrayFromImage(largest_component_image)
largest_component_array = largest_component_array.astype(bool)  # 确保 mask 是布尔型

# 创建膨胀过滤器
dilate_filter = sitk.BinaryDilateImageFilter()
dilate_filter.SetKernelType(sitk.sitkBall)
dilate_filter.SetKernelRadius(5)  # 设置膨胀核半径

# 应用膨胀
dilated_image = dilate_filter.Execute(largest_component_image)

# # 调整 disk 的半径以适应你的图像
erode_filter = sitk.BinaryErodeImageFilter()
erode_filter.SetKernelType(sitk.sitkBall)
erode_filter.SetKernelRadius(5)  # 设置腐蚀核半径

# 应用腐蚀
eroded_image = erode_filter.Execute(dilated_image)  # 假设阈值为200进行二值化
eroded_image_array = sitk.GetArrayFromImage(eroded_image)
eroded_image_array = eroded_image_array.astype(np.uint8)  # 确保 mask 是布尔型

# itk_image_type = itk.Image[itk.F, eroded_image.GetDimension()]
# itk_image = itk.GetImageFromArray(eroded_image_array)
#
# print('Image type: ', itk_image)
#
# skeleton = itk.BinaryThinningImageFilter3D.New(itk_image)
#
# skeleton_np = itk.GetArrayViewFromImage(itk_image)
#
# # 获取图像的尺寸
# z_dim, y_dim, x_dim = skeleton_np.shape
#
# # 创建一个空的数组来存储骨架点的坐标
# x, y, z = [], [], []
#
# # 遍历图像，找到骨架点
# for i in range(x_dim):
#     for j in range(y_dim):
#         for k in range(z_dim):
#             if skeleton_np[k, j, i] > 0:
#                 x.append(i)
#                 y.append(j)
#                 z.append(k)
#
# # 使用 Matplotlib 进行可视化
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, c='blue', marker='o', alpha=0.2, s=1)
#
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Z Axis')
#
# plt.title('3D Skeleton Visualization')
# plt.show()


vtk_image = transfer_vtk(eroded_image_array, eroded_image)

# 创建等值面提取器
contour_filter = vtk.vtkContourFilter()
contour_filter.SetInputData(vtk_image)
contour_filter.SetValue(0, 1)  # 设置合适的阈值
contour_filter.Update()

# 获取提取的表面
surface = contour_filter.GetOutput()

# 获取表面的点坐标
points = surface.GetPoints()

# 创建一个多边形网格
polydata = vtk.vtkPolyData()
polydata.SetPoints(points)

# 使用 VTK 保存为 STL
stlWriter = vtk.vtkSTLWriter()
stlWriter.SetFileName("output.stl")
stlWriter.SetInputData(polydata)
stlWriter.Write()

centerline = compute(surface, 'final.vtp')

viewer = surfaceviewer.vmtkSurfaceViewer()
viewer.Surface = centerline
viewer.Execute()

#
# # 执行骨架化
# if CACHE:
#     if not os.path.exists(cache_dir):
#         print('calculate skel')
#         os.makedirs(cache_dir, exist_ok=True)
#         skel = skeletonize_3d(eroded_image_array)
#         skel_np = sitk.GetArrayFromImage(skel)
#         np.save(os.path.join(cache_dir, 'skel_np.npy'), skel_np)
#     else:
#         print('load_cached skel_np.npy')
#         skel_np = np.load(os.path.join(cache_dir, 'skel_np.npy'))
#         skel = sitk.GetImageFromArray(skel_np)
#
# #
# # 3D 可视化
# # mlab.contour3d(skel, contours=[0.5])
# # mlab.show()
# # sitk.Show(skel)
# # x, y, z = np.where(skel_np==1)
#
# z_dim, y_dim, x_dim = skel_np.shape
#
# # 创建一个空的数组来存储骨架点的坐标
# x, y, z = [], [], []
#
# # 遍历图像，找到骨架点
# for i in range(x_dim):
#     for j in range(y_dim):
#         for k in range(z_dim):
#             if skel_np[k, j, i] > 0:
#                 x.append(i)
#                 y.append(j)
#                 z.append(k)
#
# # # 创建3D散点图
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, c='blue', marker='o', alpha=0.2, s=1)
# #
# # #
# # # # 寻找分支点
# branches = find_branch_points(skel_np)
# print('branches',branches)
# for ras in branches:
#     z, y, x = np.array(ras)
#     # ax.scatter(x, y, z, c='red', marker='x', alpha=0.2, s=1)
# plt.show()
#
# # radius = 3
# # for ras in branches:
# #     for dx in range(-radius, radius + 1):
# #         for dy in range(-radius, radius + 1):
# #             for dz in range(-radius, radius + 1):
# #                 x, y, z = np.array(ras) + np.array([dx, dy, dz])
# #                 if 0 <= x < largest_component_array.shape[0] and 0 <= y < largest_component_array.shape[1] and 0 <= z < largest_component_array.shape[2]:
# #                     largest_component_array[int(x), int(y), int(z)] = -1023  # 将像素值设置为黑色
# #
# final_save = sitk.GetImageFromArray(largest_component_array.astype(np.uint8) * 255)
#
# # 如果需要，您还可以设置原点、间距和方向
# final_save.SetOrigin(connected_component_image.GetOrigin()) # 这里的值取决于您的具体情况
# final_save.SetSpacing(connected_component_image.GetSpacing()) # 同上
# final_save.SetDirection(connected_component_image.GetDirection())
#
#
# # 将结果保存回文件
# if not DEBUG:
#     sitk.WriteImage(eroded_image, r'E:\GuidedResearchProject\nnUNet\nnUNet_results\Dataset027_Aorta\nnUNetTrainer__nnUNetPlans__3dfull\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_4\validation\arota_00311_new.nii.gz')
#
#
#



# b, c, h, w = key.shape
#             center = np.zeros((b, c, len(self.plans_manager.plans['biases']), 3), dtype=np.float32)
#             center_predict = np.zeros((b, c, len(self.plans_manager.plans['biases']), 3), dtype=np.float32)
#
#             total_distance = 0
#
#             for batch_idx in range(b):  # 使用 range(b)
#                 for channel_idx in range(c):  # 使用 range(c)
#                     kernel = key[batch_idx][channel_idx]
#                     output = key_out[batch_idx][channel_idx]
#                     non_zero = np.where(kernel > 0)
#
#                     if non_zero[0].size == 0:  # 如果没有非零元素，跳过当前kernel
#                         continue
#
#                     for i, bias in enumerate(self.plans_manager.plans['biases']):
#                         adjusted_kernel = kernel - bias - 0.7  # hard-coded
#                         adjusted_output = output - bias
#
#                         mask = (adjusted_kernel > 0) & (adjusted_kernel < 1)  # 不再使用non_zero
#                         if not np.any(mask):  # 如果mask全为False，则跳过
#                             continue
#                         # 获取满足条件的元素的索引
#                         gt_indices = np.array(np.nonzero(mask))
#
#                         max_index = np.argmax(adjusted_kernel[mask])
#                         peak_coords_abs = gt_indices[:, max_index]
#                         center[batch_idx, channel_idx, i, :] = peak_coords_abs
#
#                         mask_output = (adjusted_output > 0) & (adjusted_output < 1)  # 不再使用non_zero
#                         if not np.any(mask_output):  # 如果mask全为False，则跳过
#                             continue
#                         # 获取满足条件的元素的索引
#                         gt_indices = np.array(np.nonzero(mask_output))
#
#                         max_index_predict = np.argmax(adjusted_output[mask_output])
#                         peak_coords_abs_predict = gt_indices[:, max_index_predict]
#                         center_predict[batch_idx, channel_idx, i, :] = peak_coords_abs_predict
#             total_distance = np.mean(np.linalg.norm(center - center_predict, axis=-1))