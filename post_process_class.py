import numpy as np
import SimpleITK as sitk
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from skspatial.objects import Plane, Points, Line, Vector
from skspatial.plotting import plot_3d
from scipy.io import loadmat
# import vmtk.vmtkcenterlines as centerlines
import json
from itertools import combinations
import random
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import pickle

import h5py as h5
import numpy as np
from pathlib import Path
from Geometry_Proj_Matrix import get_index_from_world
from Geometry_Proj_Matrix import get_camera_intrinsics, rotation_angles
from PIL import Image
import os
import re
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

random.seed(2024)


class Postprocess1:
    def __init__(self, file_path, save_path, z_threshold=120):
        self.file_base_path = file_path
        self.save_base_path = save_path
        self.z_threshold = z_threshold

        self.original_img = None
        self.original_img_processed = None
        self.final_img = None
        self.final_img_processed = None
        self.original_properties = None

    def run(self, file):
        self.file_path = os.path.join(self.file_base_path, file)
        self.save_path = os.path.join(self.save_base_path, file)

        self.original_img = self.read_image()
        self.original_img_processed = self.morphological_process(1, self.original_img)
        self.final_img = self.connected_component_analysis(self.original_img_processed, self.z_threshold)
        self.final_img_processed = self.morphological_process(1, self.final_img)
        return self.final_img_processed

    def save(self):
        self.original_properties = self.get_img_property(self.original_img)
        self.save_img(self.final_img_processed, self.original_properties)

    def get_center(self):
        np_image = sitk.GetArrayFromImage(self.final_img_processed)
        # 找到值为1的所有点的坐标
        indices = np.argwhere(np_image == 1)
        # 计算中心坐标
        centroid = np.mean(indices, axis=0)
        return centroid #ijk np style

    def read_image(self):
        '''
        使用simpleitk读取nii文件
        '''
        original_image = sitk.ReadImage(self.file_path)
        return original_image

    def save_img(self, img, properties):
        largest_component_array = sitk.GetArrayFromImage(img)
        largest_component_array = largest_component_array.astype(bool)  # 确保 mask 是布尔型
        final_save = sitk.GetImageFromArray(largest_component_array.astype(np.uint8))
        final_save.SetSpacing(properties[2]) # 同上
        final_save.SetDirection(properties[3])
        final_save.SetOrigin(properties[4])
        # 将结果保存回文件
        sitk.WriteImage(final_save, self.save_path)

    def get_img_property(self, img):
        Size = img.GetSize()
        PixelIDValue = img.GetPixelIDValue()
        Spacing = img.GetSpacing()
        Direction = img.GetDirection()
        Origin = img.GetOrigin()
        properties = [Size, PixelIDValue, Spacing, Direction, Origin]
        return properties


    def morphological_process(self, raduis, processed_img):
        dilate_filter = sitk.BinaryDilateImageFilter()
        dilate_filter.SetKernelType(sitk.sitkBall)
        dilate_filter.SetKernelRadius(raduis)  # 设置膨胀核半径

        # 应用膨胀
        dilated_image = dilate_filter.Execute(processed_img)

        # 调整 disk 的半径以适应你的图像
        erode_filter = sitk.BinaryErodeImageFilter()
        erode_filter.SetKernelType(sitk.sitkBall)
        erode_filter.SetKernelRadius(raduis)  # 设置腐蚀核半径

        # 应用腐蚀
        eroded_image = erode_filter.Execute(dilated_image)
        return eroded_image

    def connected_component_analysis(self, processed_img, z_threshold):
        connected_component_image = sitk.ConnectedComponent(processed_img)
        # 使用LabelShapeStatisticsImageFilter计算形态学特征，Execute 更新形态学特征计算，对每个块进行形态学（大小，label）的分析
        original_shape_analysis = sitk.LabelShapeStatisticsImageFilter()
        original_shape_analysis.Execute(connected_component_image)

        # 新建的图像
        # 对图片属性层级的操作：
        # .GetSize()，.GetSpacing()，.GetDirection()，.GetOrigin()
        # https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1Image.html
        original_properties = self.get_img_property(connected_component_image)
        filtered_image = sitk.Image(original_properties[0], original_properties[1])
        filtered_image.SetSpacing(original_properties[2])
        filtered_image.SetDirection(original_properties[3])
        filtered_image.SetOrigin(original_properties[4])

        # 获取每个连通组件的体积
        # 形态学特征.GetPhysicalSize(), .GetLabels(), .GetCentroid()
        # https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1LabelShapeStatisticsImageFilter.html#details

        # 遍历每个连通区域
        for label in original_shape_analysis.GetLabels():
            # 获取连通区域的中心点坐标
            centroid = original_shape_analysis.GetCentroid(label)  # 是RAS还是IJK？
            print(centroid)
            # 判断中心点的z坐标是否低于阈值
            if centroid[2] < z_threshold:
                # 如果低于阈值，跳过当前连通区域
                continue
            # 生成当前连通区域的二值图像
            binary_image = connected_component_image == label
            binary_image_cast = sitk.Cast(binary_image, original_properties[1])

            # 设置重采样器
            resampler = sitk.ResampleImageFilter()
            resampler.SetSize(original_properties[0])
            resampler.SetOutputSpacing(original_properties[2])
            resampler.SetOutputDirection(original_properties[3])
            resampler.SetOutputOrigin(original_properties[4])

            # 设置插值方法
            # 对于二值图像，通常使用最近邻插值
            # TODO: 如果对于0/1/2，这里需要修改为三次插值, 同时改为做两次形态学分析
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            # 重采样 binary_image_cast
            resampled_binary_image = resampler.Execute(binary_image_cast)
            # 将当前连通区域添加到过滤后的图像中
            filtered_image = sitk.Add(filtered_image, resampled_binary_image)

        # 执行连通区域分析
        filer_component_image = sitk.ConnectedComponent(filtered_image)
        filter_shape_analysis = sitk.LabelShapeStatisticsImageFilter()
        filter_shape_analysis.Execute(filer_component_image)
        # 获取每个连通组件的标签
        labels = filter_shape_analysis.GetLabels()
        # 计算每个连通组件的实际体积
        volumes = {label: filter_shape_analysis.GetPhysicalSize(label) for label in labels}

        # 对连通组件按体积排序并选择第二大的
        # sorted_labels = sorted(volumes, key=volumes.get, reverse=True)
        # second_largest_label = sorted_labels[1] if len(sorted_labels) > 1 else None

        # 如果你只想保留最大的连通组件（假设它是主血管）
        largest_label = max(volumes, key=volumes.get)
        largest_component_image = sitk.BinaryThreshold(filer_component_image,
                                                       lowerThreshold=largest_label,
                                                       upperThreshold=largest_label,
                                                       insideValue=1, outsideValue=0)
        return largest_component_image

#TODO:中间的对模型输出的处理，可能也需要做连通性分析
class Postprocess2:
    def __init__(self, mat_path, save_path, predicted_key_points, AAA_point, original_spacing, original_origin):
        '''
        predicted_key_points, AAA_point 应该是和读入simpleitk变成np数组的顺序，z，y，x
        处理中会把centerline_graph变成一个dict

        '''
        self.mat_path = mat_path
        self.save_path = save_path
        self.pose_list = ['ra_1*ra_2*main_branch*lowbranch_left*lowbranch_right', 'main_branch*lowbranch_left*lowbranch_right', 'ra_1*ra_2', 'main_branch', 'lowbranch_left', 'lowbranch_right'] #
        self.dict_key_list = ['ra_1', 'ra_2', 'main_branch', 'lowbranch_left', 'lowbranch_right']

        self.mat = loadmat(self.mat_path)
        self.branches = np.transpose(np.array(self.mat['branches']).astype(np.float32), (2,1,0))
        self.start = np.transpose(np.array(self.mat['start_ps']).astype(np.float32), (2, 1, 0))
        self.centerline_graph_input = np.array(self.mat['Centerlines'][0])
        self.skel = np.transpose(np.array(self.mat['skel']), (2,1,0))
        self.skel_point = self.process_skel()
        self.start_point = self.process_start()
        self.centerline_graph = self.process_centerline_gragh() #dict
        self.key_points = self.process_branch()
        self.key_points_cluster = self.cluster_center_points(self.key_points)

        self.AAA_point = self.find_AAA(self.skel_point, AAA_point)

        self.final_key_points = self.match_points(self.key_points_cluster, predicted_key_points, self.AAA_point) #dict
        self.process_key_list()   #因为matlab导出的文件左右和我们现在投影的左右是反的，所以我们交换一下对应key的value
        self.related_vessel = self.find_key_point_related_centerline(self.final_key_points, self.centerline_graph)

        itk_img = sitk.ReadImage(r'E:\Newdataset\Patient_3\VesselMask_2_label.nii')
        direction = itk_img.GetDirection()

        self.original_volume = self.get_original_volume(self.skel, original_spacing, original_origin, direction)
        self.ijk_normal_trans, self.center, self.spacing = self.transformation_matrix(self.original_volume)

        # itk_img = sitk.ReadImage(r'E:\Newdataset\Patient_3\VesselMask_2_label.nii')
        # itk_img_array = sitk.GetArrayFromImage(itk_img).transpose(2,1,0)
        #
        # debug = self.centerline_graph[0]
        # debug1 = self.centerline_graph[0][0]
        #
        # reordered_indices = debug1[[2, 1, 0]]  # 根据新的顺序重排debug1中的元素
        # i, j, k = reordered_indices.astype(int)
        # check = itk_img_array[i,j,k]  # 使用重排后的索引来访问itk_img_array

        self.centerline_graph = self.transform_points_dict1(self.centerline_graph, self.ijk_normal_trans, self.center, self.spacing)
        self.final_key_points = self.transform_points_dict2(self.final_key_points, self.ijk_normal_trans, self.center, self.spacing)
        self.skel_point = self.transform_points_dict3(self.skel_point, self.ijk_normal_trans, self.center, self.spacing)
        self.start_point = self.transform_points_dict3(self.start_point, self.ijk_normal_trans, self.center, self.spacing)

        self.filter_key_points_vessel()

        # #确定使用的3D点（要确定view，目前按照全局view来选，考虑剩下的点选哪些）
        full_select_3d_point_dict = {}
        merged_plane = {}
        for i, item in enumerate(self.pose_list):
            # print('item',item)
            parts = item.split('*')
            all_ids = []
            key_points = []
            normals = []
            centers = []
            related_local_points = []
            for part in parts:
                all_ids.extend(self.related_vessel.get(part, []))
                key_points.extend(self.final_key_points[part])
                normals.extend(self.plane_dict[part]['normal'])
                centers.extend(self.plane_dict[part]['center'])
                related_local_points.extend(self.choosed_key_related_points[part])

            all_points = []
            end_start = []
            for vessel_id in all_ids:
                points = self.centerline_graph.get(vessel_id)
                if points is not None:
                    all_points.extend(points)
                    end_start.extend(points[-1])
            all_points_array = np.array(all_points).reshape(-1, 3)
            local_points_array = np.array(related_local_points).reshape(-1, 3)
            num_points_to_select = min(20, len(all_points_array))
            selected_indices = np.random.choice(len(all_points_array), size=num_points_to_select, replace=False)
            selected_points = all_points_array[selected_indices]  # 这里是选择一些点做PnP算法的3D点

            full_select_3d_point_dict[i] = np.concatenate(
                (selected_points, np.array(key_points).reshape(-1, 3), np.array(end_start).reshape(-1, 3)), axis=0)

            if len(local_points_array) < 90:
                size = len(local_points_array)
            else:
                size = 90
            # selected_indices_plot_plane_idx = np.random.choice(len(local_points_array), size=size, replace=False)
            # selected_indices_plot_plane = local_points_array[selected_indices_plot_plane_idx]  #这里是选择一些点找相关的3d平面
            _, new_normal, new_center = self.fit_plane(local_points_array, plot=True)
            new_normal = new_normal / np.linalg.norm(new_normal)
            merged_plane[i] = {'normal': new_normal, 'center': new_center}

            # if len(parts) == 1:
            #     normal = self.plane_dict[parts[0]]['normal']
            #     normal_normalized = normal / np.linalg.norm(normal)
            #     merged_plane[i] = {'normal': normal_normalized, 'center': self.plane_dict[parts[0]]['center']}
            # else:
            # # normals = np.array(normals).reshape(-1, 3)
            # # centers = np.array(centers).reshape(-1, 3)
            # # new_normal, new_center = self.merge_plane(normals, centers)
            #     selected_indices_plot_plane_idx = np.random.choice(len(all_points_array), size=90, replace=False)
            #     selected_indices_plot_plane = all_points_array[selected_indices_plot_plane_idx]  #这里是选择一些点找相关的3d平面
            #     _, new_normal, new_center = self.fit_plane(selected_indices_plot_plane, plot=True)
            #     new_normal = new_normal / np.linalg.norm(new_normal)
            #     merged_plane[i] = {'normal': new_normal, 'center': new_center}



        self.ensure_merged_plane_same_direction(merged_plane)

        PnPSolver = PnPPoseEstimition()
        for key in full_select_3d_point_dict.keys():
            self.landmark_3d[key] = self.get_3d_landmarks(full_select_3d_point_dict[key], self.ijk_normal_trans, self.center)
            self.landmark_2d[key] = self.get_2d_landmarks(full_select_3d_point_dict[key], merged_plane[key], key)
            pose, success, error = PnPSolver.PnP(self.landmark_3d[key], self.landmark_2d[key])
            plt.figure(figsize=(3, 3))

            plt.scatter(self.landmark_2d[key][:, 0], self.landmark_2d[key][:, 1], c='red', label='Mapped 2D Points')
            plt.xlim(0, 256)
            plt.ylim(0, 256)
            plt.gca().invert_yaxis()  # Y轴反向，以符合图像坐标系统
            plt.xlabel('Pixel X')
            plt.ylabel('Pixel Y')
            # plt.title('2D Points Mapped to Pixel Coordinates')
            # plt.legend()
            plt.grid(False)
            print(f'The {self.pose_list[key]} related pose is \n {pose}')

        # print('original', direction)
        # arr = sitk.GetArrayFromImage(itk_img).transpose(2,1,0)
        # arr -= 1024
        #
        # three_d_ijk = self.inv_get_3d_landmarks(self.landmark_3d[0], self.ijk_normal_trans, self.center)
        # radius = 10
        # for ijk in three_d_ijk:
        #     for dx in range(-radius, radius + 1):
        #         for dy in range(-radius, radius + 1):
        #             for dz in range(-radius, radius + 1):
        #                 # print('ijk_before', ijk)
        #                 x, y, z = np.array(ijk) + np.array([dz, dy, dx])
        #                 # print('ijk_end',[z,x,y])
        #                 if 0 <= x < arr.shape[0] and 0 <= y < arr.shape[1] and 0 <= z < \
        #                         arr.shape[2]:
        #                     arr[int(x), int(y), int(z)] = 300
        # # three_d_ras = self.trans_ras(three_d_ijk, np.array(self.original_volume.GetSpacing()), np.array(self.original_volume.GetDirection()).reshape(-1,3))
        # arr = arr.transpose(2,1,0)
        # print(np.where(arr == 300))

        # sitk_vol = sitk.GetImageFromArray(arr)
        # debug11 = itk_img.GetDirection()
        # print('debug11', debug11)

        # spacing = np.asarray(itk_img.GetSpacing())
        # sitk_vol.SetSpacing(spacing)
        # direction = np.asarray(itk_img.GetDirection())
        # print('original', direction)
        # sitk_vol.SetDirection(direction)
        # sitk.WriteImage(sitk_vol, './debug.nii')
        # sitk_vol = sitk.GetImageFromArray(arr)
        # debug11 = self.original_volume.GetDirection()
        # direction = np.asarray(self.original_volume.GetDirection())
        # spacing = np.asarray(self.original_volume.GetSpacing())
        # sitk_vol.SetSpacing(spacing)
        # sitk_vol.SetDirection(direction)
        # sitk.WriteImage(sitk_vol, './debug.nii')
        #
        # reloaded_img = sitk.ReadImage('./debug.nii')
        #
        # # 检查方向
        # reloaded_direction = reloaded_img.GetDirection()
        # print("原始方向:", direction)
        # print("重新加载后的方向:", reloaded_direction)
        #
        # # 检查方向矩阵是否一致
        # if np.allclose(direction, reloaded_direction):
        #     print("方向保持不变。")
        # else:
        #     print("方向发生了变化。")

    def process_key_list(self):
        left_values = self.final_key_points['lowbranch_left']
        right_values = self.final_key_points['lowbranch_right']

        # Swap the values in the dictionary
        self.final_key_points['lowbranch_left'] = right_values
        self.final_key_points['lowbranch_right'] = left_values

    def filter_key_points_vessel(self):
        self.plane_dict = {}
        self.landmark_3d = {}
        self.landmark_2d = {}
        self.choosed_key_related_points = {}

        for name, points in self.final_key_points.items():
            choosed_centerline_dict = {key: self.centerline_graph[key] for key in self.related_vessel[name]}
            if name == 'AAA':
                normal, center, points_list = self.construct_local_plane(points, choosed_centerline_dict, choosed_dist=50)
                self.plane_dict[name] = {'normal': normal, 'center': center}
                self.choosed_key_related_points[name] = points_list
            elif name == 'ra_1' or name == 'ra_2':
                final_related_vessel, removed_index = self.exclude_outliers(choosed_centerline_dict, name)
                if len(final_related_vessel.keys()) <= 2:
                    add_centerline_dict = {key: self.centerline_graph[key] for key in
                                           self.related_vessel['main_branch']}
                    final_related_vessel.update(add_centerline_dict)
                for index in removed_index:
                    if index in self.related_vessel[name]:
                        self.related_vessel[name].remove(index)

                normal, center, points_list = self.construct_local_plane(points, final_related_vessel)
                self.plane_dict[name] = {'normal': normal, 'center': center}
                self.choosed_key_related_points[name] = points_list

            elif len(choosed_centerline_dict.keys()) == 3:
                normal, center, points_list = self.construct_local_plane(points, choosed_centerline_dict)
                self.plane_dict[name] = {'normal': normal, 'center': center}
                self.choosed_key_related_points[name] = points_list

            else:
                final_related_vessel, removed_index = self.exclude_outliers(choosed_centerline_dict, name)
                normal, center, points_list = self.construct_local_plane(points, final_related_vessel)
                for index in removed_index:
                    if index in self.related_vessel[name]:
                        self.related_vessel[name].remove(index)
                self.plane_dict[name] = {'normal': normal, 'center': center}
                self.choosed_key_related_points[name] = points_list

    def ensure_merged_plane_same_direction(self, plane):
        for i in range(1,len(plane.keys())):
            self.ensure_same_direction(plane[0]['normal'], plane[i]['normal'])

    def find_AAA(self, skel_point, aaa_point):
        distances = [np.linalg.norm(np.array(aaa_point) - np.array(ref_point)) for ref_point in
                     skel_point]
        # 找到最小距离的索引
        closest_index = np.argmin(distances)
        choosed_aaa_pos = skel_point[closest_index]

        return choosed_aaa_pos

    def trans_ras(self, points: np.ndarray, spacing: np.ndarray, direction_matrix: np.ndarray):
        origin = np.array([0,0,0])
        scale_matrix = np.diag(spacing)
        transform_matrix = np.dot(direction_matrix, scale_matrix)
        ijk_points = []
        for point in points:
            ijk_point = np.dot(transform_matrix,
                               np.array(point) - np.array(origin))
            ijk_points.append(ijk_point)
        return np.array(ijk_points)

    def transform_points_dict1(self, points_dict, trans_matrix, center, spacing):
        transformed_dict = {}
        for key, points in points_dict.items():
            # 如果需要调整点的顺序，可以在这里对整个数组进行操作
            points_reordered = points[:, [2, 1, 0]]

            # 将 points 从 (n, 3) 扩展到 (n, 4) 的齐次坐标形式，最后一列设为 1
            points_homogeneous = np.hstack([points_reordered, np.ones((points_reordered.shape[0], 1))])

            # 使用变换矩阵进行变换，结果是 (4, n)
            oriented_points_homogeneous = trans_matrix @ points_homogeneous.T

            center_homogeneous = np.append(center, 0)
            center_homogeneous_reshaped = center_homogeneous.reshape(-1, 1)  # 每-1取一个
            tmp_spacing = np.append(spacing, 1)
            tmp_spacing = tmp_spacing.reshape(-1 , 1)
            final_3d_landmarks_homogeneous = oriented_points_homogeneous * tmp_spacing - center_homogeneous_reshaped

            # 将结果从齐次坐标转换回原始坐标，即丢弃最后一维，并转换回原来的形状
            transformed_points = final_3d_landmarks_homogeneous[:3, :].T

            # 将转换后的点集存储回字典
            transformed_dict[key] = transformed_points

        return transformed_dict

    def transform_points_dict2(self, points_dict, trans_matrix, center, spacing):
        transformed_dict = {}
        for key, points in points_dict.items():
            # 调整点的顺序
            points_reordered = points[[2, 1, 0]]

            # 将点扩展到齐次坐标形式，添加 1 作为第四个维度
            point_homogeneous = np.append(points_reordered, 1)
            # 使用变换矩阵进行变换，结果是 (4,)
            oriented_point_homogeneous = trans_matrix @ point_homogeneous

            center_homogeneous = np.append(center, 0)
            tmp_spacing = np.append(spacing, 1)
            oriented_point_homogeneous = oriented_point_homogeneous * tmp_spacing - center_homogeneous

            # 将结果从齐次坐标转换回原始坐标，即丢弃最后一维
            transformed_point = oriented_point_homogeneous[:3]

            # 将转换后的点存储回字典
            transformed_dict[key] = transformed_point

        return transformed_dict

    def transform_points_dict3(self, points, trans_matrix, center, spacing):
        # 调整点的顺序
        points_reordered = points[:, [2, 1, 0]]

        # 将 points 从 (n, 3) 扩展到 (n, 4) 的齐次坐标形式，最后一列设为 1
        points_homogeneous = np.hstack([points_reordered, np.ones((points_reordered.shape[0], 1))])

        # 使用变换矩阵进行变换，结果是 (4, n)
        oriented_points_homogeneous = trans_matrix @ points_homogeneous.T

        # 将 center 从 (3,) 转换到齐次坐标形式 (4,)
        center_homogeneous = np.append(center, 0)
        center_homogeneous_reshaped = center_homogeneous.reshape(-1, 1)  # 每-1取一个
        tmp_spacing = np.append(spacing, 1)
        tmp_spacing = tmp_spacing.reshape(-1, 1)
        final_3d_landmarks_homogeneous = oriented_points_homogeneous * tmp_spacing - center_homogeneous_reshaped

        # 将结果从齐次坐标转换回原始坐标，即丢弃最后一维，并转换回原来的形状
        transformed_points = final_3d_landmarks_homogeneous[:3, :].T

        return transformed_points

    def inv_get_3d_landmarks(self, points, trans_matrix, center):
        # 使用变换矩阵进行变换
        center_homogeneous = np.append(center, 0)
        # 将齐次坐标形式的 center 重塑为 (4, 1) 以进行广播
        center_homogeneous_reshaped = center_homogeneous.reshape(-1, 1)  # 每-1取一个
        # 从变换后的齐次坐标点中减去齐次坐标形式的 center
        final_3d_landmarks_homogeneous = points.transpose(1,0) + center_homogeneous_reshaped
        oriented_points_homogeneous = np.linalg.inv(trans_matrix) @ final_3d_landmarks_homogeneous # 结果是 (4, 22)
        transformed_points = oriented_points_homogeneous[:3, :].T
        return transformed_points  # 转置以匹配原始点的布局

    def get_3d_landmarks(self, points, trans_matrix, center):
        # 将 points 从 (22, 3) 扩展到 (22, 4) 的齐次坐标形式，最后一列设为 1
        points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
        # # 使用变换矩阵进行变换
        # # oriented_points_homogeneous = trans_matrix @ points_homogeneous.T  # 结果是 (4, 22)
        # # 将 center 从 (3,) 转换到齐次坐标形式 (4,)
        # center_homogeneous = np.append(center, 0)
        # # 将齐次坐标形式的 center 重塑为 (4, 1) 以进行广播
        # center_homogeneous_reshaped = center_homogeneous.reshape(-1, 1) #每-1取一个
        # # 从变换后的齐次坐标点中减去齐次坐标形式的 center
        # final_3d_landmarks_homogeneous = points_homogeneous.transpose(1,0) - center_homogeneous_reshaped
        return points_homogeneous  # 转置以匹配原始点的布局

    def get_2d_landmarks(self, points, plane_info, key):
        plane_normal = plane_info['normal']
        plane_center = plane_info['center']
        plane_normal = np.array(plane_normal, dtype=np.float64)
        plane_center = np.array(plane_center, dtype=np.float64)
        points = np.array(points, dtype=np.float64)
        Points_on_3D = self.project_points_to_plane(points, plane_normal, plane_center)
        # 获取2D坐标点
        axis1, axis2 = self.select_axes_based_on_smallest_angle(plane_normal)

        Points_on_2d = self.project_points_to_2d(Points_on_3D, plane_normal, plane_center, axis1, axis2)
        # 转换为C形臂的2D坐标点
        c_arm_plane_2d = self.map_2d_points_to_pixels(Points_on_2d, key)
        return c_arm_plane_2d

    def select_axes_based_on_smallest_angle(self, normal):
        """
        根据与当前平面法向量构成的夹角最小选择两个轴构成的平面
        :param normal: 当前平面的法向量
        :return: 选择的两个轴
        """
        # 定义三个基本平面的法向量
        normal_ri = np.array([0, 0, 1])  # RI平面的法向量
        normal_ia = np.array([1, 0, 0])  # IA平面的法向量
        normal_ra = np.array([0, 1, 0])  # RA平面的法向量

        # 计算当前平面法向量与基本平面法向量的点积
        dot_ri = abs(np.dot(normal, normal_ri))
        dot_ia = abs(np.dot(normal, normal_ia))
        dot_ra = abs(np.dot(normal, normal_ra))

        # 选择点积最大（即夹角最小）的平面
        max_dot = max(dot_ri, dot_ia, dot_ra)

        # if max_dot == dot_ri:
        #     return np.array([1, 0, 0]), np.array([0, 1, 0])  # 选择RI平面，即X和Y轴
        # elif max_dot == dot_ia:
        #     return np.array([0, 0, 1]), np.array([0, 1, 0])  # 选择IA平面，即Z和Y轴
        # else:
        #     return np.array([1, 0, 0]), np.array([0, 0, 1])  # 选择RA平面，即X和Z轴
        return np.array([1, 0, 0]), np.array([0, 1, 0])  # 选择RI平面，即X和Y轴

    def abs(self, number):
        return np.abs(number)

    # def map_2d_points_to_pixels(self, coords_2d, pixel_size=(256,256)):
    #     """
    #     将相对于中心的2D点坐标映射到标准的像素坐标系（左上角为原点）
    #     :param coords_2d: 2D坐标数组，维度为 (n, 2)
    #     :param pixel_size: 目标像素范围，形式为 (width, height)
    #     :return: 映射到像素范围内的2D坐标数组
    #     """
    #     # 计算2D坐标的范围
    #     x_min, y_min = np.min(coords_2d, axis=0)
    #     x_max, y_max = np.max(coords_2d, axis=0)
    #
    #     lenghth = np.max(self.abs(x_min),self.abs(y_min), self.abs(x_max), self.abs(y_max))
    #
    #     # 计算2D坐标的实际宽度和高度
    #     width = x_max - x_min
    #     height = y_max - y_min
    #
    #     # 防止除以零
    #     if width == 0 or height == 0:
    #         raise ValueError("坐标范围的宽度或高度不能为零。")
    #
    #     # 计算缩放比例
    #     scale_x = pixel_size[0] / width
    #     scale_y = pixel_size[1] / height
    #     scale = min(scale_x, scale_y)  # 保持纵横比
    #
    #     # 应用缩放和平移使坐标居中
    #     coords_scaled = (coords_2d - [x_min, y_min]) * scale
    #
    #     # 为了使图像居中，计算需要的额外平移
    #     x_offset = (pixel_size[0] - width * scale) / 2
    #     y_offset = (pixel_size[1] - height * scale) / 2
    #
    #     coords_pixels = coords_scaled
    #
    #     # # 将Y坐标反转以匹配像素坐标系（因为Y轴在像素坐标系中是向下的）
    #     # coords_pixels[:, 1] = pixel_size[1] - coords_pixels[:, 1]
    #
    #     return coords_pixels
    # def map_2d_points_to_pixels(self, coords_2d, pixel_size=(256, 256), inner_size=224, fixed_scale=1.5):
    #     """
    #     将相对于中心点的2D点坐标映射
    #     :param coords_2d: 2D坐标数组，维度为 (n, 2)，原点为中心点
    #     :param pixel_size: 画布的像素大小
    #     :param inner_size: 内部缩放区域的大小，考虑边缘留白
    #     :return: 映射并缩放后的2D坐标数组
    #     """
    #     # 计算所有点相对于中心点的最大偏移量
    #     max_offset = np.max(np.abs(coords_2d))
    #
    #     # 防止除以零
    #     if max_offset == 0:
    #         raise ValueError("坐标的最大偏移量不能为零。")
    #
    #     # 计算缩放比例，以便在inner_size大小的正方形内映射所有点
    #     scale = inner_size / (2 * max_offset)  # 2*max_offset是原始正方形的全长
    #     coords_2d = np.array(coords_2d, dtype=np.float64)
    #     scale = np.array(scale, dtype=np.float64)
    #
    #     if scale >= fixed_scale:
    #         changed_scale = fixed_scale
    #     else:
    #         changed_scale = scale
    #
    #     # 缩放坐标
    #     coords_scaled_tmp = coords_2d * changed_scale
    #     # 计算缩放后的最大偏移量，确保不超出inner_size
    #     max_scaled_offset = np.max(np.abs(coords_scaled_tmp))
    #     if max_scaled_offset > pixel_size[0] / 2:
    #     # 如果缩放后的最大偏移量超出了预期范围，则进一步调整缩放比例
    #         changed_scale = scale
    #
    #     coords_scaled = coords_2d * changed_scale
    #     cover_size = np.max(np.abs(coords_scaled))
    #
    #     # 平移到新的256x256画布的中心
    #     center_offset = (np.array(pixel_size) - cover_size*2) / 2  # 计算内部区域与画布边界的偏移
    #     coords_translated = coords_scaled + center_offset + cover_size*2 / 2  # 加上内部区域的一半是为了从原点移至中心
    #     coords_translated[:, 0] = pixel_size[0] - coords_translated[:, 0]
    #
    #     return coords_translated

    def map_2d_points_to_pixels(self, coords_2d, key, image_size=256, desired_image=220, spacing=5):
        """
        将相对于中心点的2D点坐标映射
        :param coords_2d: 2D坐标数组，维度为 (n, 2)，原点为中心点
        :param pixel_size: 画布的像素大小
        :param inner_size: 内部缩放区域的大小，考虑边缘留白
        :return: 映射并缩放后的2D坐标数组
        """

        coords_2d = np.array(coords_2d, dtype=np.float64)

        if 'ra_' in self.pose_list[key] and 'lowbranch' in self.pose_list[key]:
            spacing = 30
        elif 'ra_' in self.pose_list[key]:
            spacing = spacing
        elif 'lowbranch' in self.pose_list[key]:
            spacing = spacing
        else:
            pass
        
        scale = (image_size - 2*spacing) / desired_image  # 2*max_offset是原始正方形的全长
        scale = np.array(scale, dtype=np.float64)

        # 缩放坐标
        coords_scaled = coords_2d * scale

        if 'ra_' in self.pose_list[key] and 'lowbranch_' in self.pose_list[key]:
            pass
        elif 'ra_' in self.pose_list[key]:
            coords_scaled[:, 1] = coords_scaled[:, 1] + 40
        elif 'lowbranch_' in self.pose_list[key]:
            coords_scaled[:, 1] = coords_scaled[:, 1] - 40  # hard-code here,调整的是在256的成像空间的位置，当然也可以调整物理空间的，这样的话和上面的if融合在一起
        else:
            pass

        # 平移到新的256x256画布的中心
        center_offset = np.array((image_size/2, image_size/2), dtype=np.float64)   # 计算内部区域与画布边界的偏移
        coords_translated = coords_scaled + center_offset   # 加上内部区域的一半是为了从原点移至中心
        coords_translated[:, 0] = image_size - coords_translated[:, 0] #翻转一下，适应PnP算法

        return coords_translated

    def project_vector_to_plane(self, vector, normal):
        """将向量投影到平面上"""
        # 计算向量在法向量方向上的分量
        projection_length = np.dot(vector, normal)
        # 从向量中减去其在法向量方向上的分量得到投影
        projection = vector - projection_length * normal
        return projection

    def project_points_to_2d(self, points, normal, center, axis1, axis2):
        """
        将3D点投影到2D平面，并获取2D坐标
        :param points: 投影到3D平面的点，维度为 (n, 3)
        :param normal: 平面的法向量
        :param center: 平面的中心点
        :param axis1: 第一个基向量对应的3D坐标系轴
        :param axis2: 第二个基向量对应的3D坐标系轴
        :return: 2D坐标数组，维度为 (n, 2)
        """
        # 计算3D坐标系轴在平面上的投影作为基向量
        points = np.array(points, dtype=np.float64)
        normal = np.array(normal, dtype=np.float64)
        axis1 = np.array(axis1, dtype=np.float64)
        axis2 = np.array(axis2, dtype=np.float64)

        center = self.project_single_point_to_plane(self.final_key_points['main_branch'], normal, center) #TODO:这里也可以改为AAA的位置 center = self.project_single_point_to_plane(self.AAA_pos, normal, center)
        # center = center

        base1 = self.project_vector_to_plane(axis1, normal)
        base2 = self.project_vector_to_plane(axis2, normal)

        # 单位化基向量
        base1 = base1 / np.linalg.norm(base1)
        base2 = base2 / np.linalg.norm(base2)

        # base1 = -base1  # 这里将base1取反，从而反转X坐标轴

        # 原点

        # 绘制向量
        # self.ax.quiver(*center, *base1, color='y', length=20, label='base1')
        # self.ax.quiver(*center, *base2, color='g', length=20, label='base2')

        # 计算投影点相对于中心点的偏移
        relative_positions = points - center

        # 计算2D坐标：点积得到每个点在新基向量下的分量
        x_coords = np.dot(relative_positions, base1)
        y_coords = np.dot(relative_positions, base2)

        # 合并x和y坐标
        coords_2d = np.column_stack((x_coords, y_coords))
        return coords_2d


    def merge_plane(self, normals, centers):
        """
        计算法向量的平均值，并归一化得到综合法向量
        :param normals: 法向量数组，维度为 (n, 3)
        :param centers: 中心点数组，维度为 (n, 3)
        :return: 综合法向量和所有局部中心点的平均值
        """
        # 计算法向量的平均值
        average_normal = np.mean(normals, axis=0)
        # 归一化综合法向量
        global_normal_normalized = average_normal / np.linalg.norm(average_normal)
        # 使用PCA找到法向量集合的主方向
        # pca = PCA(n_components=len(normals))
        # pca.fit(normals)
        # # PCA的第三个成分是最佳拟合平面的法向量
        # global_normal = pca.components_[-1]
        # global_normal_normalized = global_normal / np.linalg.norm(global_normal)
        # 计算所有局部中心点的平均值
        global_center = np.mean(centers, axis=0)

        return global_normal_normalized, global_center

    def get_original_volume(self, arr_volume, spacing_zyx, origin_zyx, direction_zyx):
        sitk_vol = sitk.GetImageFromArray(arr_volume)
        sitk_vol.SetSpacing(spacing_zyx[[2,1,0]]) #TODO: 这里的spacing， direction_matrix是numpy顺序的spacing，不应该重新交换顺序？？？？
        # direction_diag = np.diag(direction_zyx)
        # direction_diag_zxy = direction_diag[[2, 1, 0]]
        # direction = np.diag(direction_diag_zxy)
        direction = np.asarray(direction_zyx)
        sitk_vol.SetDirection(direction)  # 使用 flatten 更直观

        return sitk_vol

    def transformation_matrix(self, sitk_vol):
        matrix_temp = np.asarray(sitk_vol.GetDirection()).reshape(3, 3)
        spacing_temp = np.asarray(sitk_vol.GetSpacing())
        matrix_temp = matrix_temp * np.expand_dims(spacing_temp, 0).repeat(3, axis=0)
        origin_temp = np.asarray(sitk_vol.GetOrigin())
        ijk2_origin_Orient = np.identity(4)
        ijk2_origin_Orient[:3, :3] = matrix_temp
        ijk2_origin_Orient[:3, 3] = origin_temp

        sitk_orient_vol = sitk.DICOMOrient(sitk_vol, 'RIA')
        matrix_temp = np.asarray(sitk_orient_vol.GetDirection()).reshape(3, 3)
        spacing_temp = np.asarray(sitk_orient_vol.GetSpacing())
        matrix_temp = matrix_temp * np.expand_dims(spacing_temp, 0).repeat(3, axis=0)
        origin_temp = np.asarray(sitk_orient_vol.GetOrigin())
        ijk2_new_Orient = np.identity(4)
        ijk2_new_Orient[:3, :3] = matrix_temp
        ijk2_new_Orient[:3, 3] = origin_temp
        ijk_origin2new = np.linalg.inv(ijk2_new_Orient) @ ijk2_origin_Orient

        arr_vol_temp = sitk.GetArrayFromImage(sitk_orient_vol).transpose(2,1,0)
        volume_size_temp = arr_vol_temp.shape
        volume_center = (np.asarray(volume_size_temp) - 0) * np.asarray(spacing_temp) / 2.0

        return ijk_origin2new, volume_center, spacing_temp

    def define_plane_basis(self, plane_normal):
        # 选择一个与平面法向量不平行的向量（这里选择Z轴，如果法向量接近Z轴，可以选择其他轴）
        if np.allclose(plane_normal, [0, 0, 1]):
            arbitrary_vector = np.array([1, 0, 0])
        else:
            arbitrary_vector = np.array([0, 0, 1])
        # 计算基向量1：法向量与任意向量的叉积
        basis1 = np.cross(plane_normal, arbitrary_vector)
        basis1 /= np.linalg.norm(basis1)  # 归一化
        # 计算基向量2：法向量与基向量1的叉积
        basis2 = np.cross(plane_normal, basis1)
        basis2 /= np.linalg.norm(basis2)  # 归一化
        return basis1, basis2

    def convert_to_plane_coordinates(self, point, plane_center, basis1, basis2):
        # 将点转换到平面坐标系中
        relative_point = point - plane_center
        x = np.dot(relative_point, basis1)
        y = np.dot(relative_point, basis2)
        return np.array([x, y])

    def process_centerline_gragh(self):
        center_line_gragh = {}
        for index, centerline in enumerate(self.centerline_graph_input):
            center_points = np.zeros((len(centerline), 3), dtype=np.float32)
            for num, point_coords in enumerate(centerline):
                point = np.array(point_coords).astype(np.float32)
                # 重新排列轴以匹配 (z, x, y) 的顺序
                point_rearranged = point[[2, 1, 0]]
                # 将每个元素减去 1
                center_point = point_rearranged - 1
                # 更新 center_points 数组
                center_points[num] = center_point
            # 使用索引作为字典键
            center_line_gragh[index] = center_points
        return center_line_gragh

    def process_start(self):
        # mask = self.branches == 1
        # key_points = self.branches[mask] - 1
        start_point_indices = np.where(self.start == 1)
        start_points = np.array([point for point in zip(*start_point_indices)])
        return start_points

    def process_skel(self):
        # mask = self.branches == 1
        # key_points = self.branches[mask] - 1
        skel_point_indices = np.where(self.skel == 1)
        skel_points = np.array([point for point in zip(*skel_point_indices)])
        num_rows = skel_points.shape[0]
        selected_row_indices = np.random.choice(num_rows, size=400, replace=False)
        skel_points_selected = skel_points[selected_row_indices, :]
        return skel_points_selected

    def process_branch(self):
        # mask = self.branches == 1
        # key_points = self.branches[mask] - 1
        key_point_indices = np.where(self.branches == 1)
        key_points = np.array([point for point in zip(*key_point_indices)])
        return key_points

    def cluster_center_points(self, branch_points):
        # 计算所有点对的欧式距离
        Y = pdist(branch_points, 'euclidean')

        # 尝试不同的链接方法，如 'complete', 'average', 'ward'
        Z = linkage(Y, 'ward')

        # 调整阈值 t，找到最佳聚类效果
        t = 4  # 根据需要调整阈值
        clusters = fcluster(Z, t, criterion='distance')

        # 初始化一个数组来存储合并后的关键点
        key_points = []

        # 遍历每个簇，合并簇中的点，并计算它们的平均位置
        for cluster_id in np.unique(clusters):
            indices = np.where(clusters == cluster_id)[0]
            mean_point = np.mean(branch_points[indices], axis=0)
            key_points.append(mean_point)

        key_points = np.array(key_points)
        return key_points

    def match_points(self, key_analysis, key_predicted, AAA_pos):
        """
            找到与高斯核最大概率坐标点距离最近的参考点。

            :param max_prob_points: 高斯核最大概率的坐标点列表，格式为[(z1, y1, x1), (z2, y2, x2), ...]
            :param reference_points: 参考坐标点列表，格式为[(z1, y1, x1), (z2, y2, x2), ...]
            :return: 替换后的坐标点列表
            """
        closest_points = {}
        dict_key_list = self.dict_key_list
        count = 0
        for max_point in key_predicted:
            if count == 0:
                # 计算与所有参考点的距离
                distances = [np.linalg.norm(np.array(max_point) - np.array(ref_point)) for ref_point in key_analysis]
                # 找到最小距离的索引
                closest_index = np.argsort(distances)
                # 添加距离最近的参考点到列表
                closest_points[dict_key_list[count]] = key_analysis[closest_index[0]]
                count += 1
                closest_points[dict_key_list[count]] = key_analysis[closest_index[1]]
                count += 1

            elif 1 < count < 5:
                distances = [np.linalg.norm(np.array(max_point) - np.array(ref_point)) for ref_point in
                             key_analysis]
                # 找到最小距离的索引
                closest_index = np.argmin(distances)
                closest_points[dict_key_list[count]] = key_analysis[closest_index]
                count += 1
            else:
                pass
            closest_points['AAA'] = AAA_pos

        return closest_points

    def find_key_point_related_centerline(self, key_points, gragh):
        # del key_points['AAA']
        related_vessels = {i: [] for i in key_points.keys()}
        dict_key_list = ['ra_1', 'ra_2', 'AAA', 'main_branch', 'lowbranch_left', 'lowbranch_right']

        # 设置距离阈值，用于确定血管线段是否与分叉点相关联
        distance_threshold = 5  # 可以根据需要调整这个阈值

        # 遍历每个分叉点
        for name, key_point in key_points.items():
            if name != 'AAA':
                # 遍历每个血管线段
                for vessel_index, centerline in gragh.items():
                    # 计算分叉点与血管线段起始点之间的欧氏距离
                    distance_start = np.linalg.norm(key_point - centerline[0])
                    distance_end = np.linalg.norm(key_point - centerline[-1])
                    # 如果距离小于阈值，则认为这个血管线段与当前分叉点相关联
                    if distance_start < distance_threshold or distance_end < distance_threshold:
                        assert name in dict_key_list
                        related_vessels[name].append(vessel_index)
            else:
                for vessel_index, centerline in gragh.items():
                    break_flag = False
                    # 计算分叉点与血管线段起始点之间的欧氏距离
                    # mid_center = self.find_midpoint(centerline[0], centerline[-1])
                    for point in centerline:
                        distance = np.linalg.norm(key_point - point)
                        # 如果距离小于阈值，则认为这个血管线段与当前分叉点相关联
                        if distance < 10: #hard-code here
                            assert name in dict_key_list
                            related_vessels[name].append(vessel_index)
                            break_flag = True
                            break  # 跳出内层循环
                    if break_flag == True:
                        continue

        return related_vessels

    def get_points_within_distance(self, points, start_point, max_distance=20):
        """在给定距离内找到所有点，并返回这些点的列表。"""
        points_within_distance = []  # 用于存储距离小于等于max_distance的所有点

        for point in points:
            distance = self.calculate_distance(start_point, point)
            if distance <= max_distance:
                points_within_distance.append(point)

        return np.array(points_within_distance)

    def find_farthest_point_within_distance(self, points, start_point, max_distance=20):
        """在给定距离内找到最远点"""
        farthest_point = None
        farthest_distance = 0
        i = 0

        for i, point in enumerate(points):
            distance = self.calculate_distance(start_point, point)
            if distance <= max_distance and distance > farthest_distance:
                farthest_distance = distance
                farthest_point = point
            elif distance > max_distance:
                break
            else:
                pass

        return farthest_point, i

    def find_midpoint(self, point1, point2):
        """找到两点之间的中点"""
        return [(p1 + p2) / 2 for p1, p2 in zip(point1, point2)]

    def calculate_direction_vector(self, start_point, end_point):
        """计算从p1到p2的方向向量并标准化"""
        direction = np.array(end_point) - np.array(start_point)
        if np.linalg.norm(direction) == 0:
            return None  # 如果两点重合，返回 None
        return direction / np.linalg.norm(direction)

    def exclude_outliers(self, related_vessels, name, angle_threshold=25):
        to_remove = set()  # 使用集合避免重复添加

        # 计算所有血管线段的法向量并确保方向一致
        normals = {}
        direction_vectors = {}
        reference_vector = None  # 参考向量
        for vessel_index, centerline in related_vessels.items():
            if len(centerline) < 10:
                continue

            start_pt = self.final_key_points[name]
            point_exclude = self.get_points_within_distance(centerline, start_pt)
            # mid_pt = self.find_midpoint(start_pt, end_pt)
            # point_exclude = centerline[:end_i + 1]
            Points_exclude = Points(point_exclude.tolist())

            # normal = self.calculate_normal(start_pt, mid_pt, end_pt)

            # plane_ex = Plane.best_fit(Points_exclude)
            # normal = plane_ex.normal

            _, normal, _ = self.fit_plane(point_exclude, plot=False)

            if normal is None:
                normal = self.calculate_normal(start_pt, centerline[len(centerline)//2], centerline[-1])

            start_pt = centerline[0]
            end_pt = centerline[-1]  # 假设centerline是已经排序好的
            # direction_vector = self.calculate_direction_vector(start_pt, end_pt)

            line_ex = Line.best_fit(Points_exclude)

            if line_ex.direction is not None:
                direction_vectors[vessel_index] = line_ex.direction

            if normal is not None:
                if reference_vector is None:
                    reference_vector = normal  # 设置第一个有效法向量为参考向量
                else:
                    normal = self.ensure_same_direction(reference_vector, normal)
                normals[vessel_index] = normal

        # 对所有血管线段的法向量进行组合比较
        for vessel_index, normal_a in normals.items():
            angles = []
            for other_vessel_index, normal_b in normals.items():
                if other_vessel_index == vessel_index:
                    continue
                angle = self.calculate_angle_with_plane(direction_vectors[vessel_index], normal_b)
                angles.append(angle)

            # 如果与所有其他血管段的角度都大于阈值，则排除
            if all(180 - angle_threshold> angle > angle_threshold for angle in angles):
                to_remove.add(vessel_index)

        # 移除不一致的血管线段
        for vessel_index in to_remove:
            related_vessels.pop(vessel_index, None)

        return related_vessels, to_remove

    def angle_between_vectors(self, v1, v2):
        """计算两个向量之间的夹角（以度为单位）"""
        unit_v1 = v1 / np.linalg.norm(v1)
        unit_v2 = v2 / np.linalg.norm(v2)
        dot_product = np.dot(unit_v1, unit_v2)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        return np.degrees(angle)

    def calculate_angle_with_plane(self, direction_vector, plane_normal):
        """计算方向向量与平面法向量之间的夹角"""
        # 方向向量与平面法向量之间的夹角实际上等于方向向量与平面上的某个向量之间的互补角
        angle = self.angle_between_vectors(direction_vector, plane_normal)
        # 计算互补角
        return abs(90 - angle)

    def calculate_normal(self, p1, p2, p3):
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p1)
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm == 0:
            return None  # 三点共线
        return normal / norm

    def ensure_same_direction(self, reference_vector, vector):
        """确保两个向量方向一致"""
        if np.dot(reference_vector, vector) < 0:
            return -vector
        return vector

    def calculate_distance_batch(self, points, key_point):
        """
        计算点集中每个点到给定点的欧氏距离。
        :param points: 点集，numpy数组形式。
        :param key_point: 给定的单个点。
        :return: 距离数组。
        """
        return np.linalg.norm(points - key_point, axis=1)

    def construct_local_plane(self, key_point, centerlines_dict, choosed_dist=25):
        """
        使用至少两个向量构建局部平面，返回法向量。

        :param key_point: 分叉点坐标
        :param centerlines_dict: 各血管中心线点的字典
        :return: 平面的法向量
        """
        points_list = [key_point.tolist()]
        segment_max_list = []
        max_points = 30  # 最多使用的点数

        for vessel_index, centerline in centerlines_dict.items():
            distances = self.calculate_distance_batch(centerline, key_point)
            dist_max = np.max(distances)
            segment_max_list.append(dist_max)

        for vessel_index, centerline in centerlines_dict.items():
            # 检查 centerline 是否为 NumPy 数组
            if isinstance(centerline, np.ndarray):
                if len(centerline) < 10:
                    points_list.extend(centerline.tolist() if isinstance(centerline, np.ndarray) else centerline)
                    continue
                distances = self.calculate_distance_batch(centerline, key_point)

                if all(distances < choosed_dist):
                    warnings.warn('The chosen distance threshold is larger than the vessel segment distance')
                if np.min(np.array(segment_max_list)) > choosed_dist:
                    pass
                else:
                    choosed_dist = np.min(np.array(segment_max_list))

                close_indices = np.where(distances <= choosed_dist)[0]

                selected_indices = np.random.choice(close_indices,
                                                    size=min(len(close_indices), max_points),
                                                    replace=False)
                selected_points = centerline[selected_indices]
            else:
                # 目前代码没有走这个分支，之前代码的遗留。计算每个点到 key_point 的距离，并筛选出距离小于 choosed_dist 的点
                close_points = [point for point in centerline if
                                np.linalg.norm(np.array(point) - np.array(key_point)) < choosed_dist]
                # 随机选择最多 max_points 个点
                selected_points = random.sample(close_points, min(len(close_points), max_points - len(points_list)))

            points_list.extend(selected_points.tolist() if isinstance(selected_points, np.ndarray) else selected_points)
        # # 使用点集创建 Points 对象
        # points = Points(points_list)
        points = np.array(points_list)

        # 使用 Points 对象拟合平面，确保平面通过分叉点
        # plane = Plane.best_fit(points)
        model, normal_vector, center_point_proj = self.fit_plane(points, plot=True)

        # print('normal',normal_vector)
        # print('center',center_point_proj)

        return normal_vector, center_point_proj, points_list

    def project_single_point_to_plane(self, point, normal, center):
        """
        将一个点投影到平面上
        :param point: 要投影的点的坐标，维度为 (3,)
        :param normal: 平面的法向量
        :param center: 平面上的一点
        :return: 投影点的坐标，维度为 (3,)
        """
        # 计算点到平面中心点的向量
        vec = point - center
        # 计算向量在法线方向上的投影长度
        distance = np.dot(vec, normal.round(3))
        # 计算投影点
        projection_point = point - distance * normal.round(3)

        return projection_point

    def project_points_to_plane(self, points, normal, center):
        """
        将多个点投影到平面上
        :param points: 要投影的点的数组，维度为 (n, 3)
        :param normal: 平面的法向量
        :param center: 平面上的一点
        :return: 投影点的数组，维度为 (n, 3)
        """
        # 计算所有点到平面中心点的向量
        vecs = points - center
        # 使用 einsum 计算每个向量在法线方向上的投影长度
        distances = np.einsum('ij,j->i', vecs, normal.round(3))
        # 计算所有投影点
        projection_points = points - np.outer(distances, normal.round(3))

        # self.ax.scatter(projection_points[:, 0], projection_points[:, 1], projection_points[:, 2], color='r', label='full',
        #            alpha=1)
        self.ax.scatter(self.AAA_point[0], self.AAA_point[1], self.AAA_point[2], color='r', label='full',
                   alpha=1)
        return projection_points

    def calculate_distance(self, point1, point2):
        """计算两点之间的欧氏距离"""
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def fit_plane(self, points, plot=False):
        points = points.astype(np.float64)
        # 分离特征和目标变量
        X = points[:, :2]  # 取前两列作为特征（x, y）
        y = points[:, 2]  # 取第三列作为目标变量（z）

        # 创建并拟合模型
        model = LinearRegression()
        model.fit(X, y)

        # 从模型中提取法向量
        a, b = model.coef_
        c = model.intercept_
        normal_vector = np.array([a, b, -1])  # 平面的法向量

        # 计算数据点在平面上的投影点
        z_proj = a * X[:, 0] + b * X[:, 1] + c  # 使用平面方程计算投影点的 z 坐标
        points_proj = np.c_[X, z_proj]  # 构建投影点

        # 计算投影点的平均位置
        center_point_proj = np.mean(points_proj, axis=0)
        if plot:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')

            # 绘制原始数据点
            self.ax.scatter(X[:, 0], X[:, 1], y, color='b', label='Actual data')
            self.ax.scatter(self.skel_point[:, 0], self.skel_point[:, 1], self.skel_point[:, 2], color='g', label='full', alpha=0.2)

            # 绘制拟合平面
            xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 20, X[:, 0].max() +20, 40),
                                 np.linspace(X[:, 1].min() - 20, X[:, 1].max() +20, 40))
            zz = model.intercept_ + model.coef_[0] * xx + model.coef_[1] * yy
            self.ax.plot_surface(xx, yy, zz, color='r', alpha=0.5, label='Fitted plane')

            X, Y, Z = center_point_proj
            U, V, W = normal_vector

            # 使用 ax.quiver 绘制法向量
            # 注意：这里的 length 参数实际上不是 ax.quiver 的参数，如果需要调整箭头长度，请使用 scale 参数
            self.ax.quiver(X, Y, Z, U, V, W, color='g', arrow_length_ratio=0.1, length=10)

            R = np.array([1, 0, 0])
            I = np.array([0, 1, 0])
            A = np.array([0, 0, 1])

            # 原点
            origin = np.array([0, 0, 0])

            # 绘制向量
            # self.ax.quiver(*origin, *R, color='r', length=10, label='R')
            # self.ax.quiver(*origin, *I, color='g', length=10, label='I')
            # self.ax.quiver(*origin, *A, color='b', length=10, label='A')


            # self.ax.set_xlabel('X')
            # self.ax.set_ylabel('Y')
            # self.ax.set_zlabel('Z')
            self.ax.set_aspect('equal')
            plt.axis('off')
            # self.ax.legend()
            self.ax.view_init(10, -10)
            # plt.show()
        return model, normal_vector, center_point_proj
    
class PnPPoseEstimition:
    def __init__(self):
        '''


        '''
        self.dist_x = 742.5
        self.dist_y = 517.15
        self.senor_w = 256.0
        self.senor_h = 256.0
        self.pixel_size = 0.84375 * 2
        self.show_flag = False #是否绘制重投影误差的plot

    def PnP(self, Point3d, Point2d):
        self.point3d = Point3d[:, :3].astype(np.float64)
        # self.point3d = self.point3d[:, [2,1,0]]
        # Point3d = Point3d[:, [2, 1, 0, 3]]
        self.point2d = Point2d.astype(np.float64) #TODO: 为什么要翻转
        self.point2d_flipped = np.flip(Point2d, 1).astype(np.float64) #TODO: 为什么要翻转
        # self.point2d = Point2d.astype(np.float32) #TODO: 为什么要翻转

        camera_matrix = get_camera_intrinsics(self.dist_x, self.dist_y, self.senor_w, self.senor_h, self.pixel_size).astype(np.float32)
        dist_coeffs = np.zeros((4, 1))

        # success, rotation_vector, translation_vector = cv2.solvePnP(self.point3d, self.point2d, camera_matrix, dist_coeffs,
        #                                                             flags=0)
        # success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
        #     self.point3d,
        #     self.point2d,
        #     camera_matrix,
        #     dist_coeffs,
        #     flags=cv2.SOLVEPNP_ITERATIVE
        # )
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            self.point3d,
            self.point2d,
            camera_matrix,
            dist_coeffs,
            reprojectionError=8.0,  # 这是重投影误差阈值的示例值，通常以像素为单位
            confidence=0.99,  # 置信度，表示结果正确的概率
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        rotation_vector_optimized, translation_vector_optimized = self.robust_post_processing(self.point3d,
            self.point2d,
            camera_matrix,
            dist_coeffs,
            rotation_vector,translation_vector)

        rot_mat, jacobian = cv2.Rodrigues(rotation_vector_optimized)
        pose = self.calculate_CarmPose(rot_mat, translation_vector_optimized)

        # rot_mat, jacobian = cv2.Rodrigues(rotation_vector)
        # pose = self.calculate_CarmPose(rot_mat, translation_vector)
        pnp_error = self.pnp_loss(pose, Point3d, self.point2d_flipped, show=self.show_flag)
        return pose, success, pnp_error

    def robust_post_processing(self, point3d, point2d, camera_matrix, dist_coeffs, rotation_vector, translation_vector):
        # 设置Levenberg-Marquardt优化的停止条件
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-6)

        for i in range(3):  # 迭代3次进行优化和剔除异常值
            # 使用Levenberg-Marquardt算法进行位姿优化
            rotation_vector_optimized, translation_vector_optimized = cv2.solvePnPRefineLM(
                point3d,
                point2d,
                camera_matrix,
                dist_coeffs,
                rotation_vector,
                translation_vector,
                criteria
            )

            # 使用优化后的位姿参数计算重投影点
            projected_points, _ = cv2.projectPoints(point3d, rotation_vector_optimized, translation_vector_optimized,
                                                    camera_matrix, dist_coeffs)

            # 计算每个点的重投影误差
            errors = np.sqrt(np.sum((projected_points.squeeze() - point2d) ** 2, axis=1))

            # 计算误差的中位数，用于剔除异常值
            median_error = np.median(errors)

            # 选择误差小于1.5倍中位数的点作为内点
            inliers = errors < 1.5 * median_error

            # 更新点集，仅保留内点进行下一轮优化
            point3d = point3d[inliers]
            point2d = point2d[inliers]

            # 更新旋转向量和平移向量为优化后的值，用于下一轮迭代
            rotation_vector = rotation_vector_optimized
            translation_vector = translation_vector_optimized
        # print('median_error', median_error)

        return rotation_vector_optimized, translation_vector_optimized

    def pnp_loss(self, x, P3d, P2d, show=False):
        alpha, beta, gamma, tx, ty, tz = x
        proj_matrix = get_index_from_world(alpha, beta, gamma, tx, ty, tz,
                                           self.dist_x, self.dist_y, self.senor_w, self.senor_h, self.pixel_size)
        proj_3dto2d = proj_matrix @ P3d.T
        proj_3dto2d = (proj_3dto2d / np.expand_dims(proj_3dto2d[2, :], 0).repeat(3, axis=0)).T
        proj_3dto2d = proj_3dto2d[:, :2]
        proj_3dto2d = np.flip(proj_3dto2d, 1)  # YX-->XY
        error = np.abs(proj_3dto2d - P2d)
        if show:
            plt.figure(figsize=(8, 8))
            plt.scatter(proj_3dto2d.squeeze()[:, 0], proj_3dto2d.squeeze()[:, 1], c='red', label='2D Projection Points')
            plt.scatter(P2d[:, 0], P2d[:, 1], c='blue', label='2D Original Points')
            plt.title('Projection of 3D Points onto 2D Plane')
            plt.xlabel('X Axis')
            plt.ylabel('Y Axis')
            plt.legend()
            plt.grid(True)
            plt.show()

        loss = np.mean(error[:])

        return loss

    def calculate_CarmPose(self, rot_mat, translation_vector):
        proj_matrix = np.eye(4)
        proj_matrix[:3, :3] = np.squeeze(np.asarray(rot_mat))
        proj_matrix[:3, 3] = np.squeeze(np.asarray(translation_vector))

        proj_transform = np.array([-1, 0, 0, 0,
                                   0, 1, 0, 0,
                                   0, 0, -1, 0,
                                   0, 0, 0, 1]).reshape(4, 4)

        proj_matrix = proj_transform @ proj_matrix
        world_from_source_matrix = np.linalg.inv(proj_matrix)

        RzRxRy = world_from_source_matrix[:3, :3]
        T_vector = world_from_source_matrix[:3, 3]

        gamma, beta, alpha = rotation_angles(RzRxRy, order='zxy')
        tx = T_vector[0] - self.dist_x * RzRxRy[0, 2]
        ty = T_vector[1] - self.dist_x * RzRxRy[1, 2]
        tz = T_vector[2] - self.dist_x * RzRxRy[2, 2]

        pose = [alpha, beta, gamma, tx, ty, tz]
        return pose

class Postprocess3:
    def __init__(self, file_path, json_base):
        self.json_base = json_base
        self.file_base_path = file_path

        self.file_path = None
        self.spacing_zxy = None
        self.origin_zxy = None
        self.gt_points = None
        self.key_points = None
        self.img_arr = None
        self.dist = None

    def extract_number(self, filename):
        result = re.search(r'key(\d+)', filename)
        if result:
            return int(result.group(1))
        return None

    # 使用提取的数字对列表进行排序
    def run(self, file):
        self.group_array = []
        file = sorted(file, key=self.extract_number)
        for nii in file:
            self.file_path = os.path.join(self.file_base_path, nii)
            data_array = self.read_image(self.file_path)
            # 将数组添加到组列表中
            self.group_array.append(data_array)
            extracted_number = re.findall(r'\d+', nii)[0]
        extracted_number = int(extracted_number)
        # 沿新维度合并数组
        self.combined_array = np.stack(self.group_array, axis=0)

        gt_file_name = file[0][:9]
        gt_path = os.path.join('/home/yiwen/guidedresearch/nnUNet/nnUNet_preprocessed/Dataset027_Aorta/nnUNetPlans_3d_fullres', gt_file_name+'_key.npz')
        self.gt = np.load(gt_path)['key']

        # 构造新的 JSON 路径
        # 使用正则表达式替换路径中的数字
        # json_path = re.sub(r'Patient_\d+', f'Patient_{extracted_number}', self.json_base)
        json_path = os.path.join(self.json_base, gt_file_name + '.json')
        self.spacing_zxy, self.origin_zxy, self.gt_points = self.process_json(json_path)
        self.key_points = self.process_img(self.combined_array)
        self.dist, self.distance_variance = self.calculate_dist(self.key_points, self.gt_points)
        self.mse, self.mae, self.var = self.calculation_metric(self.gt, self.combined_array)

    def calculation_metric(self, y_true, y_pred):
        # 在 (128,160,112) 上计算 (y_true - y_pred) ** 2 的和
        squared_errors = (y_true - y_pred) ** 2
        sum_squared_errors = squared_errors.sum(axis=(1, 2, 3))  # 对最后三个维度求和

        # 在第一个维度 (4) 上计算均值
        mse = sum_squared_errors.mean()  # 在第一个维度求均值得到 MSE

        # 计算 MAE，同样先计算绝对误差之和，再计算均值
        abs_errors = np.abs(y_true - y_pred)
        sum_abs_errors = abs_errors.sum(axis=(1, 2, 3))  # 对最后三个维度求和
        mae = sum_abs_errors.mean()  # 在第一个维度求均值得到 MAE

        # 计算方差
        var = sum_squared_errors.var()  # 计算平方误差和的方差

        return mse, mae, var

    def calculate_dist(self, arr1, arr2):
        differences = arr1 - arr2
        differences = differences * self.spacing_zxy
        # 使用欧几里得距离公式计算距离
        distances = np.sqrt(np.sum(differences ** 2, axis=1))
        # 在第一个维度（4这个维度）上取平均
        average_distance = np.mean(distances)
        # 计算方差
        distance_variance = np.var(distances)
        return average_distance, distance_variance

    def process_img(self, img_array):
        n, c, h, w = img_array.shape
        center_predict = np.zeros((n, 3), dtype=np.float32)

        for channel_idx in range(n):  # 这里应该是range(n)，而不是range(c)
            output = img_array[channel_idx]
            mask_output = (output > 0)
            if not np.any(mask_output):  # 如果mask全为False，则跳过
                raise ValueError('The prediction all zero!')

            max_val_predict = np.max(output[mask_output]) - 0.02
            # 获取所有最大值点的坐标
            max_value_points = np.argwhere(output >= max_val_predict)
            # 计算这些坐标的平均值
            average_point = np.mean(max_value_points, axis=0)
            center_predict[channel_idx, :] = average_point

        return center_predict

    def grouped_read_image(self,folder_path, group_id):
        # 创建一个列表来保存该组的数组
        group_arrays = []

        # 遍历文件夹中的所有文件
        for file_name in os.listdir(folder_path):
            # 检查文件是否属于当前组
            if f"arota_{group_id:03}" in file_name:
                # 构建文件的完整路径
                file_path = os.path.join(folder_path, file_name)
                # 使用nibabel读取nii.gz文件
                data_array = self.read_image(file_path)
                # 将数组添加到组列表中
                group_arrays.append(data_array)

        # 沿新维度合并数组
        combined_array = np.stack(group_arrays, axis=-1)

        return combined_array

    def read_image(self,file_path):
        '''
        使用simpleitk读取nii文件
        '''
        original_image = sitk.ReadImage(file_path)
        img_arr = sitk.GetArrayFromImage(original_image)
        return img_arr

    def process_json(self, json_path):
        with open(json_path, 'r') as key_file_path:
            key_json = json.load(key_file_path)
        key_points_ras = np.array(key_json['ras_points'])
        direction_matrix = np.array(key_json['IJKtoRASDirectionMatrix'])
        origin = np.array(key_json['ImageOrigin'])
        spacing = np.array(key_json['ImageSpacing'])

        key_points_ras_zxy = key_points_ras[:, [2, 1, 0]]

        direction_diag = np.diag(direction_matrix)
        direction_diag_zxy = direction_diag[[2, 1, 0]]
        direction_matrix_zxy = np.diag(direction_diag_zxy)
        origin_zxy = origin[[2, 1, 0]]
        spacing_zxy = spacing[[2, 1, 0]]

        ijk_points = self.transform_points(key_points_ras_zxy, spacing_zxy, origin_zxy, direction_matrix_zxy)
        return spacing_zxy, origin_zxy, ijk_points

    def transform_points(self, points: np.ndarray, spacing: np.ndarray, origin: np.ndarray, direction_matrix: np.ndarray):
        scale_matrix = np.diag(spacing)
        transform_matrix = np.dot(direction_matrix, scale_matrix)
        ijk_points = []
        for point in points:
            ijk_point = np.dot(np.linalg.inv(transform_matrix),
                               np.array(point) - np.array(origin))
            ijk_points.append(ijk_point)
        return np.array(ijk_points)

def transform_points(points: np.ndarray, spacing: np.ndarray, origin: np.ndarray, direction_matrix: np.ndarray):
    scale_matrix = np.diag(spacing)
    transform_matrix = np.dot(direction_matrix, scale_matrix)
    ijk_points = []

    for point in points:
        ijk_point = np.dot(np.linalg.inv(transform_matrix),
                       np.array(point) - np.array(origin))
        ijk_points.append(ijk_point)
    return np.array(ijk_points)

def transform_point(point: np.ndarray, spacing: np.ndarray, origin: np.ndarray, direction_matrix: np.ndarray):
    scale_matrix = np.diag(spacing)
    transform_matrix = np.dot(direction_matrix, scale_matrix)
    ijk_points = []

    ijk_point = np.dot(np.linalg.inv(transform_matrix),
                       np.array(point) - np.array(origin))
    ijk_points.append(ijk_point)
    return np.array(ijk_point)

if __name__ == '__main__':
    # test_index = 4
    # test_patinet_list = ['arota_009', 'arota_017', 'arota_018', 'arota_019', 'arota_024']
    # choosed_file = test_patinet_list[test_index]
    # save_path = os.path.join(r'E:\results\final_result', choosed_file + '.npy')
    #
    # base_mat_path = r'E:\results\mat'
    # npy_path = r'E:\results\processed.npy'
    # with open(npy_path, 'rb') as file:
    #     loaded_data = pickle.load(file)
    # mat_path = os.path.join(base_mat_path, choosed_file + '.mat')
    #
    # ijk_points = loaded_data[1][choosed_file]
    # AAA_pos_IJK = loaded_data[0][test_index]
    # # AAA_pos_IJK = [0,0,0]
    # spacing_zxy = loaded_data[2][choosed_file]['spacing']
    # origin_zxy = loaded_data[2][choosed_file]['origin']
    # process2 = Postprocess2(mat_path, save_path, ijk_points, AAA_pos_IJK, spacing_zxy, origin_zxy)
    # plt.show()
    ####################################################Original Test ############################################################

    # file_path = r'E:\GuidedResearchProject\nnUNet\nnUNet_results\Dataset027_Aorta\nnUNetTrainer__nnUNetPlans__3dfull\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_4\validation\arota_009.nii.gz'
    save_path = r'E:\GuidedResearchProject\nnUNet\nnUNet_results\Dataset027_Aorta\nnUNetTrainer__nnUNetPlans__3dfull\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_4\validation\arota_009testclass.nii.gz'
    # process1 = Postprocess1(file_path,save_path)
    mat_file_path = r'E:\results\mat\arota_018.mat'
    json_path = r'E:\key_points\Patient_18\Patient_18.json' #在这里暂时使用gt
    with open(json_path, 'r') as key_file_path:
        key_json = json.load(key_file_path)
    key_points_ras = np.array(key_json['ras_points'])
    direction_matrix = np.array(key_json['IJKtoRASDirectionMatrix'])
    origin = np.array(key_json['ImageOrigin'])
    spacing = np.array(key_json['ImageSpacing'])

    key_points_ras_zxy = key_points_ras[:, [2, 1, 0]]

    direction_diag = np.diag(direction_matrix)
    direction_diag_zxy = direction_diag[[2, 1, 0]]
    direction_matrix_zxy = np.diag(direction_diag_zxy)
    origin_zxy = origin[[2, 1, 0]]
    spacing_zxy = spacing[[2, 1, 0]]

    ijk_points = transform_points(key_points_ras_zxy, spacing_zxy, origin_zxy, direction_matrix_zxy)

    AAA_pos = key_json['AAA_pos']
    AAA_pos_zyx = [AAA_pos[2], AAA_pos[1], AAA_pos[0]]
    AAA_pos_zyx = np.array(AAA_pos_zyx)
    AAA_pos_IJK = transform_point(np.array(AAA_pos_zyx), spacing_zxy, origin_zxy, direction_matrix_zxy)
    process2 = Postprocess2(mat_file_path, save_path, ijk_points, AAA_pos_IJK, spacing_zxy, origin_zxy)

    import matplotlib.pyplot as plt
    plt.show()


    #############################################################Not used code######################################################

    # def IJKRASTransformation(self, spacing, ijk2ras_direction, origin):
    #     #确保是np格式
    #     spacing = np.array(spacing)
    #     ijk2ras_direction = np.array(ijk2ras_direction).reshape(3, 3)
    #     origin = np.array(origin)
    #
    #     scale_matrix = np.diag(spacing)
    #     ijk2ras_matrix = np.dot(ijk2ras_direction, scale_matrix)
    #
    #     # 构建仿射变换矩阵，添加原点作为最后一列
    #     ijk2ras = np.eye(4)
    #     ijk2ras[:3, :3] = ijk2ras_matrix
    #     ijk2ras[:3, 3] = origin
    #
    #     ras2ijk = np.linalg.inv(ijk2ras)
    #     return ijk2ras, ras2ijk
    #
    # def RASLPITransformation(self, arr_volume, spacing, direction_matrix):
    #     '''
    #     arr_volume是任意一个volume即可
    #     '''
    #     # 创建SimpleITK图像对象
    #     sitk_vol = sitk.GetImageFromArray(arr_volume)
    #     sitk_vol.SetSpacing(spacing[[2,1,0]]) #TODO: 这里的spacing， direction_matrix是numpy顺序的spacing，不应该重新交换顺序？？？？
    #     sitk_vol.SetDirection(direction_matrix[[2,1,0]].flatten())  # 使用 flatten 更直观
    #     # 将图像方向调整为LPI
    #     sitk_orient_vol = sitk.DICOMOrient(sitk_vol, 'LIP')
    #     # 获取调整方向后的方向矩阵和间距，并计算新的变换矩阵
    #     RAS2LPI = np.diag([-1, -1, -1])
    #     # 使用 NumPy 广播而不是 repeat 来应用间距
    #     transform_matrix = RAS2LPI * spacing[:, np.newaxis]
    #     # matrix_temp = np.array(sitk_orient_vol.GetDirection()).reshape(3, 3)
    #     # spacing_temp = np.array(sitk_orient_vol.GetSpacing())
    #     # transform_matrix = matrix_temp * spacing_temp[:, np.newaxis]
    #     # 获取并设置新的原点
    #     origin_temp = np.array(sitk_orient_vol.GetOrigin())
    #
    #     ras2lpi = np.eye(4)
    #     # 假设 ijk2Orient 已经定义并初始化为恒等矩阵
    #     ras2lpi[:3, :3] = transform_matrix
    #     ras2lpi[:3, 3] = origin_temp
    #
    #     lpi2ras = np.linalg.inv(ras2lpi)
    #     return ras2lpi, lpi2ras

