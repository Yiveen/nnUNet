import SimpleITK as sitk
import inspect
import multiprocessing
import os
import re
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Union, Tuple, List
import pickle
import numpy as np
import json

from typing import Union

from post_process_class import Postprocess1, Postprocess2

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
        y_pred_mask = y_pred[y_pred > 0]
        var = y_pred_mask.var()  # 计算平方误差和的方差

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

base_path = '/home/yiwen/'
# raw_key_path = '/home/yiwen/nnunet_two_stages/nnUNet/nnUNet_results/Dataset027_Aorta/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_real_test_2_final_regression2/validation'
# raw_key_path = '/home/yiwen/nnunet_two_stages/nnUNet/nnUNet_results/Dataset027_Aorta/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_real_test_2_final_regression11_loss1/validation'
raw_key_path = '/home/yiwen/nnunet_two_stages/nnUNet/nnUNet_results/Dataset027_Aorta/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_real_test_2_final_2_only_distloss/validation'
# raw_key_path = '/home/yiwen/nnunet_two_stages/nnUNet/nnUNet_results/Dataset027_Aorta/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_real_test_2_final_regression3_loss12/validation'
key_ijk_points = {}
properties_dict = {}
dist = []
mse = []
mae = []
distance_variance = []
var = []
group_id = [9 ,17 ,18 ,19 ,24]
json_base = '/home/yiwen/guidedresearch/nnUNet/nnUNet_raw/Dataset027_Aorta/keypoints'
Keyprocess = Postprocess3(raw_key_path, json_base)

file_grouped = {}
for id in group_id:
    grouped_name = []
    for file in os.listdir(raw_key_path):
        if f"arota_{id:03}" in file:
            grouped_name.append(file)
    file_grouped[id] = grouped_name

for id, file in file_grouped.items():
    AAA_result = Keyprocess.run(file)
    key_ijk_points[f"arota_{id:03}"] = np.array(Keyprocess.key_points) #or
    properties_dict[f"arota_{id:03}"] = {'spacing': Keyprocess.spacing_zxy, 'origin': Keyprocess.origin_zxy}
    # key_ijk_points[f"arota_{id:03}"] = np.array(Keyprocess.gt_points)
    dist.append(Keyprocess.dist)
    mse.append(Keyprocess.mse)
    mae.append(Keyprocess.mae)
    var.append(Keyprocess.var)
    distance_variance.append(Keyprocess.distance_variance)

dist = np.mean(np.array(dist))
mse = np.mean(np.array(mse))
mae = np.mean(np.array(mae))
var = np.mean(np.array(var))
distance_variance = np.mean(np.array(distance_variance))
print('dist' ,dist)
print('mse', mse)
print('mae', mae)
print('var', var)
print('distance_variance', distance_variance)

data_to_save = [key_ijk_points, properties_dict]
# 保存这个列表为 .npy 文件
with open(os.path.join(base_path, 'processed_2loss.npy'), 'wb') as file:
    pickle.dump(data_to_save, file)