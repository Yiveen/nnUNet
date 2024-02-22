# import nibabel as nib
import numpy as np
import json
import SimpleITK as sitk


def transform_points(points: np.ndarray, spacing: np.ndarray, origin: np.ndarray, direction_matrix: np.ndarray):
    scale_matrix = np.diag(spacing)
    transform_matrix = np.dot(direction_matrix, scale_matrix)
    ijk_points = []
    for point in points:
        ijk_point = np.dot(np.linalg.inv(transform_matrix),
                           np.array(point) - np.array(origin))
        ijk_points.append(ijk_point)
    return np.array(ijk_points)

# 替换为你的NIfTI文件路径和要修改的IJK点
nifti_file_path = r'E:\Newdataset\Patient_6\CTA_2.nii'
# nifti_file_path = '/home/yiwen/dataset/Newdataset/Patient_1/CTA_2.nii'
radius = 5  # 修改像素的半径

# 加载NIfTI文件
# nifti_img = nib.load(nifti_file_path)
# print('shape',nifti_img.shape)
# img_data = nifti_img.get_fdata()

itk_img = sitk.ReadImage(nifti_file_path)
itk_img_array = sitk.GetArrayFromImage(itk_img)

print(itk_img_array.shape)

json_path = r'E:\key_points\Patient_6\Patient_6.json'

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

print('raspoints',ijk_points)

# 修改指定IJK点及其周围的像素值
for ijk in ijk_points:
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                # print('ijk_before', ijk)
                z, y, x = np.array(ijk) + np.array([dz, dy, dx])
                # print('ijk_end',[z,x,y])
                if 0 <= x < itk_img_array.shape[1] and 0 <= y < itk_img_array.shape[2] and 0 <= z < itk_img_array.shape[0]:
                    itk_img_array[int(z), int(y), int(x)] = -1023  # 将像素值设置为黑色

# 创建新的NIfTI图像并保存
# new_img = nib.Nifti1Image(img_data, nifti_img.affine)
# nib.save(new_img, 'modified_image.nii')
# 将NumPy数组转换回SimpleITK图像
modified_cta_img = sitk.GetImageFromArray(itk_img_array)
modified_cta_img.CopyInformation(itk_img)

# 保存新的NIfTI图像到磁盘
sitk.WriteImage(modified_cta_img, 'modified_image.nii')
print('saved!!')
