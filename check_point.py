import nibabel as nib
import numpy as np

# 替换为你的NIfTI文件路径和要修改的IJK点
nifti_file_path = r'E:\Newdataset\Patient_1\CTA_2.nii'
# nifti_file_path = '/home/yiwen/dataset/Newdataset/Patient_1/CTA_2.nii'
ijk_points = [[-167.852,-125.670,146.885], [-189.665,-123.619,190.222]]  # 示例IJK坐标点
radius = 2  # 修改像素的半径

# 加载NIfTI文件
nifti_img = nib.load(nifti_file_path)
print('shape',nifti_img.shape)
img_data = nifti_img.get_fdata()

# 计算RAS到IJK的转换矩阵
image_spacing = {"x": 0.5918, "y": 0.5918, "z": 1.000}
image_origin = {"x": 0, "y": 0, "z": 0}
direction_matrix = np.array([[-1.0000, 0.0000, 0.0000], [0.0000, -1.0000, 0.0000], [0.0000, 0.0000, 1.0000]])


# 考虑图像间距和方向矩阵
scale_matrix = np.diag([image_spacing['x'], image_spacing['y'], image_spacing['z']])
transform_matrix = np.dot(direction_matrix, scale_matrix)

ras_points = []
for ijk in ijk_points:
    ras_point = np.dot(np.linalg.inv(transform_matrix), np.array(ijk) - np.array([image_origin['x'], image_origin['y'], image_origin['z']]))
    ras_points.append(ras_point)

print('raspoints',ras_points)

# 修改指定IJK点及其周围的像素值
for ras in ras_points:
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                x, y, z = np.array(ras) + np.array([dx, dy, dz])
                if 0 <= x < img_data.shape[0] and 0 <= y < img_data.shape[1] and 0 <= z < img_data.shape[2]:
                    img_data[int(x), int(y), int(z)] = -1023  # 将像素值设置为黑色

# 创建新的NIfTI图像并保存
new_img = nib.Nifti1Image(img_data, nifti_img.affine)
nib.save(new_img, 'modified_image.nii')
print('saved!!')
