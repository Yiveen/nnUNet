import numpy as np


def get_patch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    # 对于每个旋转轴（x、y、z），如果旋转角度是元组或列表（表示范围），则选择最大的绝对值作为最大旋转角度
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))

    # 限制旋转角度不超过90度，因为更大的旋转可能不会增加所需的补丁大小
    rot_x = min(90 / 360 * 2. * np.pi, rot_x)
    rot_y = min(90 / 360 * 2. * np.pi, rot_y)
    rot_z = min(90 / 360 * 2. * np.pi, rot_z)

    # 导入2D和3D旋转坐标的函数
    from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d

    coords = np.array(final_patch_size)  # 将最终补丁大小转换为NumPy数组
    final_shape = np.copy(coords)  # 复制一份作为最终形状的基础

    # 如果是3D数据，分别对x、y、z轴进行旋转，并更新最终形状以确保能覆盖旋转后的整个区域
    if len(coords) == 3:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
    # 如果是2D数据，只需要对x轴（或任意一个轴，因为在2D中旋转是等效的）进行旋转
    elif len(coords) == 2:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)

    # 考虑缩放范围的最小值（因为我们想要最大可能的补丁大小以覆盖所有情况），并更新最终形状
    final_shape /= min(scale_range) #scale_range (0.85, 1.25)

    return final_shape.astype(int)  # 将最终形状转换为整数并返回

