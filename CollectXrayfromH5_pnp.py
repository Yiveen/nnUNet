# -*- coding:utf-8 -*-
"""
@Time: 26/09/2023 14:50
@Author: Baochang Zhang
@IDE: PyCharm
@File: CollectXrayfromH5_final.py
@Comment: #Enter some comments at here
"""

import h5py as h5
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from Geometry_Proj_Matrix import get_index_from_world
from Geometry_Proj_Matrix import get_camera_intrinsics, rotation_angles
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt

# Device Geometry
# dist_x = 742.5
# dist_y = 517
dist_x = 530
dist_y = 1020 - dist_x
senor_w = 1536
senor_h = 1536
pixel_size = 0.194


def pnp_loss(x, P3d, P2d):
    alpha, beta, gamma, tx, ty, tz = x
    proj_matrix = get_index_from_world(alpha, beta, gamma, tx, ty, tz,
                                       dist_x, dist_y, senor_w, senor_h, pixel_size)
    proj_3dto2d = proj_matrix @ P3d.T
    proj_3dto2d = (proj_3dto2d / np.expand_dims(proj_3dto2d[2, :], 0).repeat(3, axis=0)).T
    proj_3dto2d = proj_3dto2d[:, :2]
    proj_3dto2d = np.flip(proj_3dto2d, 1)  # YX-->XY
    error = np.abs(proj_3dto2d - P2d)
    plt.figure(figsize=(6, 6))

    plt.scatter(P2d[:, 0], P2d[:, 1], c='red', label='gt2d')
    plt.scatter(proj_3dto2d[:, 0], proj_3dto2d[:, 1], c='blue', label='gt2d')
    plt.xlim(0, 1536)
    plt.ylim(0, 1536)
    plt.gca().invert_yaxis()  # Y轴反向，以符合图像坐标系统
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.title('2D Points Mapped to Pixel Coordinates')
    plt.legend()
    plt.grid(True)
    plt.show()
    loss = np.sum(error[:])
    return loss


def calculate_CarmPose(rot_mat, translation_vector):
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
    tx = T_vector[0] - dist_x * RzRxRy[0, 2]
    ty = T_vector[1] - dist_x * RzRxRy[1, 2]
    tz = T_vector[2] - dist_x * RzRxRy[2, 2]

    pose = [alpha, beta, gamma, tx, ty, tz]
    return pose


def collect_all_xray():
    h5_file_path = r'E:\3. ipcai_2020_full_res_data\ipcai_2020_full_res_data.h5'
    h5_data = h5.File(h5_file_path, 'r')
    PIDs = ['17-1882', '17-1905', '18-0725', '18-1109', '18-2799', '18-2800']
    save_root = r'E:\6. ipcai_2020_ds_8x'
    recordtxt = Path(save_root) / 'log.txt'
    if recordtxt.exists():
        os.remove(str(recordtxt))

    for id in range(6):

        folder_path = Path(save_root) / PIDs[id]
        if not folder_path.exists():
            folder_path.mkdir(parents=True)
        pat_data = h5_data[PIDs[id]]

        # extract CT 、 CT的np数组
        CT = pat_data['vol']
        arr_volume = np.asarray(CT['pixels'])
        origin = np.squeeze(np.asarray(CT['origin']))
        spacing = np.squeeze(np.asarray(CT['spacing']))
        direction_mat = np.squeeze(np.asarray(CT['dir-mat']))

        # convert volume to itkdata， 获取ct的np数组，转换为RAS（LPS）
        sitk_vol = sitk.GetImageFromArray(arr_volume)
        sitk_vol.SetSpacing(spacing)
        sitk_vol.SetDirection(direction_mat.reshape(-1))
        # sitk_vol.SetOrigin(origin) 为什么没有set origin

        # saved volume ~ IJK. LPS到IJK的转换矩阵
        IJK2LPS_matrix = np.eye(4, dtype=np.float32)
        IJK2LPS_matrix[:3, :3] = direction_mat * np.expand_dims(spacing, 0).repeat(3, axis=0)
        IJK2LPS_matrix[:3, 3] = origin
        IJK2LPS_matrix = np.asmatrix(IJK2LPS_matrix)
        LPS2IJK_matrix = np.linalg.inv(IJK2LPS_matrix)

        # convert to new ijk when LPS-->RIA， Due to physic position is same. IJK到LPS的转换
        matrix_temp = np.asarray(sitk_vol.GetDirection()).reshape(3, 3)
        spacing_temp = np.asarray(sitk_vol.GetSpacing())
        matrix_temp = matrix_temp * np.expand_dims(spacing_temp, 0).repeat(3, axis=0)
        origin_temp = np.asarray(sitk_vol.GetOrigin())
        ijk2LPS = np.identity(4)
        ijk2LPS[:3, :3] = matrix_temp
        ijk2LPS[:3, 3] = origin_temp

        #这里有个orient的概念 不是特别懂 可以大概理解为是IJK到标准空间 再到LPS
        sitk_orient_vol = sitk.DICOMOrient(sitk_vol, 'LIP')
        matrix_temp = np.asarray(sitk_orient_vol.GetDirection()).reshape(3, 3)
        debug = sitk_orient_vol.GetDirection() #为什么这里不再将顺序转换成numpy？？
        spacing_temp = np.asarray(sitk_orient_vol.GetSpacing())
        matrix_temp = matrix_temp * np.expand_dims(spacing_temp, 0).repeat(3, axis=0)
        origin_temp = np.asarray(sitk_orient_vol.GetOrigin())
        ijk2Orient = np.identity(4)
        ijk2Orient[:3, :3] = matrix_temp
        ijk2Orient[:3, 3] = origin_temp
        LPS2Orient_ijk = np.linalg.inv(ijk2Orient) @ ijk2LPS #这行解释不通

        arr_vol_temp = sitk.GetArrayFromImage(sitk_orient_vol).transpose(2, 1, 0)
        volume_size_temp = arr_vol_temp.shape
        volume_center_temp = np.asarray(volume_size_temp) * np.asarray(spacing_temp) / 2.0

        # extract 3D landmarks
        CT_landmarks = pat_data['vol-landmarks']
        marklist = list(CT_landmarks.keys())
        landmark_3d_world = dict()
        for index, markname in enumerate(marklist):
            lps_vector = np.array([0, 0, 0, 1], dtype=np.float32)
            lps_vector[:3] = np.squeeze(np.asarray(CT_landmarks[markname]))
            ijk_vector = LPS2IJK_matrix @ lps_vector
            ijk_lps = np.squeeze(np.asarray(ijk_vector))[:3] / spacing  # ijk ~ arr_volume.transpose(2, 1, 0) 没有spacing的ijk有什么含义呢？？还是这里是LPS点，但是考虑了origin？
            ijk_orient = np.squeeze(np.asarray(LPS2Orient_ijk @ np.mat(np.append(ijk_lps, 1)).T))[:3]  # ijk ~ sitk.GetArrayFromImage(sitk_RIAvol).transpose(2, 1, 0)
            landmark_3d_world[markname] = ijk_orient - volume_center_temp

        landmark_3d_world_data = []
        for markname in landmark_3d_world.keys():
            landmark_3d_world_data.append(np.append(landmark_3d_world[markname], 1))
        landmark_3d_world_data = np.asarray(landmark_3d_world_data).reshape(-1, 4)

        landmarks3dLIP_dict = dict()
        landmarks3dLIP_dict['landmark3d_LIP'] = landmark_3d_world
        landmarks3dLIP_dict['landmark3d_raw'] = CT_landmarks

        landmarkfilename = 'landmark3d.npz'
        # np.savez(str(folder_path / landmarkfilename), **landmarks3dLIP_dict)

        Xrays = pat_data['projections']
        XrayIDs = list(Xrays.keys())
        print('There are ', len(XrayIDs), 'Xrays')
        Xrayfolder_path = folder_path / 'Xray'
        if not Xrayfolder_path.exists():
            Xrayfolder_path.mkdir(parents=True)

        for ind, Xid in enumerate(XrayIDs):
            landmark_2d = dict()
            Xray = Xrays[Xid]
            rot180 = bool(np.asarray(Xray['rot-180-for-up']))
            Xrayimage = np.asarray(Xray['image']['pixels'])
            Xrayimage = (Xrayimage - Xrayimage.min()) / (Xrayimage.max() - Xrayimage.min())
            w, h = Xrayimage.shape
            xray_landmarks = Xray['gt-landmarks']  # 2D Anatomical landmarks in pixel coordinates
            marklist = list(xray_landmarks.keys())

            for index, markname in enumerate(marklist):
                ij = np.squeeze(np.asarray(xray_landmarks[markname]))
                ij = np.flip(ij, 0)
                landmark_2d[markname] = ij

            if rot180:
                # rotation of 180 deg. is equivalent to flipping columns, then flipping rows
                Xrayimage = np.flip(np.flip(Xrayimage, 0), 1)
                for markname in landmark_2d.keys():
                    landmark_2d[markname][0] = w - 1 - landmark_2d[markname][0]
                    landmark_2d[markname][1] = h - 1 - landmark_2d[markname][1]

            # need to flip Xray, must do it, Origin image is weird
            Xrayimage = np.flip(Xrayimage, 1)
            for markname in landmark_2d.keys():
                landmark_2d[markname][1] = h - 1 - landmark_2d[markname][1]

            # collect landmark2d to array
            landmark_2d_data = []
            for markname in landmark_3d_world.keys():
                landmark_2d_data.append(landmark_2d[markname])
            landmark_2d_data = np.asarray(landmark_2d_data).reshape(-1, 2)

            Point3d = landmark_3d_world_data[:, :3].astype(np.float32)
            Point2d = np.flip(landmark_2d_data, 1).astype(np.float32)
            camera_matrix = get_camera_intrinsics(dist_x, dist_y, senor_w, senor_h, pixel_size).astype(np.float32)
            dist_coeffs = np.zeros((4, 1))

            success, rotation_vector, translation_vector = cv2.solvePnP(Point3d, Point2d, camera_matrix, dist_coeffs, flags=0)
            rot_mat, jacobian = cv2.Rodrigues(rotation_vector)
            pose = calculate_CarmPose(rot_mat, translation_vector)
            print(pose)
            pnp_error = pnp_loss(pose, landmark_3d_world_data, landmark_2d_data)

            Xrayimage_256 = Image.fromarray(np.uint8(Xrayimage * 255.0)).resize((256, 256))
            Xrayimage_256 = np.asarray(Xrayimage_256)/255.0  # save this image

            rec_line = "FId: %s/%s -> pnp-error: %f" % (PIDs[id], Xid, pnp_error)
            print(rec_line)
            with open(recordtxt, "a") as f:
                f.write(rec_line + '\n')
            data_dict = dict()
            data_dict['arr'] = Xrayimage
            data_dict['arr_256'] = Xrayimage_256
            data_dict['pose'] = pose
            data_dict['pnp-error'] = pnp_loss(pose, landmark_3d_world_data, landmark_2d_data)
            data_dict['landmarks2d'] = landmark_2d
            data_dict['landmark3d_LIP'] = landmark_3d_world
            Xrayfilename = Xid + '.npz'
            np.savez(str(Xrayfolder_path / Xrayfilename), **data_dict)


if __name__ == '__main__':
    collect_all_xray()