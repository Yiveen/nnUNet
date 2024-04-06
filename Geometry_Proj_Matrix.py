# -*- coding:utf-8 -*-
"""
@Time: 26/09/2023 14:07
@Author: Baochang Zhang
@IDE: PyCharm
@File: Geometry_Proj_Matrix.py
@Comment: #Enter some comments at here
"""

import numpy as np
cos = np.cos
sin = np.sin
np.set_printoptions(suppress=True)


def rotation_angles(matrix, order):
    """
    input
        matrix = 3x3 rotation matrix (numpy array)
        oreder(str) = rotation order of x, y, z : e.g, rotation XZY -- 'xzy'
    output
        theta1, theta2, theta3 = rotation angles in rotation order
    """
    r11, r12, r13 = matrix[0]
    r21, r22, r23 = matrix[1]
    r31, r32, r33 = matrix[2]

    if order == 'xzx':
        theta1 = np.arctan(r31 / r21)
        theta2 = np.arctan(r21 / (r11 * np.cos(theta1)))
        theta3 = np.arctan(-r13 / r12)

    elif order == 'xyx':
        theta1 = np.arctan(-r21 / r31)
        theta2 = np.arctan(-r31 / (r11 *np.cos(theta1)))
        theta3 = np.arctan(r12 / r13)

    elif order == 'yxy':
        theta1 = np.arctan(r12 / r32)
        theta2 = np.arctan(r32 / (r22 *np.cos(theta1)))
        theta3 = np.arctan(-r21 / r23)

    elif order == 'yzy':
        theta1 = np.arctan(-r32 / r12)
        theta2 = np.arctan(-r12 / (r22 *np.cos(theta1)))
        theta3 = np.arctan(r23 / r21)

    elif order == 'zyz':
        theta1 = np.arctan(r23 / r13)
        theta2 = np.arctan(r13 / (r33 *np.cos(theta1)))
        theta3 = np.arctan(-r32 / r31)

    elif order == 'zxz':
        theta1 = np.arctan(-r13 / r23)
        theta2 = np.arctan(-r23 / (r33 *np.cos(theta1)))
        theta3 = np.arctan(r31 / r32)

    elif order == 'xzy':
        theta1 = np.arctan(r32 / r22)
        theta2 = np.arctan(-r12 * np.cos(theta1) / r22)
        theta3 = np.arctan(r13 / r11)

    elif order == 'xyz':
        theta1 = np.arctan(-r23 / r33)
        theta2 = np.arctan(r13 * np.cos(theta1) / r33)
        theta3 = np.arctan(-r12 / r11)

    elif order == 'yxz':
        theta1 = np.arctan(r13 / r33)
        theta2 = np.arctan(-r23 * np.cos(theta1) / r33)
        theta3 = np.arctan(r21 / r22)

    elif order == 'yzx':
        theta1 = np.arctan(-r31 / r11)
        theta2 = np.arctan(r21 * np.cos(theta1) / r11)
        theta3 = np.arctan(-r23 / r22)

    elif order == 'zyx':
        theta1 = np.arctan(r21 / r11)
        theta2 = np.arctan(-r31 * np.cos(theta1) / r11)
        theta3 = np.arctan(r32 / r33)

    elif order == 'zxy':
        theta1 = np.arctan(-r12 / r22)
        theta2 = np.arctan(r32 * np.cos(theta1) / r22)
        theta3 = np.arctan(-r31 / r33)

    theta1 = theta1 * 180 / np.pi
    theta2 = theta2 * 180 / np.pi
    theta3 = theta3 * 180 / np.pi

    return (theta1, theta2, theta3)


def Rx(beta):
    beta = beta / 180.0 * np.pi
    data = np.array([1, 0, 0, 0,
                     0, cos(beta), -sin(beta), 0,
                     0, sin(beta), cos(beta), 0,
                     0, 0, 0, 1])
    return data.reshape(4, 4)


def Ry(alpha):
    alpha = alpha / 180.0 * np.pi
    data = np.array([cos(alpha), 0, sin(alpha), 0,
                     0, 1, 0, 0,
                     -sin(alpha), 0, cos(alpha), 0,
                     0, 0, 0, 1])
    return data.reshape(4, 4)


def Rz(gamma):
    gamma = gamma / 180.0 * np.pi
    data = np.array([cos(gamma), -sin(gamma), 0, 0,
                     sin(gamma), cos(gamma), 0, 0,
                     0, 0, 1, 0,
                     0, 0, 0, 1])
    return data.reshape(4, 4)


def get_world_from_source(alpha, beta, gamma, tx, ty, tz, dist_x):
    R_x = Rx(beta)
    R_y = Ry(alpha)
    R_z = Rz(gamma)
    T_s = np.eye(4)
    T_s[2, 3] = dist_x
    matrix = R_z@R_x@R_y@T_s
    matrix[:3, 3] = matrix[:3, 3] + np.array([tx, ty, tz])
    return matrix


def get_source_from_world(alpha, beta, gamma, tx, ty, tz, dist_x):
    world_from_source_matrix = get_world_from_source(alpha, beta, gamma, tx, ty, tz, dist_x)
    return np.linalg.inv(world_from_source_matrix)


def get_camera3d_from_world(alpha, beta, gamma, tx, ty, tz, dist_x):

    proj = np.array([-1, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, -1, 0,
                     0, 0, 0, 1]).reshape(4, 4)

    Rot = get_source_from_world(alpha, beta, gamma, tx, ty, tz, dist_x)
    camera3d_from_world = proj@Rot
    return camera3d_from_world


def get_camera_intrinsics(dist_x=530, dist_y=490, senor_w=1536, senor_h=1536, pixel_size=0.194):
    fx = (dist_x+dist_y) / pixel_size
    fy = (dist_x+dist_y) / pixel_size
    cx = senor_w / 2
    cy = senor_h / 2
    data = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1])
    return data.reshape(3, 3)


def get_camera2d_from_camera3d():
    data = np.eye(4)
    data = data[:3, :]
    return data


def get_index_from_world(alpha, beta, gamma, tx, ty, tz,
                         dist_x=530, dist_y=490, senor_w=1536, senor_h=1536, pixel_size=0.194):

    camera3d_from_world = get_camera3d_from_world(alpha, beta, gamma, tx, ty, tz, dist_x)
    camera2d_from_camera3d = get_camera2d_from_camera3d()
    camera_intrinsics = get_camera_intrinsics(dist_x, dist_y, senor_w, senor_h, pixel_size)
    index_from_world = camera_intrinsics @ camera2d_from_camera3d @ camera3d_from_world
    return index_from_world




if __name__ == '__main__':
    alpha, beta, gamma, tx, ty, tz = [1, 2, 3, 4, 5, 6]
    dist_x = 742.5
    dist_y = 517.15
    senor_w = 256.0
    senor_h = 256.0
    pixel_size = 1.6875
    # physical space
    random_3d_point = np.random.randint(-100, 100, 30).reshape(-1, 3)
    random_3d_point = np.append(random_3d_point, np.ones(10).reshape(10, 1), axis=1)
    pnp3d = np.mat(random_3d_point).T
    proj_matrix = get_index_from_world(alpha, beta, gamma, tx, ty, tz,
                                       dist_x, dist_y, senor_w, senor_h, pixel_size)
    p2d = np.squeeze(np.asarray(proj_matrix @ pnp3d))[:3]
    p2d = (p2d/p2d[2])[:2]
    pnp_2d = np.flip(p2d, 0)
    print(pnp_2d)


