import numpy as np


def find_closest_points(max_prob_points, reference_points):
    """
    找到与高斯核最大概率坐标点距离最近的参考点。

    :param max_prob_points: 高斯核最大概率的坐标点列表，格式为[(z1, y1, x1), (z2, y2, x2), ...]
    :param reference_points: 参考坐标点列表，格式为[(z1, y1, x1), (z2, y2, x2), ...]
    :return: 替换后的坐标点列表
    """
    closest_points = []
    for max_point in max_prob_points:
        # 计算与所有参考点的距离
        distances = [np.linalg.norm(np.array(max_point) - np.array(ref_point)) for ref_point in reference_points]
        # 找到最小距离的索引
        closest_index = np.argmin(distances)
        # 添加距离最近的参考点到列表
        closest_points.append(reference_points[closest_index])

    return closest_points

def find_matching_points(key_point, centerline_points, distance_threshold1=4.5,distance_threshold2=5,
                         angle_threshold=np.pi / 6):
    """
    寻找与预定义方向向量夹角小于阈值的中心线点。

    :param key_point: 关键点的坐标，形如 [x, y, z]。
    :param centerline_points: 中心线点的坐标集合。
    :param direction_vectors: 一组预定义方向的单位向量列表。
    :param distance_threshold: 考虑的距离范围。
    :param angle_threshold: 方向一致性的夹角阈值，单位为弧度。
    :return: 符合条件的中心线点的列表。
    """
    matching_points = []

    direction_vectors = {
        'left': np.array([0, 1, 0]),  # 假设y轴正方向为左
        'right': np.array([0, -1, 0]),  # 假设y轴负方向为右
        'up': np.array([0, 0, 1]),  # 假设z轴正方向为上
        'down': np.array([0, 0, -1])  # 假设z轴负方向为下
    }

    for point in centerline_points:
        # 计算关键点到中心线点的距离
        distance = np.linalg.norm(point - key_point)

        # 如果点在指定范围内
        if distance_threshold1 <= distance <= distance_threshold2:
            # 计算从关键点到该点的方向向量并单位化
            direction = (point - key_point) / distance

            # 对每个预定义方向向量，检查夹角是否小于阈值
            match = any(np.dot(direction, dir_vec) > np.cos(angle_threshold) for dir_vec in direction_vectors)

            # 如果与所有预定义方向的夹角都大于阈值，则跳过
            if not match:
                continue

            # 否则，添加到结果列表
            matching_points.append(point)
            break

    return matching_points

def find_adjacent_points_and_vectors(fork_point, centerline_coords, radius=5):
    """
    在中心线坐标上寻找分叉点的相邻点，并计算向量。

    :param fork_point: 分叉点的坐标，格式为(z, y, x)
    :param centerline_coords: 血管中心线的坐标列表，每个元素的格式为(z, y, x)
    :param radius: 相邻点的搜索半径
    :return: 相邻点列表和向量列表
    """
    adjacent_points = []
    vectors = []

    for point in centerline_coords:
        # 计算分叉点到当前点的距离
        distance = np.linalg.norm(np.array(fork_point) - np.array(point))

        # 如果距离在指定半径内，则认为是相邻点
        if distance <= radius and distance != 0:  # 排除分叉点本身
            adjacent_points.append(point)
            # 计算向量
            vector = np.array(point) - np.array(fork_point)
            vectors.append(vector)

    return adjacent_points, vectors


def construct_local_plane(vectors):
    """
    使用至少两个向量构建局部平面，返回法向量。

    :param vectors: 向量列表，每个向量表示为一个三元组(z, y, x)
    :return: 平面的法向量
    """
    if len(vectors) < 2:
        raise ValueError("需要至少两个非平行向量来定义一个平面")

    # 使用前两个向量计算法向量（叉积）
    normal_vector = np.cross(vectors[0], vectors[1])
    return normal_vector

def project_point_to_plane(point, normal, plane_point):
    """
    将点投影到平面上
    :param point: 要投影的点
    :param normal: 平面的法向量
    :param plane_point: 平面上的一点
    :return: 投影点
    """
    # 计算点到平面的向量
    vec = point - plane_point
    # 计算向量在法线方向上的投影长度
    distance = np.dot(vec, normal)
    # 计算投影点
    projection = point - distance * normal
    return projection




# 假设normals是一个包含多个法向量的数组，每个法向量是一个3元组(x, y, z)
normals = np.array([...])
center_point = np.array([.5])

# 计算平均法向量
avg_normal = np.mean(normals, axis=0)
# 单位化平均法向量
avg_normal /= np.linalg.norm(avg_normal)

A, B, C = avg_normal
x0, y0, z0 = center_point
D = - (A * x0 + B * y0 + C * z0)

# 示例：将点 P 映射到平面上
P = np.array([...])  # 替换[...]为实际的点坐标
P_proj = project_point_to_plane(P, avg_normal, center_point)