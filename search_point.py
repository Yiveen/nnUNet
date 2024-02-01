import math
import vtk

def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)


centerlineNode = slicer.util.getNode('Centerline model_4')
pointListNode = slicer.util.getNode('F_3')

centerlinePolyData = centerlineNode.GetPolyData()
numberOfPoints = centerlinePolyData.GetNumberOfPoints()

# 遍历并获取所有点的坐标
allPoints = []
for i in range(numberOfPoints):
    point = [0.0, 0.0, 0.0]
    centerlinePolyData.GetPoint(i, point)
    allPoints.append(point)

final_dict = {}
ijk_list = []
name_list = ['ra_left', 'ra_right', 'main_branch', 'lowbranch_left', 'lowbranch_right', 'strat_point', 'AAA_pos']
# 遍历Point List中的所有点
for i in range(pointListNode.GetNumberOfControlPoints()):
    pointCoordinates = [0.0, 0.0, 0.0]
    pointListNode.GetNthControlPointPosition(i, pointCoordinates)
    pointListNode.SetNthControlPointLabel(i, name_list[i])
    pointName = pointListNode.GetNthControlPointLabel(i)
    # 比较坐标

# pointCoordinates = [0.0, 0.0, 0.0]
# pointListNode.GetNthControlPointPosition(0, pointCoordinates)

    # 找到最近的点
    min_distance = float('inf')
    closest_point = None
    for point in allPoints:
        dist = distance(pointCoordinates, point)
        if dist < min_distance:
            min_distance = dist
            closest_point = point

    print("Closest point coordinates: ", closest_point)
    closest_point = [round(num, 3) for num in closest_point]

    pointListNode.SetNthControlPointPosition(i, closest_point[0], closest_point[1], closest_point[2])

    final_dict[pointName] = closest_point
    ijk_list.append(closest_point)
print('final_dict',final_dict)
print('ijk_list', ijk_list)

