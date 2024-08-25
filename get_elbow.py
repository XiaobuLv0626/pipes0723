import numpy as np
import open3d as o3d
from cylinder import Cylinder
from torus import Torus
import globals
from tools import *
import math

'''
算法思路：给出拐弯的一个端点p1（默认为起始端）、中心p0、角度angle、环面半径、起始圆柱朝内的切向量cy_tang、法向量，计算拐弯
'''


''' 坐标变换过程可以参考 https://meta-factory.feishu.cn/wiki/BT0Kwao8Qi2QQ0kXCspcdbSZnBf 最底下的贴图，'''


def get_rotation_matrix(nor) -> 'np.array[[float, float, float], [float, float, float], [float, float, float]]':
    z = nor
    y = np.cross(z, np.array([0, 0, 1]))
    y = y / np.linalg.norm(y) if np.abs(np.linalg.norm(y)) > 1e-6 else np.array([0, 1, 0])
    x = np.cross(y, z)
    assert np.abs(np.linalg.norm(x)) > 1e-6
    x = x / np.linalg.norm(x)
    rotation_matrix = np.array([x, y, z])
    return rotation_matrix

'''以x轴为参考，使之坐标变换后重合'''
def get_local_rotate(nor,center_coord,nor1):
    # 在变换到世界坐标系前，首先需要确定torus的旋转角度，获得x轴在世界坐标系中的向量
    rotation_matrix = get_rotation_matrix(nor)
    x = np.array([[1, 0, 0]])
    point_x = o3d.geometry.PointCloud()
    point_x.points = o3d.utility.Vector3dVector(x)
    point_x.rotate(rotation_matrix.T, center=(0, 0, 0))
    point_x.translate(center_coord)
    world_x = np.asarray(point_x.points)[0]
    world_x -= center_coord
    world_x = world_x / np.linalg.norm(world_x)
    align_nor = -nor1
    align_nor = align_nor / np.linalg.norm(align_nor)
    align_angle = np.arccos(np.clip(np.dot(world_x, align_nor), -1, 1))
    # 判断逆时针或顺时针
    if np.dot((np.cross(world_x, align_nor)), nor) > 0:  # 平行、同向，逆时针旋转
        align_angle = -align_angle
    local_rotation_matrix=np.array([[np.cos(align_angle), np.sin(align_angle), 0],
                                     [-np.sin(align_angle), np.cos(align_angle), 0],
                                     [0, 0, 1]])  # 绕 z 轴旋转
    return align_angle


def get_world_coord(align_angle,nor,center_coord,coord1,coord2):
    '''得到世界坐标系下拐弯的两个端点'''
    coords = np.vstack((coord1, coord2))
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(coords)
    local_rotation_matrix = np.array([[np.cos(align_angle), np.sin(align_angle), 0],
                                      [-np.sin(align_angle), np.cos(align_angle), 0],
                                      [0, 0, 1]])  # 绕 z 轴旋转
    pcd.rotate(local_rotation_matrix, center=(0, 0, 0))
    rotation_matrix=get_rotation_matrix(nor)
    pcd.rotate(rotation_matrix.T, center=(0, 0, 0))
    pcd.translate(center_coord)
    world_coord1 = np.asarray(pcd.points)[0]
    world_coord2 = np.asarray(pcd.points)[1]
    return world_coord1,world_coord2


# 得到拐弯，p0、p1参考交付文档中的图示
def get_elbow(p0,p1,angle,radius,nor,cy_tang_start=None):
    cy_tang_start=-(p1-p0)/np.linalg.norm(p1-p0)
    # 计算夹角
    start_angle = 0 * np.pi
    end_angle = angle
    torus_radius=np.linalg.norm(p0-p1)
    align_angle = get_local_rotate(nor, p0, cy_tang_start)
    # xy平面的两个端点，旋转到世界坐标系中
    coord1=np.array([torus_radius*math.sin(start_angle),torus_radius*math.cos(start_angle),0.])
    coord2=np.array([torus_radius*math.sin(end_angle),torus_radius*math.cos(end_angle),0.])
    w_coord1,w_coord2=get_world_coord(align_angle,nor,p0,coord1,coord2)
    _partial_torus = Torus(torus_radius, radius, p0, start_angle, end_angle, nor, align_angle,w_coord1,w_coord2)
    return _partial_torus
