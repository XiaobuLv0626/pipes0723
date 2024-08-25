import argparse
import math

import numpy as np
import open3d as o3d
from tqdm import tqdm
import torch

from cylinder import Cylinder
from anchor import Anchor
from tools import *
import globals
from get_elbow import get_elbow
from pytorch3d.ops import knn_points
from group import Group

'''拟合拐弯，注意圆柱的id和点云ransac_points.npy的第4列对应'''


def find_nearest_neighbors(pc1, pc2):
    # Convert point clouds to PyTorch tensors

    N = pc1.shape[0]

    if isinstance(pc1, np.ndarray):
        pc1_tensor = torch.from_numpy(pc1).cuda()
    else:
        pc1_tensor = pc1.cuda()
    if isinstance(pc2, np.ndarray):
        pc2_tensor = torch.from_numpy(pc2).cuda()
    else:
        pc2_tensor = pc2.cuda()

    idx = knn_points(pc1_tensor.unsqueeze(0), pc2_tensor.unsqueeze(0))[1].reshape((N,))

    return idx


def colpane(alternative1,alternative2):
    '''判断两个圆柱端点是否近似在拐弯拟合对应的平面内'''
    cylinder_index1, side1 = alternative1
    coord1=globals.para_top_bottom[cylinder_index1][side1]
    cylinder_index2, side2 = alternative2
    coord2=globals.para_top_bottom[cylinder_index2][side2]
    cylinder1=globals.load_cylinders[cylinder_index1]
    cylinder2=globals.load_cylinders[cylinder_index2]
    cy1_dir=get_correct_dir(cylinder1,side1)
    cy2_dir=get_correct_dir(cylinder2,side2)
    plane_nor = np.cross(cy1_dir,cy2_dir)
    plane_nor=plane_nor/np.linalg.norm(plane_nor)
    # 计算两个端点在平面法向量方向的投影距离
    args.threshold_radius = min(cylinder1.radius, cylinder2.radius)
    distance=np.abs(np.dot(plane_nor,coord2-coord1))
    if distance>args.threshold_radius:
        return False
    else:
        return True

# 预处理拐弯点云
def get_elbow_points():
    points_path="torus/torus_inst.npy"
    points_data=np.load(points_path)
    a=np.unique(points_data[:,6])
    max_num=int(max(a))
    pcd=o3d.geometry.PointCloud()
    torus_points=[]
    for x in range(max_num):
        # pcd.points=o3d.utility.Vector3dVector(points_data[points_data[:,6]==x][:,:3])
        # o3d.visualization.draw_geometries([pcd])
        torus_points.append(points_data[points_data[:,6]==x][:,:3])
    return torus_points

'''返回拐弯对象，但是存在None的情况'''
'''回归的复杂度高'''
def get_elbow_inst(points,alternative):
    points_center=np.mean(points,axis=0)
    cylinder_index, side=alternative
    cylinder=globals.load_cylinders[cylinder_index]
    side_coord=globals.para_top_bottom[cylinder_index][side]
    cy_dir=get_correct_dir(cylinder,side)
    plane_nor = np.cross(cy_dir,points_center-side_coord)
    plane_nor=plane_nor/np.linalg.norm(plane_nor)
    cy_tang=np.cross(plane_nor,cy_dir) # 切向量
    cy_tang=cy_tang/np.linalg.norm(cy_tang)
    # 回归拐弯中心和角度
    final_elbow=None
    best_count=0
    for p0 in np.linspace(side_coord+cy_tang*0.01,side_coord+cy_tang*0.5,10):
        # p0-=0.01*plane_nor
        # for dp0 in np.linspace(0,0.01,5):
        #     p0+=dp0*plane_nor
        #     plane_nor = np.cross(cy_dir, p0 - side_coord)

        for angle in np.linspace(np.pi/20,np.pi*2/3,10):
            # elbow = get_elbow(p0, side_coord, angle, cylinder.radius, cy_tang, plane_nor)
            elbow = get_elbow(p0, side_coord, angle, cylinder.radius, plane_nor,cy_tang)
            '''调整采样点数量并验证可行性'''
            vertices = np.asarray(elbow.to_o3d_mesh().sample_points_uniformly(500).points)
            idx = find_nearest_neighbors(points, vertices)
            idx = idx.cpu().numpy()
            dis = (np.linalg.norm(points - vertices[idx], axis=1))
            count = np.sum(dis < args.p_coverage_thresh)
            if count > best_count:
                best_count = count
                final_elbow=elbow
    return final_elbow,best_count

def find_intersect(coord1,dir1,coord2,dir2):
    '''下面求交算法默认一定有交点'''
    res = 1e9
    intersection_pos=None
    for i in np.arange(0, 0.25, 0.002):
        extent1=coord1+dir1*i
        for j in np.arange(0, 0.25, 0.002):
            extent2 = coord2+dir2*j
            if np.linalg.norm(extent1 - extent2) < 0.15:
                if res > np.linalg.norm(extent1 - extent2):
                    res = min(res, np.linalg.norm(extent1 - extent2))
                    intersection_pos = extent1
    return intersection_pos

def get_elbow_inst_2(points,alternative1,alternative2):
    cylinder_index1, side1 = alternative1
    cylinder1 = globals.load_cylinders[cylinder_index1]
    dir1 = get_correct_dir(cylinder1, side1)
    cylinder_index2, side2 = alternative2
    cylinder2 = globals.load_cylinders[cylinder_index2]
    dir2 = get_correct_dir(cylinder2, side2)
    side_coord1=globals.para_top_bottom[cylinder_index1][side1]
    side_coord2=globals.para_top_bottom[cylinder_index2][side2]
    plane_nor=np.cross(dir2,dir1)
    cy_tang=np.cross(plane_nor,dir1)
    cy_tang=cy_tang/np.linalg.norm(cy_tang)
    # 可以考虑在中垂线上回归p0，但可能都不能跟圆柱衔接，现在的结果已经足够了
    perpendicular_center=(side_coord1+side_coord2)/2.0
    perpendicular_dir=np.cross(side_coord1-side_coord2,plane_nor)
    perpendicular_dir=perpendicular_dir/np.linalg.norm(perpendicular_dir)
    p0=find_intersect(perpendicular_center,perpendicular_dir,side_coord1,cy_tang)
    angle=math.acos(np.abs(np.dot(p0-side_coord1,p0-side_coord2)))
    elbow = get_elbow(p0, side_coord1, angle, cylinder1.radius, plane_nor,cy_tang)
    return elbow


def cast_intersection(cylinder_idx, cylinder_side, cylinder_index2, cylinder_side2):
    intersection_pos = []
    args.threshold_radius = min(globals.load_cylinders[cylinder_idx].radius,
                                globals.load_cylinders[cylinder_index2].radius)
    # 获取指向所选面外侧的法向量
    cylinder_nor = get_correct_dir(globals.load_cylinders[cylinder_idx], cylinder_side)
    neighbor_nor = get_correct_dir(globals.load_cylinders[cylinder_index2], cylinder_side2)
    # 为了方便，将角度条件放在此处
    angle_cos = np.dot(neighbor_nor, cylinder_nor)
    if np.abs(angle_cos) > 0.8:
        return False, []

    flag = False
    res = 1e9
    for i in np.arange(0, 0.25, 0.002):
        extent_nor1 = globals.para_top_bottom[cylinder_idx][cylinder_side] + cylinder_nor * i
        for j in np.arange(0, 0.25, 0.002):
            extent_nor2 = globals.para_top_bottom[cylinder_index2][cylinder_side2] + neighbor_nor * j
            if np.linalg.norm(extent_nor1 - extent_nor2) < args.threshold_radius:
                flag = True
                if res > np.linalg.norm(extent_nor1 - extent_nor2):
                    res = min(res, np.linalg.norm(extent_nor1 - extent_nor2))
                    intersection_pos = [extent_nor1]
    return flag, intersection_pos

def extend_cylinder(alternative1,alternative2,intersection_pos):
    cylinder_index1, side1 = alternative1
    cylinder1 = globals.load_cylinders[cylinder_index1]
    dir1 = get_correct_dir(cylinder1, side1)
    cylinder_index2, side2 = alternative2
    cylinder2 = globals.load_cylinders[cylinder_index2]
    dir2 = get_correct_dir(cylinder2, side2)

    tangency1 = globals.para_top_bottom[cylinder_index1][side1]
    tangency2 = globals.para_top_bottom[cylinder_index2][side2]

    l1 = np.linalg.norm(intersection_pos - tangency1)
    l2 = np.linalg.norm(intersection_pos - tangency2)
    # 唯一更新圆柱参数的地方（修改长度）
    if l1 > l2:
        # 更新cylinder1
        if side1==TOP:
            globals.load_cylinders[cylinder_index1].top_center+=dir1*(l1-l2)
            temp=globals.para_top_bottom[cylinder_index1][side1] + dir1*(l1-l2)
            globals.para_top_bottom[cylinder_index1]=(temp,globals.para_top_bottom[cylinder_index1][1])
        else:
            globals.load_cylinders[cylinder_index1].bottom_center+=dir1*(l1-l2)
            temp=globals.para_top_bottom[cylinder_index1][side1] + dir1*(l1-l2)
            globals.para_top_bottom[cylinder_index1]=(globals.para_top_bottom[cylinder_index1][0],temp)

    else:
        # 更新cylinder2
        if side2==TOP:
            globals.load_cylinders[cylinder_index2].top_center+=dir2*(l2-l1)
            temp=globals.para_top_bottom[cylinder_index2][side2] + dir2 * (l2 - l1)
            globals.para_top_bottom[cylinder_index2]=(temp,globals.para_top_bottom[cylinder_index2][1])
        else:
            globals.load_cylinders[cylinder_index2].bottom_center+=dir2*(l2-l1)
            temp=globals.para_top_bottom[cylinder_index2][side2] + dir2 * (l2 - l1)
            globals.para_top_bottom[cylinder_index2]=(globals.para_top_bottom[cylinder_index2][0],temp)




def get_elbow_inst_from_sides(points, alternative1, alternative2):
    '''从两端拟合拐弯'''
    elbow=None
    points_center=np.mean(points,axis=0)
    cylinder_index1, side1 = alternative1
    cylinder_index2, side2 = alternative2
    # 如果两圆柱平行，用两个拐弯拟合
    # 但是不利于代码的统一性，平行圆柱放到后续再处理
    '''返回值存在其中一个为空的情况'''
    # if parallel(alternative1, alternative2):
    #     elbow1, count1 = get_elbow_inst(points, alternative1)
    #     elbow2, count2 = get_elbow_inst(points, alternative2)
    #     return elbow1, elbow2

    can_fit, intersection_pos = cast_intersection(cylinder_index1,side1,cylinder_index2,side2)
    if can_fit:
        # 计算需要延长的部分，通过get_elbow_inst得到内点数最多的拐弯
        extend_cylinder(alternative1,alternative2,intersection_pos)

        if colpane(alternative1,alternative2):
            elbow=get_elbow_inst_2(points,alternative1,alternative2)
            return elbow
        # 如果不共面，拟合一端即可
        # 但是在目前gt数据中，不存在这种情况
        else:
            elbow1, count1 = get_elbow_inst(points, alternative1)
            elbow2, count2 = get_elbow_inst(points, alternative2)
            if count1>count2:
                elbow=elbow1
            else:
                elbow=elbow2
            return elbow
    # 不相交，拟合一端即可
    else:
        elbow1, count1 = get_elbow_inst(points, alternative1)
        elbow2, count2 = get_elbow_inst(points, alternative2)
        if count1 > count2:
            elbow = elbow1
        else:
            elbow = elbow2
        return elbow

'''给出圆柱和拐弯，更新相应的一个anchor'''
def update_elbow(elbow,elbow_side,alternative,torus_index=None):
    if torus_index is not None: # 因为有的点云无法成功拟合拐弯，所以拐弯基元的参数并不连续
        elbow.tid=torus_index
        globals.save_elbows.append(elbow)
    cylinder_index,side=alternative
    if side==TOP:
        anchor_id=globals.load_cylinders[cylinder_index].top_id
    else:
        anchor_id=globals.load_cylinders[cylinder_index].bottom_id
    if elbow_side==P1:
        elbow.p1_id=anchor_id
        new_anchor=Anchor(len(globals.load_anchors),elbow.p_coord2,elbow.group_id)
        elbow.p2_id=len(globals.load_anchors)
        globals.load_anchors.append(new_anchor)

    else:
        elbow.p2_id=anchor_id

def find_anchor(specific_id):
    return [obj for obj in globals.load_anchors if obj.tid == specific_id][0]
def find_cylinder(specific_id):
    return [obj for obj in globals.load_cylinders if obj.tid == specific_id][0]
def find_elbow(specific_id):
    return [obj for obj in globals.save_elbows if obj.tid == specific_id][0]
def find_group(specific_id):
    return [obj for obj in globals.save_groups if obj.tid == specific_id][0]

'''在拐弯重建过程中，不会增加新的anchor,原有的anchor都是通过圆柱索引得到'''
def update_anchor(elbow,alternative):
    cylinder_index,side=alternative
    if side==TOP:
        anchor_id=globals.load_cylinders[cylinder_index].top_id
    else:
        anchor_id=globals.load_cylinders[cylinder_index].bottom_id
    anchor=find_anchor(anchor_id)
    anchor.add_part(("elbow",elbow.tid))

def update_group(elbow,alternative):
    '''
    在创建拐弯后更新group参数，在此之前拐弯肯定没有group_id
    从一端创建拐弯的情况
    '''
    cylinder_index,side=alternative
    cylinder=globals.load_cylinders[cylinder_index]
    if cylinder.group_id==-1:
        # 创建group并更新
        group=Group(globals.group_tid)
        globals.group_tid+=1
        group.add_part("elbow",elbow.tid)
        group.add_part("cylinder",cylinder.tid)

        if side==TOP:
            anchor_tid=cylinder.top_id
        else:
            anchor_tid=cylinder.bottom_id
        group.add_anchor(anchor_tid)
        # 更新基元
        cylinder.group_id=group.tid
        elbow.group_id=group.tid

        globals.save_groups.append(group)
    else: # 圆柱已经有所属的组，不再需更新圆柱
        group=find_group(cylinder.group_id)
        group.add_part("elbow",elbow.tid)
        if side==TOP:
            anchor_tid=cylinder.top_id
        else:
            anchor_tid=cylinder.bottom_id
        group.add_anchor(anchor_tid)
        elbow.group_id=group.tid

def del_group(group_inst):
    globals.save_groups=[x for x in globals.save_groups if x.tid != group_inst.tid]
def update_group_2(elbow,alternative1,alternative2):
    '''从两端创建拐弯，如果group_id不同，合并group'''
    cylinder_index1, side1 = alternative1
    cylinder_index2, side2 = alternative2
    cylinder1 = globals.load_cylinders[cylinder_index1]
    cylinder2 = globals.load_cylinders[cylinder_index2]
    if cylinder1.group_id == -1 and cylinder2.group_id == -1:
        # 创建group并更新
        group = Group(globals.group_tid)
        globals.group_tid += 1
        group.add_part("cylinder",cylinder1.tid)
        group.add_part("cylinder",cylinder2.tid)
        globals.save_groups.append(group)
    elif cylinder1.group_id!=-1 and cylinder2.group_id==-1:
        group = find_group(cylinder1.group_id)
        # 更新group
        group.add_part("cylinder",cylinder2.tid)
    elif cylinder1.group_id==-1 and cylinder2.group_id!=-1:
        group = find_group(cylinder2.group_id)
        # 更新group
        group.add_part("cylinder", cylinder1.tid)
    elif cylinder1.group_id!=-1 and cylinder2.group_id!=-1 and cylinder1.group_id==cylinder2.group_id:
        group=find_group(cylinder1.group_id)
    # 分属两个组的情况
    elif cylinder1.group_id != -1 and cylinder2.group_id != -1 and cylinder1.group_id != cylinder2.group_id:
        group=find_group(cylinder1.group_id)
        group2=find_group(cylinder2.group_id)
        group.parts+=group2.parts
        group.anchors+=group2.anchors
        for part in group2.parts:
            part_type, part_tid=part
            if part_type=="cylinder":
                inst=find_cylinder(part_tid)
            else:
                inst=find_elbow(part_tid)
            inst.group_id=group.tid
        for anchor_id in group2.anchors:
            anchor=find_anchor(anchor_id)
            anchor.group_id=group.tid
        '''因为存在删除，group_id不一定连续'''
        del_group(group2)

    group.add_part("elbow", elbow.tid)
    if side1 == TOP:
        anchor_tid = cylinder1.top_id
    else:
        anchor_tid = cylinder1.bottom_id
    group.add_anchor(anchor_tid)
    if side2 == TOP:
        anchor_tid = cylinder2.top_id
    else:
        anchor_tid = cylinder2.bottom_id
    group.add_anchor(anchor_tid)
    # 更新基元
    cylinder1.group_id = group.tid
    cylinder2.group_id = group.tid
    elbow.group_id = group.tid

def fit_side(torus_points,torus_index):
    anchor_center=np.mean(torus_points,0)
    # 按照距离排序得到最近的点
    neighbors_index = get_neighbors(globals.para_top_bottom, anchor_center)[:args.num_consider_cylinders]
    alternatives=[(neighbor_cylinder_index,neighbor_side) for \
            neighbor_cylinder_index,neighbor_side in neighbors_index if \
       np.min(np.linalg.norm(torus_points-globals.para_top_bottom[neighbor_cylinder_index][neighbor_side],axis=1)) \
     <3*globals.load_cylinders[neighbor_cylinder_index].radius and used_cylinder_sides[neighbor_cylinder_index][neighbor_side]==False]
    if len(alternatives)==1:
        used_cylinder_sides[alternatives[0][0]][alternatives[0][1]]=True
        elbow,_=get_elbow_inst(torus_points,alternatives[0])
        if elbow is not None:
            update_elbow(elbow,P1,alternatives[0],torus_index)
            update_anchor(elbow,alternatives[0])
            update_group(elbow,alternatives[0])
            return elbow
    elif len(alternatives)>=2:
        used_cylinder_sides[alternatives[0][0]][alternatives[0][1]] = True
        used_cylinder_sides[alternatives[1][0]][alternatives[1][1]]=True
        elbow = get_elbow_inst_from_sides(torus_points,alternatives[0],alternatives[1])
        if elbow is not None:
            update_elbow(elbow,P1,alternatives[0],torus_index)
            update_elbow(elbow,P2,alternatives[1])
            update_anchor(elbow,alternatives[0])
            update_anchor(elbow,alternatives[1])
            update_group_2(elbow,alternatives[0],alternatives[1])
            o3d.visualization.draw_geometries(
                [elbow.to_o3d_mesh(), globals.load_cylinders[alternatives[0][0]].to_o3d_mesh(), \
                 globals.load_cylinders[alternatives[1][0]].to_o3d_mesh()])

            return elbow

    return None


def main():
    mesh=o3d.geometry.TriangleMesh()
    # 从每个拐弯点云出发进行拟合
    for i in tqdm(range(len(torus_points))):
        res=fit_side(torus_points[i],i)
        if res is not None:
            mesh+=res.to_o3d_mesh()

    for i in globals.load_cylinders:
        mesh+=i.to_o3d_mesh()
    o3d.io.write_triangle_mesh("0802.obj", mesh)

    save_json("parameters_cy_elbow_anchor_group0805.json")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--num_consider_cylinders', help="neighbor cylinders taken into account", default=10)
    parser.add_argument('--p_coverage_thresh', type=float, default=0.1,help="consider 0.1 or 0.05")
    parser.add_argument('--threshold_radius', help="threshold of two cylinders in cast intersection", default=0)

    args = parser.parse_args()
    '''
    导入json说明：
    包含cylinder和anchor，cylinder已经与anchor建立联系
    '''
    globals.load_cylinders = Cylinder.load_cylinders_from_json('cylinders/parameters_cylinders_anchor.json')
    globals.load_anchors = Anchor.load_anchors_from_json('cylinders/parameters_cylinders_anchor.json')
    para_numpy=np.asarray([cylinder.numpy_get() for cylinder in globals.load_cylinders])
    globals.para_top_bottom = list((row1, row2) for row1, row2 in zip(para_numpy[:, :3], para_numpy[:, 3:6]))
    cylinder_nums=len(globals.load_cylinders)

    torus_points=get_elbow_points()
    used_cylinder_sides = [[False, False] for _ in range(cylinder_nums)]

    main()