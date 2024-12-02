import argparse
from distutils.command.clean import clean

import numpy as np
import open3d as o3d
from tqdm import tqdm
import torch
from src.group import globals

from geometry.cylinder import Cylinder
from geometry.anchor import Anchor
from geometry.torus import Torus
from geometry.tee import Tee


from pytorch3d.ops import ball_query
from sklearn.cluster import DBSCAN
from src.group.group import Group
from src.group.json_group_testing import get_torus_elbow
from src.group.pointcloud_factory import FactoryPointcloud

from group_direction import PipeGroupGraph, PipeGroupConstructor


def find_group(specific_id):
    return [obj for obj in globals.save_groups if obj.tid == specific_id][0]


def del_group(group_inst):
    globals.save_groups = [x for x in globals.save_groups if x.tid != group_inst.tid]
    for ids in range(len(globals.save_groups)):
        globals.save_groups[ids].tid = ids
    globals.group_tid = len(globals.save_groups)


def print_pointcloud_with_open3d(pointcloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    o3d.visualization.draw_geometries([pcd])


def load_inst_points_from_h2(singal, pc_path, la_path):
    pcd = FactoryPointcloud(pc_path[-3:], pc_path, la_path)

    full_cloud = pcd.get_full_cloud()
    pipe_cloud = full_cloud[full_cloud[:, 6] == 1]
    instance_label = np.unique(pipe_cloud[:, 9])

    # 离群点的first try，使用DBSCAN扫除离群点
    fl_cloud = np.empty((0, 10))
    inst_label = np.unique(instance_label)
    for idx, i in tqdm(enumerate(inst_label)):
        singal.emit(int(idx / len(inst_label) * 100))
        if i == -1:
            continue
        i_cloud = pipe_cloud[pipe_cloud[:, 9] == i]
        if i_cloud[0, 6] != 1:
            continue
        db = DBSCAN(eps=0.5, min_samples=10).fit(i_cloud[:, :3])
        labels = db.labels_
        largest_cluster = i_cloud[labels != -1]
        fl_cloud = np.concatenate((fl_cloud, largest_cluster), axis=0)
    singal.emit(0)
    pipe_cloud = fl_cloud[fl_cloud[:, 6] == 1]
    return full_cloud, pipe_cloud


def find_cloud_with_and_without_type(pointcloud, inst_type):
    not_type_pointcloud = pointcloud[pointcloud[:, 8] != inst_type]
    type_pointcloud = []
    type_num = np.max(pointcloud[:, 9], axis=0).astype(np.int32)
    for t in range(type_num + 1):
        t_cloud = pointcloud[pointcloud[:, 9] == t]
        if t_cloud.shape[0] == 0:
            continue
        if t_cloud[0, 8] == inst_type:
            type_pointcloud.append(np.concatenate((t_cloud[:, :3], t_cloud[:, 8:10]), axis=1))

    return type_pointcloud, not_type_pointcloud


def find_inst_in_group(tid):
    for gid in range(globals.group_tid):
        _group = find_group(gid)
        _group_num = _group.get_parts()
        if any(tid == part[1] for part in _group_num):
            return gid
    return -1


def farthest_points(pointcloud):
    points = np.array(pointcloud, dtype=np.float32)
    # 计算点对之间的距离的平方矩阵
    distances = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=-1)
    # 将对角线置为0，以防止将同一对点识别为最远点
    np.fill_diagonal(distances, 0)
    # 找到最远的两个点的索引
    point1, point2 = np.unravel_index(np.argmax(distances), distances.shape)
    return [point1, point2]


def find_points_within_radius(query_points, pointcloud, rad):
    qp_tensor = torch.from_numpy(np.array(query_points).astype(np.float32)).unsqueeze(0)
    pcd_tensor = torch.tensor(np.array(pointcloud[:, :3]).astype(np.float32)).unsqueeze(0)

    distances, indices, _ = ball_query(qp_tensor, pcd_tensor, radius=rad)

    indices = torch.squeeze(indices)
    indices = indices[indices != -1]
    return indices


def grouping(insts):
    # 打组，分配组号
    group_id = []
    for inst in insts:
        group_id.append(find_inst_in_group(inst[1]))
    uni_group_id = np.unique(group_id, axis=0)
    inst_names = [
        "cylinder",
        "elbow",
        "tees",
        "flange",
        "valve",
        'instrument',
        'support',
    ]
    if len(uni_group_id) == 1 and uni_group_id[0] == -1:
        # 若所有部分都不存在对应组，创建一个新的组，将整个部分放入组内
        group = Group(globals.group_tid)
        globals.group_tid += 1
        for inst in insts:
            group.add_part(inst_names[inst[0].astype(np.int32) - 10], inst[1])
        globals.save_groups.append(group)
    elif len(uni_group_id) <= 1:
        # 若所有部分都在一个组内，跳过处理即可
        return
    elif len(uni_group_id) == 2 and uni_group_id[0] == -1:
        # 若打组的部分中仅有单个成员在某个组内，则把整个部分放入对应组内
        group = find_group(uni_group_id[1])
        for _ in range(len(group_id)):
            if group_id[_] == -1:
                inst = insts[_]
                group.add_part(inst_names[inst[0].astype(np.int32) - 10], inst[1])
    else:
        # 若部分属于不同的组，则合并对应的组
        # 直接默认合并至编号最小的一个组
        # -1直接合并进最终一组
        starter = 0
        if uni_group_id[0] == -1:
            starter = 1
        group = find_group(uni_group_id[starter])
        counter = 0
        for _ in range(starter + 1, len(uni_group_id)):
            remove_group = find_group(uni_group_id[_] - counter)
            for inst in remove_group.get_parts():
                group.add_part(inst[0], inst[1])
            del_group(remove_group)
            counter += 1
        if starter == 1:
            for _ in range(len(group_id)):
                if group_id[_] == -1:
                    inst = insts[_]
                    group.add_part(inst_names[inst[0].astype(np.int32) - 10], inst[1])


def save_group_result(pointcloud, inst, output_path):
    # 初步打组结果单独导出为单个pcd
    inst_to_gid = np.zeros(inst.astype(np.int32) + 2)
    for g in globals.save_groups:
        for part in g.get_parts():
            inst_to_gid[part[1].astype(np.int32)] = g.tid

    pcd = o3d.t.geometry.PointCloud()
    pcd.point['positions'] = o3d.core.Tensor(pointcloud[:, :3], dtype=o3d.core.Dtype.Float32)
    pcd.point['colors'] = o3d.core.Tensor(pointcloud[:, 3:6], dtype=o3d.core.Dtype.Float32)
    pcd.point['inst_type'] = o3d.core.Tensor(pointcloud[:, 9].reshape(-1, 1), dtype=o3d.core.Dtype.Int32)
    pcd.point['groups'] = o3d.core.Tensor(
        inst_to_gid[pointcloud[:, 9].astype(np.int32)].reshape(-1, 1),
        dtype=o3d.core.Dtype.Int32,
    )
    o3d.t.io.write_point_cloud(output_path, pcd)


def get_group_inst(points, other_points, radius):
    # 得到点云实例附近的实例
    # points = clean_inst_with_dbscan(points[:, :3])
    points = points[:, :3]
    fps_samples_idx = farthest_points(points)
    ind = find_points_within_radius(points[fps_samples_idx], other_points, radius)
    _inst_list = []
    for p in ind:
        if other_points[p, 9] != -1:
            _inst_list.append(other_points[p, 8:10])
    return _inst_list


def find_instance_with_json_anchors(anchor_point, pointcloud):
    min_dist = 1000.0
    min_inst = 0
    for _ in range(len(pointcloud)):
        points = pointcloud[_]
        inst = points[0, 4]
        dist = np.linalg.norm(points[:, :3] - anchor_point[0], axis=1)
        if min_dist > np.mean(dist):
            min_dist = np.mean(dist)
            min_inst = inst
    return min_inst


def create_inst_to_tid(all_points, parameters_cy_elbow_tee_anchor_group_pth):
    # 得到标注编号到json tid的映射
    anchors = Anchor.load_anchors_from_json(parameters_cy_elbow_tee_anchor_group_pth)
    cylinders = Cylinder.load_cylinders_from_json(parameters_cy_elbow_tee_anchor_group_pth)
    elbows = Torus.load_elbows_from_json(parameters_cy_elbow_tee_anchor_group_pth)
    tees = Tee.load_tees_from_json(parameters_cy_elbow_tee_anchor_group_pth)

    elbow_tee_anchor_list = get_torus_elbow(elbows, tees, anchors)

    cy_pointcloud, _ = find_cloud_with_and_without_type(all_points, 10)
    torus_pointcloud, _ = find_cloud_with_and_without_type(all_points, 11)
    tees_pointcloud, _ = find_cloud_with_and_without_type(all_points, 12)

    # 建立从标注序号到json tid的映射表（后续应先做统一）
    inst_to_tid = {}
    for cylinder in cylinders:
        points = np.asarray([anchors[cylinder.top_id].get_coord(), anchors[cylinder.bottom_id].get_coord()])
        inst_num = find_instance_with_json_anchors(points, cy_pointcloud)
        inst_to_tid[inst_num] = cylinder.tid
    for elbow in elbows:
        points = np.asarray([anchors[elbow.p1_id].get_coord(), anchors[elbow.p2_id].get_coord()])
        inst_num = find_instance_with_json_anchors(points, torus_pointcloud)
        inst_to_tid[inst_num] = int(elbow.tid)
    for tee in tees:
        points = np.asarray(
            [
                anchors[tee.top1_id].get_coord(),
                anchors[tee.bottom1_id].get_coord(),
                anchors[tee.top2_id].get_coord(),
            ]
        )
        inst_num = find_instance_with_json_anchors(points, tees_pointcloud)
        inst_to_tid[inst_num] = int(tee.tid)
    return inst_to_tid


def group_json_test(all_points, rad, parameters_cy_elbow_tee_anchor_group_pth):
    anchors = Anchor.load_anchors_from_json(parameters_cy_elbow_tee_anchor_group_pth)
    elbows = Torus.load_elbows_from_json(parameters_cy_elbow_tee_anchor_group_pth)
    tees = Tee.load_tees_from_json(parameters_cy_elbow_tee_anchor_group_pth)

    elbow_tee_anchor_list = get_torus_elbow(elbows, tees, anchors)

    torus_pointcloud, _ = find_cloud_with_and_without_type(all_points, 11)
    tees_pointcloud, _ = find_cloud_with_and_without_type(all_points, 12)

    for elbow in elbows:
        # 先找到这个基元在点云中对应的实例，直接找距离关键点最近的基元
        points = np.asarray([anchors[elbow.p1_id].get_coord(), anchors[elbow.p2_id].get_coord()])
        inst_num = find_instance_with_json_anchors(points, torus_pointcloud)
        # 之后找关键点中距离最近的点进行打组
        # 目前的问题是关键点之间的距离偏大，没法打出比较好的组
        anchors_list = find_points_within_radius(points, elbow_tee_anchor_list, rad)
        anchors_list = np.unique(anchors_list)
        if len(anchors_list) > 2:
            inst_list = []
            for anchor in anchors_list:
                if np.any(np.all(points == elbow_tee_anchor_list[anchor, :3], axis=1)):
                    continue
                if elbow_tee_anchor_list[anchor, 4] == -1:
                    continue
                if elbow_tee_anchor_list[anchor, 3] == 1:
                    target_inst = find_instance_with_json_anchors(elbow_tee_anchor_list[anchor, :3], torus_pointcloud)
                    inst_list.append([11, target_inst.astype(np.int32)])
                if elbow_tee_anchor_list[anchor, 3] == 2:
                    target_inst = find_instance_with_json_anchors(elbow_tee_anchor_list[anchor, :3], tees_pointcloud)
                    inst_list.append([12, target_inst.astype(np.int32)])
            inst_list.append([11, inst_num.astype(np.int32)])
            inst_list = np.unique(inst_list, axis=0)
            if len(inst_list) > 1:
                grouping(inst_list)

    for tee in tees:
        points = np.asarray(
            [
                anchors[tee.top1_id].get_coord(),
                anchors[tee.bottom1_id].get_coord(),
                anchors[tee.top2_id].get_coord(),
            ]
        )
        inst_num = find_instance_with_json_anchors(points, tees_pointcloud)
        anchors_list = find_points_within_radius(points, elbow_tee_anchor_list, rad)
        anchors_list = np.unique(anchors_list)
        if len(anchors_list) > 3:
            # 找关键点对应基元在点云中的编号，然后将其打组
            inst_list = []
            for anchor in anchors_list:
                if np.any(np.all(points == elbow_tee_anchor_list[anchor, :3], axis=1)):
                    continue
                if elbow_tee_anchor_list[anchor, 4] == -1:
                    continue
                if elbow_tee_anchor_list[anchor, 3] == 1:
                    target_inst = find_instance_with_json_anchors(elbow_tee_anchor_list[anchor, :3], torus_pointcloud)
                    inst_list.append([11, target_inst.astype(np.int32)])
                if elbow_tee_anchor_list[anchor, 3] == 2:
                    target_inst = find_instance_with_json_anchors(elbow_tee_anchor_list[anchor, :3], tees_pointcloud)
                    inst_list.append([12, target_inst.astype(np.int32)])
            inst_list.append([12, inst_num.astype(np.int32)])
            inst_list = np.unique(inst_list, axis=0)
            if len(inst_list) > 1:
                grouping(inst_list)


def load_points(
    pc_path = '',
    la_path = '',
):
    # 适应新版软件的编号，每种实例有自己连续的编号值
    # 因此可以根据点云实例给定一个新的全局编号
    pcd = FactoryPointcloud(pc_path[-3:], "src/group/inst_mapping.txt", pc_path, la_path)

    full_cloud = pcd.get_full_cloud()
    downsample_step = int(100 / (100 - group_downsample_rate))

    full_cloud = full_cloud[::downsample_step, :]
    pipe_cloud = np.empty((0, 10))
    type_list = []

    for t in range(10, 16):
        type_cloud = full_cloud[full_cloud[:, 8] == t]
        instance_label = np.unique(type_cloud[:, 9])
        type_list.append(len(type_cloud))
        for idx, i in tqdm(enumerate(instance_label)):
            if i == -1:
                continue
            ith_cloud = type_cloud[type_cloud[:, 9] == i]
            clean_res = DBSCAN(eps=0.5, min_samples=10).fit(ith_cloud)
            labels = clean_res.labels_
            pipe_cloud = np.concatenate((pipe_cloud, ith_cloud[labels != -1]), axis=0)

    return pcd, full_cloud, pipe_cloud, type_list






def get_group_list(
    radius=0.05,
    pointcloud_path='',
    label_path='',
    parameters_cylinders_pth='',
    parameters_cy_elbow_tee_anchor_group_pth='',
    result_path='',
):

    # data initialization
    # pointcloud
    pcd, full_cloud, all_pipe_pointcloud, type_list = load_points(pointcloud_path, label_path)
    # json
    globals.load_cylinders = Cylinder.load_cylinders_from_json(parameters_cylinders_pth)
    globals.load_anchors = Anchor.load_anchors_from_json(parameters_cylinders_pth)
    # 完全按照点云进行打组
    pointcloud_grouping_constructor = PipeGroupConstructor(parameters_cylinders_pth, pcd)
    for t in range(10, 12):
        type_points = all_pipe_pointcloud[all_pipe_pointcloud[:, 8] == t]
        for i in tqdm(range(type_list[t - 10])):
            cur_points = type_points[type_points[:, 9] == i]
            other_points = all_pipe_pointcloud[(all_pipe_pointcloud[:, 8] != t) | (all_pipe_pointcloud[:, 9] != i)]
            farthest_sample = farthest_points(cur_points[:, 3])
            ind = find_points_within_radius(cur_points[farthest_sample], other_points, radius)
            for p in ind:
                inst_type, inst_num = other_points[p, 8:10]
                pointcloud_grouping_constructor.grouping_by_pcd(i, t, inst_num, inst_type)




    # 根据拟合得到的json进行打组
