import argparse
import math
import numpy as np
import open3d as o3d
from tqdm import tqdm
import torch
import argparse

from cylinder import Cylinder
from anchor import Anchor
from torus import Torus
from tee import Tee
from tools import *
import globals
from get_elbow import get_elbow
from pytorch3d.ops import knn_points, ball_query
from sklearn.cluster import DBSCAN
from group import Group


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


def get_cylinder_points():
    points_path = 'cylinders/cylinders_inst.npy'
    points_data = np.load(points_path)
    inst_nums = int(max(np.unique(points_data[:, 3])))
    cylinder_points = []
    for it in range(inst_nums):
        inst_point = points_data[points_data[:, 3] == it, :3]
        cylinder_points.append(inst_point)
        # Visual Test
        # print_pointcloud_with_open3d(inst_point)
    return cylinder_points


def load_inst_points_from_h2(points_path):
    pcd = o3d.t.io.read_point_cloud(points_path)

    points = np.asarray(pcd.point['positions'].numpy())
    colors = np.asarray(pcd.point['colors'].numpy())
    object_type = np.asarray(pcd.point['object_type'].numpy())
    object_label = np.asarray(pcd.point['object_label'].numpy())
    instance_type = np.asarray(pcd.point['instance_type'].numpy())
    instance_label = np.asarray(pcd.point['instance_label'].numpy())
    full_cloud = np.concatenate((points, colors, object_type, object_label, instance_type, instance_label), axis=1)

    pipe_cloud = full_cloud[full_cloud[:, 6] == 1]

    # 离群点的first try，使用DBSCAN扫除离群点
    fl_cloud = np.empty((0, 10))
    inst_label = np.unique(instance_label)
    for i in tqdm(inst_label):
        if i == -1:
            continue
        i_cloud = full_cloud[full_cloud[:, 9] == i]
        if i_cloud[0, 6] != 1:
            continue
        db = DBSCAN(eps=0.5, min_samples=10).fit(i_cloud[:, :3])
        labels = db.labels_
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(counts) == 0:
            continue
        largest_cluster_label = unique_labels[np.argmax(counts)]  # 最大簇的标签
        # 提取出最大的簇的点
        largest_cluster = i_cloud[labels == largest_cluster_label]
        fl_cloud = np.concatenate((fl_cloud, largest_cluster), axis=0)
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


def clean_inst_with_dbscan(pointcloud):
    db = DBSCAN(eps=0.3, min_samples=5).fit(pointcloud)
    labels = db.labels_
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(counts) == 0:
        return pointcloud
    largest_cluster_label = unique_labels[np.argmax(counts)]  # 最大簇的标签
    # 提取出最大的簇的点
    largest_cluster = pointcloud[labels == largest_cluster_label]
    return largest_cluster


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
    inst_names = ["cylinder", "elbow", "tees", "flange", "valve", 'instrument', 'support']
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
    pcd.point['groups'] = o3d.core.Tensor(inst_to_gid[pointcloud[:, 9].astype(np.int32)].reshape(-1, 1),
                                          dtype=o3d.core.Dtype.Int32)
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


if __name__ == "__main__":
    # Using Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--KNN_neighbor_num", default=3, help="Number of KNN in finding nearest element")
    parser.add_argument("-r", "--radius", default=0.05, help="Radius of Bounding Sphere, consider 5cm(0.05)/10cm(0.1)")
    args = parser.parse_args()
    pointcloud_path = 'data/H2_grid_2_labels_remake_20240926.pcd'
    result_path = "data/H2_group_first_try_0925.pcd"

    # Using jsons
    # json用cylinder/torus对应的parameter
    globals.load_cylinders = Cylinder.load_cylinders_from_json('cylinders/parameters_cylinders_anchor.json')
    globals.load_anchors = Anchor.load_anchors_from_json('cylinders/parameters_cylinders_anchor.json')

    cylinder_ids = [cy.tid for cy in globals.load_cylinders]
    # cylinders_pointcloud = get_cylinder_points()
    full_cloud, all_pipe_pointcloud = load_inst_points_from_h2(pointcloud_path)

    # all the others
    group = Group(globals.group_tid)
    globals.group_tid += 1
    globals.save_groups.append(group)

    # Finding Farthest Points in all cylinders
    cylinders_pointcloud, other_pointcloud = find_cloud_with_and_without_type(all_pipe_pointcloud, 10)

    for i in tqdm(range(len(cylinders_pointcloud))):
        cy_points = cylinders_pointcloud[i]
        cy_inst = cy_points[0, 4]
        inst_list = get_group_inst(cy_points, other_pointcloud, args.radius)
        inst_list.append([10, cy_inst])
        inst_list = np.unique(inst_list, axis=0)
        # print(inst_list)
        grouping(inst_list)

    # Finding Farthest Points in all torus
    torus_pointcloud, other_pointcloud = find_cloud_with_and_without_type(all_pipe_pointcloud, 11)

    for i in tqdm(range(len(torus_pointcloud))):
        tr_points = torus_pointcloud[i]
        tr_inst = tr_points[0, 4]
        inst_list = get_group_inst(tr_points, other_pointcloud, args.radius)
        inst_list.append([11, tr_inst])
        inst_list = np.unique(inst_list, axis=0)
        # print(inst_list)
        grouping(inst_list)

    # Finding Farthest Points in all tees
    tees_pointcloud, other_pointcloud = find_cloud_with_and_without_type(all_pipe_pointcloud, 12)
    for i in tqdm(range(len(tees_pointcloud))):
        ts_points = tees_pointcloud[i]
        ts_inst = ts_points[0, 4]
        ts_points = clean_inst_with_dbscan(ts_points[:, :3])
        fps_samples_idx = farthest_points(ts_points)

        dist_to_p1 = np.linalg.norm(ts_points - ts_points[fps_samples_idx[0]], axis=1)
        dist_to_p2 = np.linalg.norm(ts_points - ts_points[fps_samples_idx[1]], axis=1)
        third_point_idx = np.argmax(dist_to_p1 + dist_to_p2)
        fps_samples_idx.append(third_point_idx)
        all_points = ts_points
        ind = find_points_within_radius(ts_points[fps_samples_idx], other_pointcloud, args.radius)
        inst_list = []
        for p in ind:
            if other_pointcloud[p, 9] != -1:
                inst_list.append(other_pointcloud[p, 8:10])
        inst_list.append([12, ts_inst])
        inst_list = np.unique(inst_list, axis=0)
        # print(inst_list)
        grouping(inst_list)

    # 按照组打印对应的点云，以查看打组结果
    '''
    for g in globals.save_groups:
        print(f"Group {g.tid}: {g.get_parts()}")
        pointcloud = np.empty((0, 3))
        for inst in g.get_parts():
            inst_points = all_pipe_pointcloud[(all_pipe_pointcloud[:, 9] == inst[1]), :3]
            # print_pointcloud_with_open3D(inst_points)
            pointcloud = np.vstack((pointcloud, inst_points))
        print_pointcloud_with_open3d(pointcloud)
    '''
    # 将点云打组结果储存在pcd内
    save_group_result(full_cloud, np.max(all_pipe_pointcloud[:, 9], axis=0), result_path)
