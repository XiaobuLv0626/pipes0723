import open3d as o3d
import argparse
import globals
import numpy as np
from tqdm import tqdm

from cylinder import Cylinder
from group import Group
from torus import Torus
from anchor import Anchor
from tools import *
from tee import Tee
import torch

from pytorch3d.ops import knn_points


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


# 预处理三通点云
def get_tees_points():
    points_path = "tees/tees_inst.npy"
    points_data = np.load(points_path)
    a = np.unique(points_data[:, 6])
    max_num = int(max(a))
    tees_points = []
    for x in range(max_num):
        tees_points.append(points_data[points_data[:, 6] == x][:, :3])
    return tees_points


def exist_intersect(coord1, dir1, coord2, thresh_radius):
    '''用于检测是否平行'''
    for i in np.arange(0.01, 1.85, 0.005):
        extent1 = coord1 + dir1 * i
        if np.linalg.norm(extent1 - coord2) < thresh_radius:
            return True
    return False


def get_extend_center(points, cylinder):
    vertices = np.asarray(cylinder.to_o3d_mesh().sample_points_uniformly(500).points)
    idx = find_nearest_neighbors(points, vertices)
    idx = idx.cpu().numpy()
    dis = (np.linalg.norm(points - vertices[idx], axis=1))
    un_cy_points = dis > args.p_coverage_thresh
    need_points = points[un_cy_points]
    if len(need_points) == 0:
        return None
    else:
        return np.mean(need_points, axis=0)


def get_tee_inst2(points, alternative1, alternative2):
    '''
    如果两端平行（但只有圆柱才能得到方向向量，所以从圆柱延长判断平行），直接处理
    其他的情况，不进行拟合
    '''
    index1, side1 = alternative1
    index2, side2 = alternative2
    flag = False
    coord1 = globals.para_top_bottom[index1][side1]
    coord2 = globals.para_top_bottom[index2][side2]

    if isinstance(globals.load_primitives[index1], Cylinder):
        cylinder1 = globals.load_primitives[index1]
        radius1 = cylinder1.radius
        dir = get_correct_dir(cylinder1, side1)
        if exist_intersect(coord1, dir, coord2, radius1):  # 平行
            flag = True
    elif isinstance(globals.load_primitives[index2], Cylinder):
        cylinder2 = globals.load_primitives[index2]
        radius1 = cylinder2.radius
        dir = get_correct_dir(cylinder2, side2)
        if exist_intersect(coord2, dir, coord1, radius1):
            flag = True
    if flag:
        # 回归coord3和coord4的位置
        line1 = coord2 - coord1
        if np.linalg.norm(line1) < 0.01:
            return None
        points_extend_center = get_extend_center(points, Cylinder(coord1, coord2, radius1))
        if points_extend_center is None:
            points_extend_center = np.mean(points, axis=0)
        best_count = 0
        final_tee = None
        for step1 in np.linspace(0.2, 0.8, 20):
            extend_coord3 = coord1 + line1 * step1
            # 默认这里的三通是垂直的
            infer = points_extend_center - coord1
            part = np.dot(infer, line1) / np.linalg.norm(line1)
            part /= np.linalg.norm(line1)
            line2 = infer - line1 * part  # line1和line2垂直
            line2 = line2 / np.linalg.norm(line2)
            for step2 in np.linspace(0.02, 0.1, 10):
                extend_coord4 = extend_coord3 + line2 * step2
                for radius2 in np.linspace(radius1 / 2, radius1, 5):
                    tee = Tee(coord1, coord2, radius1, extend_coord3, extend_coord4, radius2)
                    vertices = np.asarray(tee.to_o3d_mesh().sample_points_uniformly(500).points)
                    idx = find_nearest_neighbors(points, vertices)
                    idx = idx.cpu().numpy()
                    dis = (np.linalg.norm(points - vertices[idx], axis=1))
                    count = np.sum(dis < args.p_coverage_thresh)
                    if count > best_count:
                        best_count = count
                        final_tee = tee
        return final_tee
    else:
        return None

    '''如果没有交点，在主函数中已经判断'''


def regress_tee_with_coord4(points, top1, bottom1, radius1, coord3, radius2):
    level_dir = (bottom1 - top1)
    if np.linalg.norm(level_dir) < 0.05:
        return None
    best_count = 0
    final_tee = None
    for t in np.arange(0, 1, 0.05):
        extend_coord4 = top1 + level_dir * t
        tee = Tee(top1, bottom1, radius1, coord3, extend_coord4, radius2)
        vertices = np.asarray(tee.to_o3d_mesh().sample_points_uniformly(500).points)
        idx = find_nearest_neighbors(points, vertices)
        idx = idx.cpu().numpy()
        dis = (np.linalg.norm(points - vertices[idx], axis=1))
        count = np.sum(dis < args.p_coverage_thresh)
        if count > best_count:
            best_count = count
            final_tee = tee
    return final_tee


def get_tee_inst3(points, alternatives):
    indexes, sides = zip(*alternatives)
    indexes = np.asarray(indexes)
    sides = np.asarray(sides)
    radiuses = [globals.load_primitives[indice].radius for indice in indexes]
    flag = None
    # 如果三个圆柱都平行，跳过
    if all([isinstance(globals.load_primitives[indice], Cylinder) for indice in indexes]):
        cy = [globals.load_primitives[index] for index in indexes]
        if (np.abs(np.dot(get_correct_dir(cy[0], sides[0]), get_correct_dir(cy[1], sides[1]))) > 0.8) and \
                (np.abs(np.dot(get_correct_dir(cy[0], sides[0]), get_correct_dir(cy[2], sides[2])))) > 0.8 and \
                np.abs(np.dot(get_correct_dir(cy[1], sides[1]), get_correct_dir(cy[2], sides[2]))) > 0.8:
            return None,0,0,0

    coords = [globals.para_top_bottom[x][y] for x, y in alternatives]
    # 因为部件存在拐弯，所以不能直接通过方向向量平行来计算
    for ii, indice in enumerate(indexes):
        if isinstance(globals.load_primitives[indice], Cylinder):
            cylinder = globals.load_primitives[indice]
            radius = cylinder.radius
            side = sides[ii]
            dir = get_correct_dir(cylinder, side)
            coord1 = coords[ii]
            for j, _ in enumerate(indexes):
                if j == ii:
                    continue
                coord2 = coords[j]
                if exist_intersect(coord1, dir, coord2, 10 * args.p_coverage_thresh):  # 平行
                    if abs(radiuses[ii] - radiuses[j]) < min(radiuses[ii], radiuses[j]):  # 半径差距在2倍以内
                        flag = (ii, j)
    if flag is not None:
        map = {1: 2, 3: 0, 2: 1}
        coord1 = coords[flag[0]]
        coord2 = coords[flag[1]]
        another_serial = map[flag[0] + flag[1]]
        coord3 = coords[another_serial]
        # 返回的后三个alternatives方便判断连接顺序
        return regress_tee_with_coord4(points, coord1, coord2, radiuses[flag[0]],\
                                       coord3, radiuses[another_serial]),alternatives[flag[0]],alternatives[flag[1]],alternatives[another_serial]
    else:
        return None,0,0,0


'''如果没有交点，在主函数中已经判断'''


def cast_intersection(top1, bottom1, coord, dir, cy1_radius, cy2_radius):
    level_dir = (bottom1 - top1)
    intersection_pos = []
    length = []
    res = 1e9
    for t in np.arange(0, 1, 0.05):
        level_extend = top1 + level_dir * t
        for k in np.arange(0, 0.5, 0.01):
            vertical_extend = coord + dir * k
            if res > np.linalg.norm(level_extend - vertical_extend):
                res = min(res, np.linalg.norm(level_extend - vertical_extend))
                intersection_pos = vertical_extend
                length = k
    if res < 2 * cy1_radius:
        return Tee(top1, bottom1, cy1_radius, intersection_pos, coord, cy2_radius)  # 记住这里的顺序，level圆柱的两端和Tee的第一个圆柱对应
    else:
        return None


def get_tee_slope(points, alternative):
    cylinder_idx, side = alternative
    cylinder = globals.load_primitives[cylinder_idx]
    if isinstance(cylinder, Torus):  # 如果后续有需要，在torus中添加direction方法实现统一
        return None,None
    points_center = np.mean(points, 0)
    neighbors_index = get_neighbors(globals.para_top_bottom, points_center)[:args.num_consider_cylinders]
    for neighbor_cylinder_index, neighbor_side in neighbors_index:
        neighbor_cylinder = globals.load_primitives[neighbor_cylinder_index]
        if isinstance(neighbor_cylinder, Torus):  # 只有圆柱才能斜插
            continue
        args.threshold_radius = (globals.load_primitives[cylinder_idx].radius + \
                                 neighbor_cylinder.radius) / 1.5
        if np.abs(cylinder.radius - neighbor_cylinder.radius) > args.threshold_radius:  # 如果半径相差5倍关系，则跳过
            continue

        # 检测方向条件是否满足
        residual = np.abs(np.dot(cylinder.get_direction(), get_correct_dir(neighbor_cylinder, neighbor_side)))
        if residual < args.threshold_parallel:  # 角度至少要大于30度
            # 计算圆柱延长线的交点
            coord = globals.para_top_bottom[cylinder_idx][side]
            dir = get_correct_dir(cylinder, side)
            tee = cast_intersection(neighbor_cylinder.top_center, neighbor_cylinder.bottom_center, coord, dir,
                                    neighbor_cylinder.radius, cylinder.radius)
            if tee is not None:
                return tee, neighbor_cylinder_index
    return None,None


def del_group(group_inst):
    globals.save_groups = [x for x in globals.save_groups if x.tid != group_inst.tid]


def find_anchor(specific_id):
    return [obj for obj in globals.load_anchors if obj.tid == specific_id][0]


def find_cylinder(specific_id):
    return [obj for obj in globals.load_cylinders if obj.tid == specific_id][0]


def find_elbow(specific_id):
    return [obj for obj in globals.save_elbows if obj.tid == specific_id][0]


def find_group(specific_id):
    return [obj for obj in globals.save_groups if obj.tid == specific_id][0]


def find_tee(specific_id):
    return [obj for obj in globals.save_tees if obj.tid == specific_id][0]


def update_tee(tee, tee_side, alternative, tee_index=None):
    if tee_index is not None:
        tee.tid = tee_index
        globals.save_tees.append(tee)
    part_index, side = alternative
    part = globals.load_primitives[part_index]
    if side == P1:
        anchor_id = part.pr1_id
    else:
        anchor_id = part.pr2_id

    if tee_side == P1:
        tee.top1_id = anchor_id
    elif tee_side == P2:
        tee.bottom1_id = anchor_id
    else:
        tee.top2_id = anchor_id


def update_anchor(tee, alternative):
    part_index, side = alternative
    if side == P1:
        anchor_id = globals.load_primitives[part_index].pr1_id
    else:
        anchor_id = globals.load_primitives[part_index].pr2_id
    anchor = find_anchor(anchor_id)
    anchor.add_part(("Tee", tee.tid))


def update_group_2(tee, alternative1, alternative2):
    '''从两个部件创建三通，如果group_id不同，合并group'''
    index1, side1 = alternative1
    index2, side2 = alternative2
    part1 = globals.load_primitives[index1]
    part1_type = "cylinder" if isinstance(part1, Cylinder) else "elbow"
    part2 = globals.load_primitives[index2]
    part2_type = "cylinder" if isinstance(part2, Cylinder) else "elbow"
    if part1.group_id == -1 and part2.group_id == -1:
        # 创建group并更新
        group = Group(globals.group_tid)
        globals.group_tid += 1
        if (part1_type, part1.tid) not in group.parts:
            group.add_part(part1_type, part1.tid)
        if (part2_type, part2.tid) not in group.parts:
            group.add_part(part2_type, part2.tid)
        globals.save_groups.append(group)
    elif part1.group_id != -1 and part2.group_id == -1:
        group = find_group(part1.group_id)
        if (part2_type, part2.tid) not in group.parts:
            group.add_part(part2_type, part2.tid)
    elif part1.group_id == -1 and part2.group_id != -1:
        group = find_group(part2.group_id)
        if (part1_type, part1.tid) not in group.parts:
            group.add_part(part1_type, part1.tid)
    elif part1.group_id != -1 and part2.group_id != -1 and part1.group_id == part2.group_id:
        group = find_group(part1.group_id)
    # 分属两个组的情况
    elif part1.group_id != -1 and part2.group_id != -1 and part1.group_id != part2.group_id:
        group = find_group(part1.group_id)
        group2 = find_group(part2.group_id)
        group.parts += group2.parts
        group.anchors += group2.anchors
        for part in group2.parts:
            part_type, part_tid = part
            if part_type == "cylinder":
                inst = find_cylinder(part_tid)
            elif part_type == "elbow":
                inst = find_elbow(part_tid)
            else:
                inst = find_tee(part_tid)
            inst.group_id = group.tid
        for anchor_id in group2.anchors:
            anchor = find_anchor(anchor_id)
            anchor.group_id = group.tid
        '''因为存在删除，group_id不一定连续'''
        del_group(group2)

    if ("tee", tee.tid) not in group.parts:
        group.add_part("tee", tee.tid)
    if side1 == TOP:
        anchor_tid = part1.pr1_id
    else:
        anchor_tid = part1.pr2_id
    if anchor_tid not in group.anchors:
        group.add_anchor(anchor_tid)

    if side2 == TOP:
        anchor_tid = part2.pr1_id
    else:
        anchor_tid = part2.pr2_id
    if anchor_tid not in group.anchors:
        group.add_anchor(anchor_tid)
    # 更新基元
    part1.group_id = group.tid
    part2.group_id = group.tid
    tee.group_id = group.tid



def fit_tees(tee_points, tee_index):
    points_center = np.mean(tee_points, 0)
    neighbors_index = get_neighbors(globals.para_top_bottom, points_center)[:args.num_consider_cylinders]
    seen = set()
    alternatives = [(neighbor_index, neighbor_side) for \
                    neighbor_index, neighbor_side in neighbors_index if \
                    np.min(np.linalg.norm(tee_points - globals.para_top_bottom[neighbor_index][neighbor_side], axis=1)) \
                    < 2 * globals.load_primitives[neighbor_index].p_radius and used_primitive_sides[neighbor_index][
                        neighbor_side] == False \
                    and not (neighbor_index in seen or seen.add(neighbor_index))]
    if len(alternatives) == 1:
        '''插入型三通'''
        my_tee, level_cylinder = get_tee_slope(tee_points, alternatives[0])
        if my_tee is not None:
            # 打组
            update_tee(my_tee, P1, (level_cylinder, TOP), tee_index)
            update_tee(my_tee, P2, (level_cylinder, BOTTOM))
            update_tee(my_tee, P3, alternatives[0])
            update_anchor(my_tee, (level_cylinder, TOP))
            update_anchor(my_tee, (level_cylinder, BOTTOM))
            update_anchor(my_tee, alternatives[0])
            update_group_2(my_tee, (level_cylinder, TOP), (level_cylinder, BOTTOM))
            update_group_2(my_tee, alternatives[0], (level_cylinder, BOTTOM))
            return my_tee
        else:
            return None
    if len(alternatives) == 2:  # 两端直接拼接，回归垂直部分
        my_tee = get_tee_inst2(tee_points, alternatives[0], alternatives[1])
        if my_tee is not None:
            update_tee(my_tee,P1,alternatives[0],tee_index)
            update_tee(my_tee,P2,alternatives[1])
            update_anchor(my_tee,alternatives[0])
            update_anchor(my_tee,alternatives[1])
            update_group_2(my_tee,alternatives[0],alternatives[1])
            return my_tee
        else:
            return None
    if len(alternatives) == 3:
        al=[0,0,0]
        my_tee,al[0],al[1],al[2] = get_tee_inst3(tee_points, alternatives[:3])
        temp=tee_index
        if my_tee is not None:
            for i in range(3):
                update_tee(my_tee,i,al[i],temp)
                temp=None
                update_anchor(my_tee,al[i])
            update_group_2(my_tee,al[0],al[1])
            update_group_2(my_tee,al[0],al[2])
            return my_tee
        else:
            # alternatives = [(neighbor_index, neighbor_side) for \
            #                 neighbor_index, neighbor_side in neighbors_index][:5]
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(tee_points)
            # x = [globals.load_primitives[neighbor_index].to_o3d_mesh() for neighbor_index, neighbor_side in
            #      alternatives]
            # x.append(pcd)
            # o3d.visualization.draw_geometries(x)
            return None
    # 几乎不存在四通样例，先不处理


def main():
    for i in tqdm(range(len(tees_points))):
        fit_tees(tees_points[i], i)

    mesh=o3d.geometry.TriangleMesh()
    for t1 in globals.save_tees:
        mesh+=t1.to_o3d_mesh()
    for t2 in globals.load_primitives:
        mesh+=t2.to_o3d_mesh()
    o3d.io.write_triangle_mesh("mesh_of_cy_elbow_tee_0805.obj",mesh)

    save_json("parameters_cy_elbow_tee_anchor_group0805.json")

'''
因为三通的端点可能是拐弯、圆柱，所以拟合三通时不再具体询问是什么部件，只对关键点进行处理
也即不用load_cylinders、save_elbows中的数据，用para_top_bottom中的数据
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_consider_cylinders', help="neighbor cylinders taken into account", default=10)
    parser.add_argument('--p_coverage_thresh', type=float, default=0.01, help="consider 0.1 or 0.05")
    parser.add_argument('--threshold_radius', help="threshold of two cylinders in cast intersection", default=0)
    parser.add_argument('--threshold_parallel', help="threshold of two parallel cylinders", default=0.9)  # 限制为30度以内

    args = parser.parse_args()

    globals.load_cylinders = Cylinder.load_cylinders_from_json("parameters_cy_elbow_anchor_group0805.json")
    cylinder_nums = len(globals.load_cylinders)
    globals.save_elbows = Torus.load_elbows_from_json("parameters_cy_elbow_anchor_group0805.json")
    elbow_nums = len(globals.save_elbows)

    globals.load_anchors = Anchor.load_anchors_from_json('parameters_cy_elbow_anchor_group0805.json')
    globals.save_groups = Group.load_groups_from_json('parameters_cy_elbow_anchor_group0805.json')
    globals.group_tid=len(globals.save_groups)


    globals.load_primitives = globals.load_cylinders + globals.save_elbows
    sides_para_numpy = np.asarray([primitive.sides_get() for primitive in globals.load_primitives])
    primitive_nums = len(globals.load_primitives)
    globals.para_top_bottom = list(
        (row1, row2) for row1, row2 in zip(sides_para_numpy[:, :3], sides_para_numpy[:, 3:6]))

    tees_points = get_tees_points()

    used_primitive_sides = [[False, False] for _ in range(primitive_nums)]

    main()
