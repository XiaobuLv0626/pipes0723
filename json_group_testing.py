from cylinder import Cylinder
from torus import Torus
from anchor import Anchor
from tee import Tee
from group import Group
import open3d as o3d
import globals
import numpy as np
from stage2_group_analyse import find_tee, find_elbow, find_cylinder


def visual_json_test():
    anchors = Anchor.load_anchors_from_json('parameters_cy_elbow_tee_anchor_group0805.json')
    cylinders = Cylinder.load_cylinders_from_json('parameters_cy_elbow_tee_anchor_group0805.json')
    elbows = Torus.load_elbows_from_json("parameters_cy_elbow_tee_anchor_group0805.json")
    tees = Tee.load_tees_from_json('parameters_cy_elbow_tee_anchor_group0805.json')

    print("json mesh test")
    mesh_list = [cylinder.to_o3d_mesh() for cylinder in cylinders]
    mesh_list += [tee.to_o3d_mesh() for tee in tees]
    mesh_list += [elbow.to_o3d_mesh() for elbow in elbows]
    o3d.visualization.draw_geometries(mesh_list)

    print("json anchors test")
    anchor_list = [anc.coord for anc in anchors]
    points = np.asarray([anchors[1].get_coord(), anchors[2].get_coord()])
    print(points.shape)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(anchor_list)
    print(anchor_list)
    o3d.visualization.draw_geometries([point_cloud])


def get_torus_elbow(elbows, tees, anchors):
    elbow_tee_anchor_list = []
    elbow_tee_anchor_id_list = []
    for elbow in elbows:
        elbow_tee_anchor_list.append(anchors[elbow.p1_id].get_coord())
        elbow_tee_anchor_list.append(anchors[elbow.p2_id].get_coord())
        elbow_tee_anchor_id_list.append([1, elbow.p1_id])
        elbow_tee_anchor_id_list.append([1, elbow.p2_id])
    for tee in tees:
        elbow_tee_anchor_list.append(anchors[tee.top1_id].get_coord())
        elbow_tee_anchor_list.append(anchors[tee.bottom1_id].get_coord())
        elbow_tee_anchor_list.append(anchors[tee.top2_id].get_coord())
        elbow_tee_anchor_id_list.append([2, tee.top1_id])
        elbow_tee_anchor_id_list.append([2, tee.bottom1_id])
        elbow_tee_anchor_id_list.append([2, tee.top2_id])

    elbow_tee_anchor_list = np.asarray(elbow_tee_anchor_list)
    elbow_tee_anchor_list = np.concatenate((elbow_tee_anchor_list, np.asarray(elbow_tee_anchor_id_list)),
                                           axis=1)
    return elbow_tee_anchor_list
