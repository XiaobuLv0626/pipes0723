from cylinder import Cylinder
# from elbow import Elbow
from torus import Torus
from anchor import Anchor
from tee import Tee
from group import Group
import open3d as o3d
import globals
import numpy as np

def find_cylinder(specific_id):
    return [obj for obj in cylinders if obj.tid == specific_id][0]

def find_elbow(specific_id):
    return [obj for obj in elbows if obj.tid == specific_id][0]

def find_tee(specific_id):
    return [obj for obj in tees if obj.tid == specific_id][0]


anchors = Anchor.load_anchors_from_json('parameters_cy_elbow_tee_anchor_group.json')
groups = Group.load_groups_from_json('parameters_cy_elbow_tee_anchor_group.json')
cylinders = Cylinder.load_cylinders_from_json('parameters_cy_elbow_tee_anchor_group.json')
elbows = Torus.load_elbows_from_json("parameters_cy_elbow_tee_anchor_group.json")
tees = Tee.load_tees_from_json('parameters_cy_elbow_tee_anchor_group.json')

def test_inst_to_tid():
    pcd = o3d.t.io.read_point_cloud("data/H2_grid_2_labels_remake_20240926.pcd")

    points = np.asarray(pcd.point['positions'].numpy())
    colors = np.asarray(pcd.point['colors'].numpy())
    object_type = np.asarray(pcd.point['object_type'].numpy())
    object_label = np.asarray(pcd.point['object_label'].numpy())
    instance_type = np.asarray(pcd.point['instance_type'].numpy())
    instance_label = np.asarray(pcd.point['instance_label'].numpy())
    full_cloud = np.concatenate((points, colors, object_type, object_label, instance_type, instance_label), axis=1)
    pipe_cloud = full_cloud[full_cloud[:, 8] == 10]

    value, count = np.unique(pipe_cloud[:, 9], return_counts=True)
    print(value.shape)
    print(len(cylinders))
    for i in range(value.shape[0]):
        mesh = []
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pipe_cloud[pipe_cloud[:, 9] == value[i]][:, :3])
        mesh.append(cylinders[i-2].to_o3d_mesh())
        mesh.append(pcd)
        o3d.visualization.draw_geometries(mesh)



if __name__ == "__main__":
    # test_inst_to_tid()
    for group in groups:
        print(f"X={group.tid}")
        mesh=[]
        parts=group.parts
        for part in parts:
            type,tid=part
            if type=="cylinder":
                mesh.append(find_cylinder(tid).to_o3d_mesh())
                pcd=o3d.geometry.PointCloud()
                points=np.load("cylinders/cylinders_inst.npy")
                points=points[points[:,3]==tid][:,:3]
                pcd.points=o3d.utility.Vector3dVector(points)
                mesh.append(pcd)
            elif type=="elbow":
                mesh.append(find_elbow(tid).to_o3d_mesh())
                points=np.load("torus/torus_inst.npy")
                points=points[points[:,6]==tid][:,:3]
                pcd = o3d.geometry.PointCloud()
                pcd.points=o3d.utility.Vector3dVector(points)
                mesh.append(pcd)
            elif type=="tee":
                mesh.append(find_tee(tid).to_o3d_mesh())
                points=np.load("tees/tees_inst.npy")
                points=points[points[:,6]==tid][:,:3]
                pcd = o3d.geometry.PointCloud()
                pcd.points=o3d.utility.Vector3dVector(points)
                mesh.append(pcd)
        o3d.visualization.draw_geometries(mesh)

    print("test")
    # mesh_list=[cylinder.to_o3d_mesh() for cylinder in cylinders]
    # mesh_list+=[tee.to_o3d_mesh() for tee in tees]
    # elbows = Elbow.load_elbows_from_json('test_para.json')
    # mesh_list+=[elbow.to_o3d_mesh() for elbow in elbows]
    # mesh=o3d.geometry.TriangleMesh()
    # for t in mesh_list:
    #     mesh+=t
    # o3d.visualization.draw_geometries(mesh_list)
    # o3d.io.write_triangle_mesh("example.obj",mesh)