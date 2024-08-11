from cylinder import Cylinder
from elbow import Elbow
from anchor import Anchor
from tee import Tee
from group import Group
import open3d as o3d
import globals

globals.anchors = Anchor.load_anchors_from_json('test_para.json')
globals.groups = Group.load_groups_from_json('test_para.json')

cylinders = Cylinder.load_cylinders_from_json('test_para.json')
mesh_list=[cylinder.to_o3d_mesh() for cylinder in cylinders]
tees = Tee.load_tees_from_json('test_para.json')
mesh_list+=[tee.to_o3d_mesh() for tee in tees]
elbows = Elbow.load_elbows_from_json('test_para.json')
mesh_list+=[elbow.to_o3d_mesh() for elbow in elbows]
mesh=o3d.geometry.TriangleMesh()
for t in mesh_list:
    mesh+=t
# o3d.visualization.draw_geometries(mesh_list)
o3d.io.write_triangle_mesh("example.obj",mesh)