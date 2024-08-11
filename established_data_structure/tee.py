import open3d as o3d
from cylinder import Cylinder
import globals
import trimesh
import numpy as np


def find_anchor(specific_id):
    return [obj for obj in globals.anchors if obj.tid == specific_id][0]


class Tee:
    def __init__(self, tid, top1_id, bottom1_id, top2_id, bottom2_id, radius1, radius2, group_id):
        self.tid = tid
        self.top1_id = top1_id
        self.bottom1_id = bottom1_id
        self.top2_id = top2_id
        self.bottom2_id = bottom2_id
        self.radius1 = radius1
        self.radius2 = radius2
        self.group_id = group_id

        # # 计算额外参数
        # self.top1 = find_anchor(self.top1_id).coord
        # self.bottom1 = find_anchor(self.bottom1_id).coord
        # self.top2 = find_anchor(self.top2_id).coord
        # self.bottom2 = find_anchor(self.bottom2_id).coord

    @classmethod
    def load_tees_from_json(cls, json_file: str) -> 'list[Tee]':
        import json
        with open(json_file, 'r') as f:
            data = json.load(f)
        tees = []
        if isinstance(data, dict) and 'tees' in data:
            for item in data['tees']:
                tee = cls(item['tid'], item['top1_id'], item['bottom1_id'], item['top2_id'], item['bottom2_id'], item['radius1'], item['radius2'], item['group_id'])
                tees.append(tee)
        return tees

    def to_o3d_mesh(self)->'o3d.geometry.TriangleMesh':
        cylinderx=Cylinder(self.top1_id,self.bottom1_id,self.radius1,self.group_id)
        cylinderz=Cylinder(self.top2_id,self.bottom2_id,self.radius2,self.group_id)
        o3d_cylinderx=cylinderx.to_o3d_mesh()
        o3d_cylinderz=cylinderz.to_o3d_mesh()
        # 转换数据格式，求交后再变换回去
        tri_cylinderx=trimesh.Trimesh(vertices=np.asarray(o3d_cylinderx.vertices),
                                      faces=np.asarray(o3d_cylinderx.triangles),
                                      process=False)
        tri_cylinderz=trimesh.Trimesh(vertices=np.asarray(o3d_cylinderz.vertices),
                                      faces=np.asarray(o3d_cylinderz.triangles),
                                      process=False)
        tee=trimesh.boolean.union([tri_cylinderx,tri_cylinderz])
        tee.export("tee.obj")
        mesh=o3d.geometry.TriangleMesh()
        mesh.vertices=o3d.utility.Vector3dVector(tee.vertices)
        mesh.triangles=o3d.utility.Vector3iVector(tee.faces)
        mesh.compute_vertex_normals()
        return mesh
