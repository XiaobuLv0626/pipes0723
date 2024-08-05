from typing import Tuple, Any

import numpy as np
import open3d as o3d
import trimesh
from cylinder import Cylinder

class Tee:
    def __init__(self, top1,bottom1,radius1,top2,bottom2,radius2,tid=-1,top1_id=-1,bottom1_id=-1,top2_id=-1,group_id=-1):
        self.top1 = top1
        self.bottom1 = bottom1
        self.radius1 = radius1
        self.top2 = top2
        self.bottom2 = bottom2
        self.radius2 = radius2
        self.tid=tid
        self.top1_id=top1_id # 按照如下顺序，top1用P1标志位表示，bottom1用P2，top2用P3
        self.bottom1_id=bottom1_id
        self.top2_id=top2_id # bottom2_id只有四通才有
        self.group_id=group_id

    def get_direction(self) -> 'np.array[float,float,float]':
        pass

    def get_rotation_matrix(self) -> 'np.array[[float,float,float],[float,float,float],[float,float,float]]':
        pass
    def get(self) -> tuple[Any, Any, Any, Any, Any, Any]:
        return self.top1, self.bottom1,self.radius1,self.top2,self.bottom2, self.radius2,self.tid,self.top1_id,self.bottom1_id,self.top2_id,self.group_id

    def to_o3d_mesh(self)->'o3d.geometry.TriangleMesh':
        cylinderx=Cylinder(self.top1,self.bottom1,self.radius1)
        cylinderz=Cylinder(self.top2,self.bottom2,self.radius2)
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
    @classmethod
    def load_tees_from_json(cls, json_file: str) -> 'list[Tee]':
        import json
        with open(json_file, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'tees' in data:
            data = data['tees']
        assert isinstance(data, list)
        cylinders = []
        for item in data:
            cylinder = cls(np.array(item['top1']), np.array(item['bottom1']), item['radius1'],
                    np.array(item['top2']), np.array(item['bottom2']), item['radius2']
                           ,item['tid'],item['top1_id'],item['bottom1_id'],item['top2_id'],item['group_id'])
            cylinders.append(cylinder)
        return cylinders
