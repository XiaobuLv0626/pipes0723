import numpy as np
import open3d as o3d
import math
from anchor import Anchor
import globals

def find_anchor(specific_id):
    return [obj for obj in globals.anchors if obj.tid == specific_id][0]

class Cylinder:
    def __init__(self, top_id, bottom_id, radius, group_id):
        self.top_id = top_id
        self.bottom_id = bottom_id
        self.radius = radius
        self.group_id = group_id

        self.top_center=find_anchor(self.top_id).coord
        self.bottom_center=find_anchor(self.bottom_id).coord



    @classmethod
    def load_cylinders_from_json(cls, json_file: str) -> 'list[Cylinder]':
        import json
        with open(json_file, 'r') as f:
            data = json.load(f)
        cylinders = []
        if isinstance(data, dict) and 'cylinders' in data:
            for item in data['cylinders']:
                cylinder = cls(item['top_id'], item['bottom_id'], item['radius'], item['group_id'])
                cylinders.append(cylinder)
        return cylinders

    def get_height(self) -> float:
        return np.linalg.norm(np.array(self.top_center) - np.array(self.bottom_center))

    def get_direction(self) -> 'np.array[float, float, float]':
        delta = np.array(self.top_center) - np.array(self.bottom_center)
        return delta / np.linalg.norm(delta)

    def get_rotation_matrix(self) -> 'np.array[[float, float, float], [float, float, float], [float, float, float]]':
        z = self.get_direction()
        y = np.cross(z, np.array([0, 0, 1]))
        y = y / np.linalg.norm(y) if np.abs(np.linalg.norm(y)) > 1e-6 else np.array([0, 1, 0])
        x = np.cross(y, z)
        assert np.abs(np.linalg.norm(x)) > 1e-6
        x = x / np.linalg.norm(x)
        rotation_matrix = np.array([x, y, z])
        return rotation_matrix


    def to_o3d_mesh(self) -> 'o3d.geometry.TriangleMesh':
        cylinder_mesh = o3d.geometry.TriangleMesh.create_cylinder(self.radius, self.get_height())
        rotation_matrix = self.get_rotation_matrix()
        cylinder_mesh.rotate(rotation_matrix.T, center=(0, 0, 0))
        cylinder_mesh.compute_vertex_normals()
        center = (np.array(self.top_center) + np.array(self.bottom_center)) / 2
        cylinder_mesh.translate(center)
        return cylinder_mesh
