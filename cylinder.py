import numpy as np
import open3d as o3d
import math
from primitive import Primitive
class Cylinder(Primitive):
    def __init__(self ,top_center: 'tuple[float, float, float]', bottom_center: 'tuple[float, float, float]', radius: float,tid=-1,top_id=-1,bottom_id=-1,group_id=-1):
        super().__init__(top_center,bottom_center,radius,top_id,bottom_id)
        self.top_center = top_center
        self.bottom_center = bottom_center
        self.radius = radius
        self.points = None
        self.tid=tid # tid为全局id，用于索引点云
        self.top_id=top_id # 顶面对应的anchor编号
        self.bottom_id=bottom_id
        self.group_id=group_id

    # 判断两个圆柱是否相同
    def __eq__(self, other):
        if isinstance(other,Cylinder):
            if np.allclose(self.top_center,other.top_center, rtol=1e-5) and np.allclose(self.bottom_center,other.bottom_center, rtol=1e-5)\
                and np.allclose(self.radius,other.radius, rtol=1e-5):
                return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def get(self) -> 'tuple[tuple[float, float, float], tuple[float, float, float], float]':
        return self.top_center, self.bottom_center, self.radius,self.tid,self.top_id,self.bottom_id,self.group_id

    def split_get(self) -> 'tuple[float, float, float, float, float, float, float]':
        return self.top_center[0], self.top_center[1], self.top_center[2], self.bottom_center[0], self.bottom_center[1], self.bottom_center[2], self.radius

    def numpy_get(self) -> 'np.array[float, float, float, float, float, float, float]':
        return np.array([self.top_center[0], self.top_center[1], self.top_center[2], self.bottom_center[0], self.bottom_center[1], self.bottom_center[2], self.radius])

    def get_direction(self) -> 'np.array[float, float, float]':
        delta = np.array(self.top_center) - np.array(self.bottom_center)
        return delta / np.linalg.norm(delta)

    def get_height(self) -> float:
        return np.linalg.norm(np.array(self.top_center) - np.array(self.bottom_center))

    def get_distance_from_another_cylinder(self, another_cylinder: 'Cylinder') -> float:
        # TODO: check if the two cylinders are not parallel
        minimum = 1e9
        for point in [self.top_center, self.bottom_center]:
            for another_point in [another_cylinder.top_center, another_cylinder.bottom_center]:
                minimum = min(minimum, np.linalg.norm(np.array(point) - np.array(another_point)))
        return minimum

    def is_point_inside(self, point: 'tuple(float, float, float)') -> bool:
        direction = self.get_direction()
        delta = np.array(point) - np.array(self.bottom_center)
        if np.linalg.norm(np.cross(delta, direction)) > self.radius:
            return False
        if np.dot(delta, direction) < 0 or np.dot(delta, direction) > np.linalg.norm(np.array(self.bottom_center) - np.array(self.top_center)):
            return False
        return True

    def is_point_near_the_surface(self, point: 'tuple(float, float, float)',
                                  threshold: float) -> bool:
        direction = self.get_direction()
        delta = np.array(point) - np.array(self.bottom_center)
        if np.linalg.norm(np.cross(delta, direction)) > self.radius + threshold or \
            np.linalg.norm(np.cross(delta, direction)) < self.radius - threshold:
            return False
        if np.dot(delta, direction) < 0-threshold or \
            np.dot(delta, direction) > np.linalg.norm(np.array(self.bottom_center) - np.array(self.top_center))+threshold:
            return False
        return True

    def is_point_near_the_surface_batch(self, points: 'np.array[[float, float, float]]',
                                        threshold: float) -> 'np.array[bool]':
        direction = self.get_direction()
        delta = np.array(points) - np.array(self.bottom_center)
        delta_cross = np.linalg.norm(np.cross(delta, direction), axis=1)
        delta_dot = np.dot(delta, direction)
        condition1 = np.logical_and(delta_cross > self.radius - threshold,
                                    delta_cross < self.radius + threshold)
        condition2 = np.logical_and(delta_dot > -threshold,
                                    delta_dot < self.get_height() + threshold)
        return np.logical_and(condition1, condition2)

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

    def get_minimum_bounding_box(self) -> np.ndarray: # (8, 3)
        # TODO: proceed quickly
        return np.asarray(self.to_o3d_mesh().get_minimal_bounding_box().get_box_points())
        pass

    def set_points(self, points: 'np.array[[float, float, float]]') -> None:
        self.points = points

    def reverse(self) -> 'Cylinder':
        new = Cylinder(self.bottom_center, self.top_center, self.radius)
        new.set_points(self.points)
        return new

    @classmethod
    def can_merge_together(cls, cylinder_x: 'Cylinder', cylinder_y: 'Cylinder', angle_threshold: float,
                           distance_threshold: float, radius_threshold_rate: float) -> bool:
        distance_threshold=abs(cylinder_x.radius+cylinder_y.radius)/2

        if np.abs(np.dot(cylinder_x.get_direction(), cylinder_y.get_direction())) < np.cos(angle_threshold):
            return False
        # 放宽条件
        if cylinder_x.is_point_near_the_surface(cylinder_y.top_center,distance_threshold):
            return True
        if cylinder_x.is_point_near_the_surface(cylinder_y.bottom_center,distance_threshold):
            return True
        if cylinder_y.is_point_near_the_surface(cylinder_x.top_center,distance_threshold):
            return True
        if cylinder_y.is_point_near_the_surface(cylinder_x.bottom_center,distance_threshold):
            return True

        if cylinder_x.get_distance_from_another_cylinder(cylinder_y) > distance_threshold:
            return False

        if abs(cylinder_x.radius - cylinder_y.radius) > radius_threshold_rate * max(cylinder_x.radius, cylinder_y.radius):
            return False
        return True

    @classmethod
    def dis_from_two_points(cls, point_x: 'tuple[float, float, float]', point_y: 'tuple[float, float, float]') -> float:
        return np.linalg.norm(np.array(point_x) - np.array(point_y))

    @classmethod
    def save_cylinders(cls, cylinders: 'list[Cylinder]', save_prefix: str) -> None:
        cylinder_list, points_list = [], []
        for i, cylinder in enumerate(cylinders):
            _data = cylinder.split_get()
            _data = [*_data, i]
            cylinder_list.append(_data)
            _points = np.hstack((cylinder.points, np.ones((cylinder.points.shape[0], 1)) * i))
            points_list.append(_points)
        cylinder_list = np.vstack(cylinder_list)
        points_list = np.vstack(points_list)
        np.save(f'{save_prefix}_cylinders.npy', cylinder_list)
        np.save(f'{save_prefix}_points.npy', points_list)

    # @classmethod
    # def load_cylinders(cls, save_prefix: str) -> 'list[Cylinder]':
    #     cylinder_list = np.load(f'{save_prefix}_cylinders.npy')
    #
    #     import os, sys
    #     if os.path.exists(f'{save_prefix}_points.npy'):
    #         points_list = np.load(f'{save_prefix}_points.npy')
    #     else:
    #         print(f'Warning: {save_prefix}_points.npy not found, points loading skipped.', file=sys.stderr)
    #
    #     cylinders = []
    #     for i in range(cylinder_list.shape[0]):
    #         cylinder = Cylinder(cylinder_list[i,0],cylinder_list[i,1],cylinder_list[i,1 :3], cylinder_list[i, 3:6], cylinder_list[i, 6])
    #         if points_list is not None:
    #             cylinder.set_points(points_list[points_list[:, -1] == i, :-1])
    #         cylinders.append(cylinder)
    #     return cylinders


    @classmethod
    def load_cylinders(cls, save_prefix: str) -> 'list[Cylinder]':
        cylinder_list = np.load(f'{save_prefix}_cylinders.npy')

        import os, sys
        if os.path.exists(f'{save_prefix}_points.npy'):
            points_list = np.load(f'{save_prefix}_points.npy')
        else:
            print(f'Warning: {save_prefix}_points.npy not found, points loading skipped.', file=sys.stderr)

        cylinders = []
        for i in range(cylinder_list.shape[0]):
            cylinder = Cylinder(cylinder_list[i, :3], cylinder_list[i, 3:6], cylinder_list[i, 6])
            if points_list is not None:
                cylinder.set_points(points_list[points_list[:, -1] == i, :-1])
            cylinders.append(cylinder)
        return cylinders
    @classmethod
    def load_cylinders_from_json(cls, json_file: str) -> 'list[Cylinder]':
        import json
        with open(json_file, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'cylinders' in data:
            data = data['cylinders']
        assert isinstance(data, list)
        cylinders = []
        for item in data:
            cylinder = cls(np.array(item['top_center']), np.array(item['bottom_center']), item['radius'],item['tid'],item['top_id'],item['bottom_id'],item['group_id'])
            cylinders.append(cylinder)
        return cylinders

    @classmethod
    def stupid_merge(cls, cylinder_x: 'Cylinder', cylinder_y: 'Cylinder') -> 'Cylinder':
        """
        X bottom merge with Y top
        """
        def get_distance_from_point_to_point(point_x: 'tuple[float, float, float]', point_y: 'tuple[float, float, float]') -> float:
            return np.linalg.norm(np.array(point_x) - np.array(point_y))
        if get_distance_from_point_to_point(cylinder_x.bottom_center, cylinder_y.top_center) > \
            get_distance_from_point_to_point(cylinder_x.top_center, cylinder_y.top_center):
            cylinder_x = cylinder_x.reverse()
        if get_distance_from_point_to_point(cylinder_x.top_center, cylinder_y.top_center) > \
            get_distance_from_point_to_point(cylinder_x.top_center, cylinder_y.bottom_center):
            cylinder_y = cylinder_y.reverse()

        if cylinder_x.get_height()>cylinder_y.get_height()*3:
            cylinder=Cylinder(cylinder_x.top_center,np.asarray(cylinder_x.bottom_center)-cylinder_x.get_direction()*cylinder_y.get_height(),
                              (cylinder_x.radius + cylinder_y.radius) / 2)
        elif cylinder_y.get_height()>cylinder_x.get_height()*3:
            cylinder = Cylinder(np.asarray(cylinder_y.top_center)+cylinder_y.get_direction()*cylinder_x.get_height(), cylinder_y.bottom_center,
                                (cylinder_x.radius + cylinder_y.radius) / 2)
        else:
            cylinder = Cylinder(cylinder_x.top_center, cylinder_y.bottom_center, (cylinder_x.radius + cylinder_y.radius) / 2)
        cylinder.set_points(np.vstack((cylinder_x.points, cylinder_y.points)))
        return cylinder
        pass
