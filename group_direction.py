# 打组的方向估计
# 若一个组内无三通四通，则一定有一个完整的空间逻辑，根据每个基元的关键点进行排序即可，按照排序赋一个基础方向；
# 因此尝试根据三通四通切分得到的组，将三通四通链接着的部件之间进行切分
# 这需要对打组部分进行一定的修改，主要是对打组的一些特定操作进行限制
# 打组的优化修改
# 一个修改为尝试进行建图，根据每个基元对应的类型限制其链接的最大基元数
#    管道与拐弯限制为最多链接两个最近的基元，三通四通最多三/四个最近的基元，尝试此法是否能够有效限制误打组情况
# 另一个修改为重新回到尊重点云的状态
#    即将基于json/基于点云的打组方法分割开来，并由软件进行两种打组方法的评分，自动选择更优的一种方法
# 还有一个修改是对于输入点云中对应基元的处理
#    点云中存在未标注基元的部分，通过计算包围盒，规模大于场景1/5的，直接将这部分清除
#    点云中存在离群点的部分，通过RANSAC聚类将其离群点全部删去（阈值尚待探索）

import networkx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.style.core import available
from networkx import Graph
from scipy.spatial import distance_matrix

from cylinder import Cylinder
from anchor import Anchor
from stage2_group_analyse import find_cylinder, find_elbow, find_tee
from test_group import farthest_points
from torus import Torus
from tee import Tee
from pointcloud_factory import FactoryPointcloudManager


class PipeGroupGraph:
    # 作为Pipe建图的基础数据结构使用
    # 有关整个打组过程中如何根据基元限制调整图的属性，则交给PipeGroupConstructor考虑

    def __init__(self):
        self.group_graph = Graph()

    def instance_num(self):
        return len(list(self.group_graph.nodes))

    def add_instance(self, inst_type: str, inst_num: int):
        self.group_graph.add_node(inst_num, type=inst_type)

    def add_edge(self, part_a: int, part_b: int):
        self.group_graph.add_edge(part_a, part_b)

    def add_weight_edge(self, part_a: int, part_b: int, _weight: float, _point_pair: tuple):
        self.group_graph.add_edge(part_a, part_b, weight=_weight, point_pair=_point_pair)

    def remove_edge(self, part_a: int, part_b: int):
        self.group_graph.remove_edge(part_a, part_b)

    def connected_instance(self, part: int):
        return list(self.group_graph.neighbors(part))

    def connected_graph(self):
        return list(networkx.connected_components(self.group_graph))

    def visualize(self):
        networkx.draw(self.group_graph)

    @property
    def graph(self) -> networkx.Graph:
        return self.group_graph

    def clear(self):
        self.group_graph.clear()


class PipeGroupConstructor:
    # 负责建图（即确认两个基元是否应归于同一组内）
    # 同时也是类似于原代码中global group的存在，最后会根据图中森林的棵树提取对应的组
    max_branch = {"Cylinder": 3, "Torus": 2, "Tee": 3, "Flange": 2, "Valve": 2}
    inst_id = {"Cylinder": 11, "Torus": 12, "Tee": 13, "Flange": 14, "Valve": 15}

    def __init__(self, json, factory: FactoryPointcloudManager):
        self.pipe_graph = PipeGroupGraph()
        self.data_factory = factory
        self.anchors = []
        self.cylinders = Cylinder.load_cylinders_from_json(json)
        self.tori = Torus.load_elbows_from_json(json)
        self.tees = Tee.load_tees_from_json(json)
        self.crucial_used = [[False, False, False] for i in range(len(self.cylinders)+len(self.tori)+len(self.tees))]

    def initialize_all_pipe_inst(self, parts_list):
        """
        输入打组部分的管道部件列表(tuple list)，将其初始化至图内
        """
        for part in parts_list:
            self.pipe_graph.add_instance(part[0], part[1])

    def initialize_with_json(self):
        """
        使用提供的json初始化场景内所有的json对象，将其初始化入图中，并初始化关键点

        Coord是目前的关键点，因此也就不用存anchors了，把cylinder/torus/tees存起来，并记录使用情况即可。
        """
        for cylinder in self.cylinders:
            self.pipe_graph.add_instance('Cylinder', cylinder.tid)
        for torus in self.tori:
            self.pipe_graph.add_instance('Torus', torus.tid + len(self.cylinders))
        for tee in self.tees:
            self.pipe_graph.add_instance("Tee", tee.tid + len(self.cylinders) + len(self.tori))

    def get_crucial_point(self, inst: int):
        """
        根据实例的编号取其关键点
        """
        inst_type = self.pipe_graph.graph.nodes[inst]['type']
        if inst > len(self.cylinders) + len(self.tori):
            inst -= len(self.cylinders) + len(self.tori)
        elif inst > len(self.cylinders):
            inst -= len(self.cylinders)
        if inst_type == 'Cylinder':
            return find_cylinder(inst)
        elif inst_type == 'Torus':
            return find_elbow(inst)
        else:
            return find_tee(inst)


    def calculate_dist_of_inst(self, inst_1: int, inst_2: int, method: str):
        """
        根据输入的基元编号计算两个基元的最近距离与对应点对，返回点对相对坐标与距离
        提供两种处理方式：基于点云的距离计算/基于json的点云计算

        return:
        inst_1与inst_2最近的关键点对，与它们之间的距离。
        """
        type1, type2 = self.pipe_graph.graph.nodes[inst_1]['type'], self.pipe_graph.graph.nodes[inst_2]['type']
        if method == 'pcd':
            # Point Cloud Based Methods using Ball_query
            points_inst_1 = self.data_factory.get_pointcloud_by_type_label(self.inst_id[type1], inst_1)
            points_inst_2 = self.data_factory.get_pointcloud_by_type_label(self.inst_id[type2], inst_2)
            crucial_point_1 = farthest_points(points_inst_1)
            crucial_point_2 = farthest_points(points_inst_2)

        # Json Based Methods using crucial points
        else:
            crucial_point_1 = self.get_crucial_point(inst_1)
            crucial_point_2 = self.get_crucial_point(inst_2)

        diff = crucial_point_1[:, np.newaxis, :] - crucial_point_2[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))

        max_position = np.unravel_index(np.argmax(dist_matrix, axis=None), dist_matrix.shape)
        return crucial_point_1[max_position[0]], crucial_point_2[max_position[1]], dist_matrix[max_position]


    def grouping_by_pcd(self, inst_1: int, inst_type_1: int, inst_2: int, inst_type_2: int):


    def grouping_by_json(self, inst_1: int, inst_2: int, point_pair:(int, int), dist: float):
        """
        尝试将两个不同部件归至同一组
        对每个基元关键点记录是否使用过，使用过的情况下邻居的编号是多少
        未使用过则选择该点进行连接，并将距离作为边权
        使用过，查看目前的点对距离是否更小，是则选择替换。
        """
        # 找到两个部件对应的类型
        type1, type2 = self.pipe_graph.graph.nodes[inst_1]['type'], self.pipe_graph.graph.nodes[inst_2]['type']
        available_1 = self.crucial_used[inst_1][point_pair[0]]
        available_2 = self.crucial_used[inst_2][point_pair[1]]

        if available_1 == available_2 and available_1 == False:
            self.pipe_graph.add_weight_edge(inst_1, inst_2, dist, point_pair)
            self.crucial_used[inst_1][point_pair[0]] = inst_2
            self.crucial_used[inst_2][point_pair[1]] = inst_1
        else:
            weight1, weight2 = 10000.0, 10000.0
            neighbor_crit_1, neighbor_crit_2 = -1, -1
            if available_1:
                weight1 = self.pipe_graph.graph[inst_1][available_1]['weight']
                last_edge = self.pipe_graph.graph[inst_1][available_1]['point_pair']
                if last_edge[0] == point_pair[0]:
                    neighbor_crit_1 = last_edge[1]
                else:
                    neighbor_crit_1 = last_edge[0]
            if available_2:
                weight2 = self.pipe_graph.graph[inst_2][available_2]['weight']
                last_edge = self.pipe_graph.graph[inst_2][available_2]['point_pair']
                if last_edge[0] == point_pair[1]:
                    neighbor_crit_2 = last_edge[1]
                else:
                    neighbor_crit_2 = last_edge[0]
            if dist < weight1 and dist < weight2:
                self.pipe_graph.add_weight_edge(inst_1, inst_2, dist, point_pair)
                self.crucial_used[inst_1][point_pair[0]] = inst_2
                self.crucial_used[inst_2][point_pair[1]] = inst_1
                if neighbor_crit_1 != -1:
                    self.crucial_used[available_1][neighbor_crit_1] = False
                if neighbor_crit_2 != -1:
                    self.crucial_used[available_2][neighbor_crit_2] = False



    def check_grouping_metrics(self):
        """
        计算打组覆盖率，即该图上所有组数/点数目
        """
        return len(self.pipe_graph.connected_graph())/self.pipe_graph.instance_num()

    def print_graph(self):
        group_list = networkx.connected_components(self.pipe_graph)
        for group in group_list:
            sub = self.pipe_graph.graph.subgraph(group).copy()


test_graph = PipeGroupGraph()
test_graph.add_instance("Cylinder", 445)
test_graph.add_instance("Torus", 15)
test_graph.add_instance("Cylinder", 443)
test_graph.add_edge(445, 15)
print(test_graph.graph.nodes[443]['type'])
