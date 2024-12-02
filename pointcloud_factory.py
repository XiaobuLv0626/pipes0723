import open3d as o3d
import numpy as np


class FactoryPointcloudManager:
    # FactoryPointcloud接受（项目内）多种格式的点云数据，以numpy数组格式输出其对应的点云/标签，以pcd/npy格式重新导出对应点云
    def __init__(self, data_type, mapping_path, *paths):
        self.point = np.empty((0, 8))
        self.data_type = data_type
        with open(mapping_path, 'r') as file:
            self.inst_to_obj = [int(line.strip()) for line in file.readlines()]
        if data_type == "pcd":
            self.point_path = self.label_path = paths[0]
        elif data_type == "npy":
            self.point_path = paths[0]
            self.label_path = paths[1]
        else:
            raise Exception("FactoryPointcloud only support \"pcd/npy\" type. Try Again.")
        self.load_pcd_and_labels()


    def load_pcd_and_labels(self):
        """
        将对应的pcd/npy+labels数据格式数据读取到该结构对应的numpy数组内，防止之前版本的多次加载。
        """
        if self.data_type == "npy":
            point = np.load(self.point_path)
            shape = np.load(self.label_path).reshape(-1, 2)
            self.point = np.concatenate((point, shape), axis=1)

        elif self.data_type == "pcd":
            _point_cloud = o3d.t.io.read_point_cloud(self.point_path)
            self.point = np.concatenate((_point_cloud.point.positions.numpy(),
                                         _point_cloud.point.colors.numpy(),
                                         _point_cloud.point.instance_type.numpy(),
                                         _point_cloud.point.instance_label.numpy()), axis=1)

    def __str__(self):
        return f"Data type:{self.data_type} with pointcloud path {self.point_path} and labels path {self.label_path}"

    def get_pointcloud(self):
        """
        获取点云的坐标信息与颜色信息

        :return: (N, 6)的numpy数组，前三维为xyz坐标，后三维为RGB值
        """
        return self.point[:, :6]

    def get_object_type(self):
        """
        获取点云的一级语义标签，即点云属于（管道/管廊支架/储罐/钢结构/地面/人影鬼影/其他/联排管道/预测失败）中的哪一类。

        返回值的具体细节详见飞书文档《H1-H3格式统一后numpy存储结构》

        :return: （N）的numpy一维数组，表示整个点云中每个点的语义类别
        """
        point_inst_label = self.point[:, 6]
        return np.asarray([self.inst_to_obj[x] for x in point_inst_label])

    def get_instance_type(self):
        """
        获取点云的二级语义（实例）标签，返回值的具体细节详见飞书文档《H1-H3格式统一后numpy存储结构》

        :return: （N）的numpy一维数组，表示整个点云中每个点的实例类别
        """
        return self.point[:, 6]

    def get_instance_label(self):
        """
        获取点云的二级语义实例编号，即对应点所属的实例编号。编号为-1代表不存在对应实例

        :return:（N）的numpy一维数组，表示整个点云中每个点的实例标号
        """
        return self.point[:, 7]

    def get_full_cloud(self):
        """
        获取整个点云，包括点云的坐标，一级语义类型与编号，二级语义类型与标号，以方便算法做整体处理

        :return: (N,10)的Numpy数组，格式同pcd存储格式
        """
        obj_t = self.get_object_type().reshape(-1, 1)
        result_pcd = np.insert(self.point, 6, obj_t, axis=1)
        result_pcd = np.insert(result_pcd, 6, obj_t, axis=1)
        print(result_pcd.shape)
        return result_pcd

    def get_pointcloud_by_type(self, inst:int):
        """
        根据给定的二级语义类型筛选得到该类型所有的点云

        :return: (N, 10)的Numpy数组，格式同pcd储存格式
        """
        full_pcd = self.get_full_cloud()
        return full_pcd[full_pcd[:, 8] == inst]

    def get_pointcloud_by_id(self, label:int):
        """
        根据给定的二级语义类型筛选得到该类型所有的点云

        :return: (N, 10)的Numpy数组，格式同pcd储存格式
        """
        full_pcd = self.get_full_cloud()
        return full_pcd[full_pcd[:, 9] == label]

    def get_pointcloud_by_type_label(self, inst:int, label:int):
        """
        针对改版后的数据使用的类型独立编号设计，通过二级语义类型号与编号确定点云

        :return: (N, 10)的Numpy数组，格式同pcd储存格式
        """
        full_pcd = self.get_full_cloud()
        inst_pcd = full_pcd[full_pcd[:, 8] == inst]
        return inst_pcd[inst_pcd[:, 9] == label]


if __name__ == "__main__":
    pcd_name = "F:/FactPoints/H2_sample_for_handcraft.pcd"
    npy_name = 'F:/FactPoints/H2_sample_for_handcraft.npy'
    labels_name = "F:/FactPoints/H2_sample_for_handcraft_labels.npy"
    Test_cloud = FactoryPointcloudManager('npy', 'mapping.txt', npy_name, labels_name)
    # Test_cloud = FactoryPointcloudManager("pcd", pcd_name)
