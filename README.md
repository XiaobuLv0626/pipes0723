# 文件说明
**cylinders/cylinders_inst.npy** 圆柱点云，大小为4*n，其中最后一列为标签
**torus/torus_inst.npy**拐弯点云，大小为7*m，其中最后一列为标签
**tees/tees_inst.npy**三通点云，大小为7*k，其中最后一列为标签
**established_data_structure**打组后的数据结构，相应的用法可以参考**main_config.py**，在这里调用**Elbow**可视化，而**Elbow**调用**Torus**来绘制基元
**stage2_group_analyse.py**第二阶段为打组，该代码导入打组的数据并遍历每一个组
