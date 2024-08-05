import sys
sys.path.append("../")
import numpy as np
from cylinder import Cylinder
import json
import open3d as o3d
from anchor import Anchor

'''为cylinder类添加anchor_id，并且将坐标添加到关键点类中'''

# 自定义编码器类
class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.ndarray):
            return list(obj)  # 将 ndarray 转换为 Python 列表
        else:
            return super().default(obj)

def save_json():
    json_cylinders = []
    json_cylinder_idx = 0
    json_anchors=[]
    for x in load_cylinders:
        json_data = {
            "top_center":x.top_center,
            "bottom_center":x.bottom_center,
            "radius": x.radius,
            "tid": x.tid,
            "top_id": x.top_id,
            "bottom_id": x.bottom_id,
            "group_id":x.group_id
        }
        json_cylinders.append(json_data)
    for x in anchors:
        json_data={
            "tid":x.tid,
            "coord":x.coord,
            "parts":x.parts,
            "group_id":x.group_id
        }
        json_anchors.append(json_data)
    json_final=[]
    json_final.append({"cylinders":json_cylinders})
    json_final.append({"anchors":json_anchors})

    json_string = json.dumps({"cylinders":json_cylinders,"anchors":json_anchors},cls=EnumEncoder)
    with open("./parameters_cylinders_anchor.json", 'w') as outfile:
        outfile.write(json_string)

load_cylinders = Cylinder.load_cylinders('./ransac')
idx=0
cy_index=0
anchors=[]
# mesh=o3d.geometry.TriangleMesh()
for t in load_cylinders:
    # mesh+=t.to_o3d_mesh()
    t.tid=cy_index
    t.top_id=idx
    temp_anchor=Anchor(idx)
    temp_anchor.add_coord(t.top_center)
    temp_anchor.add_part(("cylinder",cy_index))
    anchors.append(temp_anchor)
    idx+=1
    t.bottom_id=idx
    temp_anchor = Anchor(idx)
    temp_anchor.add_coord(t.bottom_center)
    temp_anchor.add_part(("cylinder", cy_index))
    anchors.append(temp_anchor)
    idx+=1
    cy_index+=1

# o3d.io.write_triangle_mesh("cylinders.obj",mesh)
save_json()













