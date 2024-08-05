import numpy as np
import globals
import json

TOP = 0
BOTTOM = 1
P1=0
P2=1
P3=2



# 自定义编码器类
class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.ndarray):
            return list(obj)  # 将 ndarray 转换为 Python 列表
        else:
            return super().default(obj)

def parallel(alternative1,alternative2):
    cylinder_index1, side1 = alternative1
    cylinder_index2, side2 = alternative2
    cy1_dir = get_correct_dir(globals.load_cylinders[cylinder_index1], side1)
    cy2_dir = get_correct_dir(globals.load_cylinders[cylinder_index2], side2)
    if np.abs(np.dot(cy1_dir,cy2_dir))>0.8: # 36度
        return True
    else:
        return False


def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def get_neighbors(top_bottom, given_vector):
    # 存储距离和对应下标列表
    distances = []
    for i, tup in enumerate(top_bottom):
        for j, vec in enumerate(tup):
            distance = euclidean_distance(given_vector, vec)
            distances.append((distance, (i, j)))
    distances.sort(key=lambda x: x[0])
    sorted_indices = [index for _, index in distances]
    return sorted_indices
def get_correct_dir(cylinder, side):
    if side == TOP:
        delta = np.array(cylinder.top_center) - np.array(cylinder.bottom_center)
    else:
        delta = np.array(cylinder.bottom_center) - np.array(cylinder.top_center)
    return delta / np.linalg.norm(delta)

def save_json(file_name):
    json_cylinders = []
    for instance in globals.load_cylinders:
        json_results=instance.get()
        json_data = {
            "top_center": json_results[0],
            "bottom_center": json_results[1],
            "radius": json_results[2],
            "tid":json_results[3],
            "top_id":json_results[4],
            "bottom_id":json_results[5],
            "group_id":json_results[6]
        }
        json_cylinders.append(json_data)

    json_anchors = []
    for instance in globals.load_anchors:
        json_results=instance.get()
        json_data={
            "tid":json_results[0],
            "coord":json_results[1],
            "parts":json_results[2],
            "group_id":json_results[3]
        }
        json_anchors.append(json_data)

    json_elbows=[]
    for instance in globals.save_elbows:
        json_results=instance.get()
        json_data={
            "torus_radius":json_results[0],
            "radius":json_results[1],
            "center_coord":json_results[2],
            "normal":json_results[3],
            "start_angle":json_results[4],
            "end_angle":json_results[5],
            "align_angle":json_results[6],
            "coord1":json_results[7],
            "coord2":json_results[8],
            "tid":json_results[9],
            "group_id":json_results[10],
            "p1_id":json_results[11],
            "p2_id":json_results[12]
        }
        json_elbows.append(json_data)

    json_tees=[]
    for instance in globals.save_tees:
        json_results=instance.get()
        json_data={
            "top1":json_results[0],
            "bottom1":json_results[1],
            "radius1":json_results[2],
            "top2":json_results[3],
            "bottom2":json_results[4],
            "radius2":json_results[5],
            "tid":json_results[6],
            "top1_id":json_results[7],
            "bottom1_id":json_results[8],
            "top2_id":json_results[9],
            "group_id":json_results[10]
        }
        json_tees.append(json_data)

    json_groups=[]
    for instance in globals.save_groups:
        json_results=instance.get()
        json_data={
            "tid":json_results[0],
            "parts":json_results[1],
            "anchors":json_results[2]
        }
        json_groups.append(json_data)

    json_final=[]
    json_final.append({"cylinders":json_cylinders})
    json_final.append({"elbows":json_elbows})
    json_final.append({"anchors":json_anchors})
    json_final.append({"groups":json_groups})
    json_final.append({"tees":json_tees})
    json_final.append({"crosses":{}})
    json_string = json.dumps({"cylinders":json_cylinders,"elbows":json_elbows,"anchors":json_anchors,"groups":json_groups,"tees":json_tees,"crosses":[]},cls=EnumEncoder)
    with open(file_name, 'w') as outfile:
        outfile.write(json_string)