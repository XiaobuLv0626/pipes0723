import numpy as np
class Anchor:
    def __init__(self,tid,coord=None,parts=None,group_id=-1): # 不能初始化parts=[]，会导致python陷阱，后续实例化会共享parts
        self.tid=tid # 全局id，方便基元坐标索引
        '''在打组前coord并不重合，只记录一个坐标'''
        self.coord=coord
        self.parts = parts if parts is not None else []
        self.group_id=group_id

    def get(self):
        return self.tid,self.coord,self.parts,self.group_id
    def add_coord(self,coord):
        self.coord=coord

    def add_part(self,part):
        self.parts.append(part)

    def get_coord(self):
        return self.coord

    def get_parts(self):
        return self.parts

    @classmethod
    def load_anchors_from_json(cls, json_file: str) -> 'list[Anchor]':
        import json
        with open(json_file, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'anchors' in data:
            data = data['anchors']
        assert isinstance(data, list)
        anchors = []
        for item in data:
            anchor = cls(item['tid'], np.array(item['coord']),item['parts'],item['group_id'])
            anchors.append(anchor)
        return anchors
