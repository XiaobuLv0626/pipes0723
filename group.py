import numpy as np

class Group:
    def __init__(self,tid,parts=None,anchors=None):
        self.tid=tid # 全局id，方便基元坐标索引
        self.parts=parts if parts is not None else []
        self.anchors=anchors if anchors is not None else []

    def get(self):
        return self.tid,self.parts,self.anchors
    def add_part(self,part_type,part_tid):
        self.parts.append((part_type,part_tid))

    def add_anchor(self,anchor_tid):
        self.anchors.append(anchor_tid)

    def get_parts(self):
        return self.parts

    def get_anchors(self):
        return self.anchors

    @classmethod
    def load_groups_from_json(cls, json_file: str) -> 'list[Anchor]':
        import json
        with open(json_file, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'groups' in data:
            data = data['groups']
        assert isinstance(data, list)
        anchors = []
        for item in data:
            anchor = cls(item['tid'], item['parts'],item['anchors']) # 导入模块未进行检验
            anchors.append(anchor)
        return anchors
