import numpy as np

class Group:
    def __init__(self, tid, parts):
        self.tid = tid
        self.parts = parts

    @classmethod
    def load_groups_from_json(cls, json_file: str) -> 'list[Group]':
        import json
        with open(json_file, 'r') as f:
            data = json.load(f)
        groups = []
        if isinstance(data, dict) and 'groups' in data:
            for item in data['groups']:
                group = cls(item['tid'], item['parts'])
                groups.append(group)
        return groups
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

