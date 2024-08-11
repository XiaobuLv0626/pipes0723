import numpy as np
class Anchor:
    def __init__(self, tid, coord, parts, group_id):
        self.tid = tid
        self.coord = coord
        self.parts = parts
        self.group_id = group_id

    @classmethod
    def load_anchors_from_json(cls, json_file: str) -> 'list[Anchor]':
        import json
        with open(json_file, 'r') as f:
            data = json.load(f)
        anchors = []
        if isinstance(data, dict) and 'anchors' in data:
            for item in data['anchors']:
                anchor = cls(item['tid'], item['coord'], item['parts'], item['group_id'])
                anchors.append(anchor)
        return anchors

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
