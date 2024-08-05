import numpy as np
class Primitive:
    def __init__(self,p_coord1,p_coord2,p_radius,pr1_id,pr2_id):
        self.p_coord1=p_coord1
        self.p_coord2=p_coord2
        self.p_radius=p_radius
        self.pr1_id=pr1_id
        self.pr2_id=pr2_id

    def sides_get(self):
        return np.array([self.p_coord1[0], self.p_coord1[1], self.p_coord1[2], self.p_coord2[0], self.p_coord2[1], self.p_coord2[2]])