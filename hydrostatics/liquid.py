import numpy as np

import const

class Liquid:
    def __init__(self, density: float = 1.0, y_interface: float = 0.0, is_immersed_3D: bool = True):
        self.density = density
        self.y_inter = y_interface
        self.immersed_3D = is_immersed_3D

    def get_pressure(self, y: float) ->float:
        if y > self.y_inter:
            return 0.0
        else:
            return self.density * const.g * (self.y_inter - y)