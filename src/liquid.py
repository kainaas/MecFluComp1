import numpy as np

import const

class Liquid:
    '''
        A Liquid has a density and a y coordinate of its interface.
    '''
    def __init__(self, density: float = 1.0, y_interface: float = 0.0, is_immersed_3D: bool = True):
        self.density = density
        self.y_inter = y_interface
        self.immersed_3D = is_immersed_3D

    def get_pressure(self, y: float) ->float:
        '''
            Gets the pressure in determined height
        '''
        if y > self.y_inter:
            return 0.0
        else:
            return self.density * const.g * (self.y_inter - y)

    def get_pressure_vec(self, y: np.array):
        '''
            Gets the pressure in a vector of y coordinates. It must have size (1, num_points)
        '''
        mask = y > self.y_inter
        return self.density * const.g * np.where(y > self.y_inter, 0.0, self.y_inter - y)