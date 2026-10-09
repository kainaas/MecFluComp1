from objects import *
from quadrature import *
import numpy as np

def M_translation(x_step: float, y_step: float):
    '''
        Creates a transformation of translation
    '''
    return np.array([
        [1.0, 0.0, x_step],
        [0.0, 1.0, y_step],
        [0.0, 0.0, 1.0]
    ])

def M_rotation(theta: float):
    '''
        Creates a transformation of rotation
    '''
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0]
    ])


def move_cm_to_origin(ctr: Contour, cm: np.ndarray | None = None, reset_pos: bool = False, abstraction: bool = False):
    '''
        Moves the centr of mass to the origin. The current cm can be inputed, sparing some calculations.
        If reset_pos is true, this sets the new reference position for the object.
        If abstraction is true, also moves the abstraction.
    '''
    if cm is None:
        cm = int_cm_optm(ctr)
    ctr.translate(-cm[0], -cm[1], abstraction)
    if reset_pos:
        ctr.reset_position()



def rigid_transformation(
    ctr: Contour,
    theta: float,
    b2: float,
    cm: np.ndarray | None = None,
):
    '''
        Moves the object to y=b2 and theta = theta.
        center of mass can be given to spare calculations.
        
    '''
    if cm is None:
        cm = int_cm_optm(ctr)

    # Rotation relative to current orientation
    dtheta = theta - ctr.theta

    # Transformation:
    # 1. CM -> origin
    # 2. rotate
    # 3. origin -> desired CM position
    M = (
        M_translation(cm[0], b2)
        @ M_rotation(dtheta)
        @ M_translation(-cm[0], -cm[1])
    )

    ctr.apply_RT(M, M_rotation(dtheta), 0.0, b2 - cm[1], dtheta)
