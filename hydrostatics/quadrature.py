import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from objects import *
from liquid import *
import const

class Gauss_Legendre:
    '''
        This class hlds the implementation of Gauss Legendre quadrature.
        Supports quadratures with 2 or 3 points and functions ranging from
        [-1,1]
    '''
    points_2 = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
    Ak_2 = np.array([1.0, 1.0])

    points_3 = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    Ak_3 = np.array([5/9, 8/9, 5/9])

    @classmethod
    def integrate(cls, f, n_points_quad: int = 3) -> float:
        '''
            Integrate f using n_points_quad
        '''
        sum = 0.0
        tmp = []
        if n_points_quad == 2:
            for i in range(2): tmp.append(f(cls.points_2[i]))
            tmp = np.array(tmp)
            sum = np.dot(cls.Ak_2, tmp)
        elif n_points_quad == 3:
            for i in range(3): tmp.append(f(cls.points_3[i]))
            tmp = np.array(tmp)
            sum = np.dot(cls.Ak_3, tmp)
        return sum

    @classmethod
    def integrate_vector(cls, f, n_points_quad: int = 3) ->float:
        '''
            Integrate f using n_points_quad. f must return a 1 x num_points_used array where each collumn
            is the function in a point.
        '''
        sum = 0.0
        tmp = []
        if n_points_quad == 2:
            for i in range(2): tmp.append(f(cls.points_2[i]))
            tmp = np.vstack(tmp)
            integral_array = np.dot(cls.Ak_2, tmp)
            sum = np.sum(integral_array)
        elif n_points_quad == 3:
            for i in range(3): tmp.append(f(cls.points_3[i]))
            tmp = np.vstack(tmp)
            integral_array = np.dot(cls.Ak_3, tmp)
            sum = np.sum(integral_array)
        return sum


#=================================
# Volume integrals
#=================================
def int_volume_optm(ctr: Contour, n_points_quad: int = 3):
    '''
        Calculates the volume of an object.
    '''
    g: callable
    g = lambda xi : np.array((ctr.param(xi)[0, :] * ctr.Lnormals[0, :] + ctr.param(xi)[1, :] * ctr.Lnormals[1, :]) * np.linalg.norm((ctr.Lmatrix[0:2, :] - ctr.Lmatrix[3:5, :])/2,axis=0))
    return Gauss_Legendre.integrate_vector(g, n_points_quad) / 2

def calc_weight_optm(ctr: Contour) -> float:
    '''
        Calculates the weight of an object , considering it has constant density
    '''
    return - int_volume_optm(ctr) * const.g * ctr.density




#=================================
# Mass center integrals
#=================================
def int_cm_optm(ctr: Contour, n_points_quad: int = 3) -> np.array:
    '''
        Calculates the center of mass of an object.
    '''
    g1 = lambda xi: np.array((ctr.param(xi)[0,:] * ctr.Lnormals[0,:] + ctr.param(xi)[1,:] * ctr.Lnormals[1,:]) * np.linalg.norm((ctr.Lmatrix[0:2, :] - ctr.Lmatrix[3:5, :])/2,axis=0) * ctr.param(xi)[0,:])
    g2 = lambda xi: np.array((ctr.param(xi)[0,:] * ctr.Lnormals[0,:] + ctr.param(xi)[1,:] * ctr.Lnormals[1,:]) * np.linalg.norm((ctr.Lmatrix[0:2, :] - ctr.Lmatrix[3:5, :])/2,axis=0) * ctr.param(xi)[1,:])
    array = []
    array.append(Gauss_Legendre.integrate_vector(g1, n_points_quad))
    array.append(Gauss_Legendre.integrate_vector(g2, n_points_quad))
    array = np.array(array)
    return array / (3* int_volume_optm(ctr, n_points_quad))



#=================================
# Segment lines
#=================================
def get_segment_under_liquid(l: Line, liquid: Liquid) -> Line:
    '''
        Gets the part of the line l that is submerged in the liquid
    '''
    #points with y relative to the y_interface
    x_start_rel = l.x1.copy()
    x_start_rel[1] = x_start_rel[1] - liquid.y_inter

    x_end_rel = l.x2.copy()
    x_end_rel[1] = x_end_rel[1] - liquid.y_inter

    if (np.abs(x_end_rel[1] - x_start_rel[1]) < const.EPSILON and x_end_rel[1] == 0.0): #line is tangent to the surface
        return None

    elif x_end_rel[1] > 0.0 -const.EPSILON and x_start_rel[1] > 0.0 - const.EPSILON: #The whole line above the liquid
        return None

    elif x_end_rel[1] * x_start_rel[1] < 0.0: #We must treat it separating into 2 line segments: one above and the other under the liquid
        xi_surface = (2 * liquid.y_inter - (l.x2[1] + l.x1[1])) / (l.x2[1] - l.x1[1]) #xi in the parametrization that returns the point with y = y_interface
        p_surface = l.param_func(xi_surface) #Get this point

        p0: np.array #point that is under the interface of the liquid
        if x_start_rel[1] < 0.0: p0 = l.x1
        else: p0 = l.x2

        p1: np.array
        p2: np.array
        vec = p_surface - p0
        if(l.vec.T @ vec < 0.0): #inner product negative means opposite directions
            p2 = p0
            p1 = p_surface
        else:
            p2 = p_surface
            p1 = p0
        
        return Line(p1, p2, l.normal_counter_clock)
    else: #if the whole line is under the liquid
        return l


def segment_optm(ctr: Contour, liquid: Liquid):
    '''
        Given a Contour, returns a matrix with the same structure of the Contour lines matrix
        with all the segments that are under the liquid.
    '''
    x_start_rel = ctr.Lmatrix[0:2, :] - liquid.y_inter
    x_end_rel = ctr.Lmatrix[3:5, :] - liquid.y_inter

    mask = (x_end_rel[1,:] < 0.0) | (x_start_rel[1,:] < 0.0)
    mask_total = mask & ~(x_end_rel[1,:] * x_start_rel[1,:] < 0.0)
    mask_split = mask & (x_end_rel[1,:] * x_start_rel[1,:] < 0.0)

    lines_under = ctr.Lmatrix[:, mask_total]
    lines_to_split = ctr.Lmatrix[:, mask_split]

    normals_under = ctr.Lnormals[:, mask_total]
    normals_to_split = ctr.Lnormals[:, mask_split]
    normals = np.hstack([normals_under, normals_to_split])

    n_split = lines_to_split.shape[1]

    points_surface = np.empty((2, n_split))

    xi_surface = (
        2 * liquid.y_inter
        - (lines_to_split[4, :] + lines_to_split[1, :])
    ) / (
        lines_to_split[4, :] - lines_to_split[1, :]
    )

    for i, xi in enumerate(xi_surface):
        points_surface[:, i] = (
            (1 - xi) / 2 * lines_to_split[0:2, i]
            + (1 + xi) / 2 * lines_to_split[3:5, i]
        )

    under_mask = lines_to_split[1, :] < liquid.y_inter

    points_under = np.empty((2, n_split))

    points_under[:, under_mask] = lines_to_split[0:2, under_mask]
    points_under[:, ~under_mask] = lines_to_split[3:5, ~under_mask]

    lines_splitted = np.vstack([
        points_under,
        np.ones((1, n_split)),
        points_surface,
        np.ones((1, n_split))
    ])
    lines = np.hstack([lines_under, lines_splitted])

    return lines, normals



#=================================
# buoyant force integrals
#=================================
def int_bouyant_optm(ctr: Contour, liquid: Liquid, n_points_quad: int = 3) -> float:
    '''
        Calculates the bouyant force of an object.
    '''
    lines, normals = segment_optm(ctr, liquid)
    g = lambda xi: - liquid.get_pressure_vec(param(lines, xi)[1,:]) * normals[1,:] * np.linalg.norm((lines[0:2, :] - lines[3:5, :])/2,axis=0)
    return Gauss_Legendre.integrate_vector(g, n_points_quad)






#=================================
# torque integral
#=================================
def int_torque_optm(ctr: Contour, liquid: Liquid, cm: np.ndarray | None = None, n_points_quad: int = 3) -> float:
    '''
        Calculates the torque of an object.
    '''
    if cm is None:
        cm = int_cm_optm(ctr, n_points_quad)
    lines, normals = segment_optm(ctr, liquid)
    g = lambda xi: - liquid.get_pressure_vec(param(lines, xi)[1,:]) * np.linalg.norm((lines[0:2, :] - lines[3:5, :])/2,axis=0) * ((param(lines, xi)[0,:] - cm[0]) * normals[1,:] - (param(lines, xi)[1,:] - cm[1]) * normals[0,:])
    return Gauss_Legendre.integrate_vector(g, n_points_quad)



#=================================
# Moment of inertia integral
#=================================

def int_MI_optm(ctr: Contour, cm: np.ndarray | None = None, n_points_quad: int = 3) -> float:
    '''
        Calculates the moment of inertia of an object.
    '''
    if cm is None:
        cm = int_cm_optm(ctr, n_points_quad)
    g = lambda xi: ((param(ctr.Lmatrix, xi)[0,:] - cm[0])**3 * ctr.Lnormals[0,:] + (param(ctr.Lmatrix, xi)[1,:] - cm[1])**3 * ctr.Lnormals[1,:]) * np.linalg.norm((ctr.Lmatrix[0:2, :] - ctr.Lmatrix[3:5, :])/2,axis=0)
    return (ctr.density /3) * Gauss_Legendre.integrate_vector(g, n_points_quad)

#F1 = E + W
#F2 = Tau

def F1(ctr: Contour, liquid: Liquid, weight: float) ->float:
    '''
        Returns the forces applied in the y-direction. +y upwards
    '''
    return weight + int_bouyant_optm(ctr, liquid)

def F2(ctr: Contour, liquid: Liquid, cm: np.ndarray | None = None)  -> float:
    '''
        Returns the rotational forces
    '''
    if cm is None:
        cm = int_cm_optm(ctr)
    return int_torque_optm(ctr, liquid, cm)



if __name__ == "__main__":
    p1 = point(4.0, 2.0)
    p2 = point(-4.0, 2.0)
    p3 = point(-6.0, 0.0)
    p4 = point(-4.0, -2.0)
    p5 = point(4.0, -2.0)
    p6 = point(6.0, 0.0)

    line1 = Line(p1, p2)
    arc2 = Arc.from_3_points(p2, p3, p4)
    line3 = Line(p4, p5)
    arc4 = Arc.from_3_points(p5, p6, p1)

    circle1 = Circle(point(0.0,0.0), 1.0, False)

    ctr = Contour()

    ctr.add_component(line1)
    ctr.add_component(arc2)
    ctr.add_component(line3)
    ctr.add_component(arc4)
    ctr.add_component(circle1)

    ctr.discretize_n_lines(30)

    volume = int_volume_contour(ctr)
    cm = int_cm_contour(ctr)

    print("Without circle")
    print(volume)
    print(cm)


    fig = plt.figure()
    ax = fig.add_subplot(111)

    #obj_c.plot_discretized(ax, True)
    ctr.plot_discretized(ax, True)
    plt.show()
    print("With circle")
    print(volume)
    print(cm)

