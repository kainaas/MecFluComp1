import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from objects import *
from liquid import *
import const

class Gauss_Legendre:
    points_2 = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
    Ak_2 = np.array([1.0, 1.0])

    points_3 = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    Ak_3 = np.array([5/9, 8/9, 5/9])

    @classmethod
    def integrate(self, f, n_points_quad: int = 3) -> float:
        sum = 0.0
        tmp = []
        if n_points_quad == 2:
            for i in range(2): tmp.append(f(self.points_2[i]))
            tmp = np.array(tmp)
            sum = np.dot(self.Ak_2, tmp)
        elif n_points_quad == 3:
            for i in range(3): tmp.append(f(self.points_3[i]))
            tmp = np.array(tmp)
            sum = np.dot(self.Ak_3, tmp)
        return sum



#=================================
# Volume integrals
#=================================
def int_volume_line(l: Line, n_points_quad: int = 3) -> float:
    g = lambda xi: (l.param_func(xi)[0] * l.normal[0] + l.param_func(xi)[1] * l.normal[1]) * np.linalg.norm((l.x2 - l.x1)/2)
    return Gauss_Legendre.integrate(g, n_points_quad)

def int_volume_contour(ctr: Contour, n_points_quad: int = 3) -> float:
    sum = 0.0
    for _, l in enumerate(ctr.lines):
        sum += int_volume_line(l, n_points_quad)
    sum /= 2
    return sum


def calc_weight(ctr: Contour) -> float:
    return - int_volume_contour(ctr) * const.g * ctr.density

#=================================
# Mass center integrals
#=================================
def int_cm_line(l: Line, n_points_quad: int = 3) -> np.array:
    g1 = lambda xi: (l.param_func(xi)[0] * l.normal[0] + l.param_func(xi)[1] * l.normal[1]) * np.linalg.norm((l.x2 - l.x1)/2) * l.param_func(xi)[0]
    g2 = lambda xi: (l.param_func(xi)[0] * l.normal[0] + l.param_func(xi)[1] * l.normal[1]) * np.linalg.norm((l.x2 - l.x1)/2) * l.param_func(xi)[1]
    array = [] 
    array.append(Gauss_Legendre.integrate(g1, n_points_quad))
    array.append(Gauss_Legendre.integrate(g2, n_points_quad))
    array = np.array(array)
    return array

def int_cm_contour(ctr: Contour, n_points_quad: int = 3) -> np.array:
    sum = np.zeros((2,))
    for _, l in enumerate(ctr.lines):
        sum += int_cm_line(l, n_points_quad)
    sum /= 3*int_volume_contour(ctr, n_points_quad)
    return sum


#=================================
# buoyant force integrals
#=================================
def get_segment_under_liquid(l: Line, liquid: Liquid) -> Line:
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


def int_bouyant_line(l: Line,liquid: Liquid, n_points_quad: int = 3) -> float:
    l_under = get_segment_under_liquid(l, liquid)

    if l_under == None: 
        return 0.0
    else:
        g = lambda xi: - liquid.get_pressure(l_under.param_func(xi)[1]) * l_under.normal[1] * np.linalg.norm((l_under.x2 - l_under.x1)/2)
        return Gauss_Legendre.integrate(g, n_points_quad)
    
def int_bouyant_contour(ctr: Contour, liquid: Liquid, n_points_quad: int = 3) -> float:
    sum = 0.0
    for l in ctr.lines:
        sum += int_bouyant_line(l, liquid, n_points_quad)
    return sum

#=================================
# torque integrals
#=================================

def int_torque_line(l: Line, liquid: Liquid, cm: np.array, n_points_quad: int = 3):
    l_under = get_segment_under_liquid(l, liquid)
    if l_under == None: 
        return 0.0
    else:
        g = lambda xi: - liquid.get_pressure(l_under.param_func(xi)[1]) *  np.linalg.norm((l_under.x2 - l_under.x1)/2) * ((l_under.param_func(xi)[0] - cm[0]) * l_under.normal[1] - (l_under.param_func(xi)[1] - cm[1]) * l_under.normal[0])
        return Gauss_Legendre.integrate(g, n_points_quad)

def int_torque_contour(ctr: Contour, liquid: Liquid, cm: np.ndarray | None = None, n_points_quad: int = 3) -> float:
    if cm is None:
        cm = int_cm_contour(ctr, n_points_quad)
    sum = 0.0
    for l in ctr.lines:
        sum += int_torque_line(l, liquid, cm, n_points_quad)
    return sum

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

