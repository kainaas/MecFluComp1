import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from objects import *

class Gauss_Legendre:
    points_2 = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
    Ak_2 = np.array([1.0, 1.0])

    points_3 = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    Ak_3 = np.array([5/9, 8/9, 5/9])

    @classmethod
    def integrate(self, f, n_points: int = 3) -> float:
        sum = 0.0
        tmp = []
        if n_points == 2:
            for i in range(2): tmp.append(f(self.points_2[i]))
            tmp = np.array(tmp)
            sum = np.dot(self.Ak_2, tmp)
        elif n_points == 3:
            for i in range(3): tmp.append(f(self.points_3[i]))
            tmp = np.array(tmp)
            sum = np.dot(self.Ak_3, tmp)
        return sum


def int_volume_line(l: Line, n_points: int = 3) -> float:
    g = lambda xi: (l.param_func(xi)[0] * l.normal[0] + l.param_func(xi)[1] * l.normal[1]) * np.linalg.norm((l.x2 - l.x1)/2)
    return Gauss_Legendre.integrate(g, n_points)

def int_volume_contour(ctr: Contour, n_points: int = 3) -> float:
    sum = 0.0
    for _, l in enumerate(ctr.lines):
        sum += int_volume_line(l, n_points)
    sum /= 2
    return sum

def int_cm_line(l: Line, n_points: int = 3) -> np.array:
    g1 = lambda xi: (l.param_func(xi)[0] * l.normal[0] + l.param_func(xi)[1] * l.normal[1]) * np.linalg.norm((l.x2 - l.x1)/2) * l.param_func(xi)[0]
    g2 = lambda xi: (l.param_func(xi)[0] * l.normal[0] + l.param_func(xi)[1] * l.normal[1]) * np.linalg.norm((l.x2 - l.x1)/2) * l.param_func(xi)[1]
    array = [] 
    array.append(Gauss_Legendre.integrate(g1, n_points))
    array.append(Gauss_Legendre.integrate(g2, n_points))
    np.array(array)
    return array

def int_cm_contour(ctr: Contour, n_points: int = 3) -> np.array:
    sum = np.zeros((2,))
    for _, l in enumerate(ctr.lines):
        sum += int_cm_line(l, n_points)
    sum /= 3*int_volume_contour(ctr, n_points)
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

    ctr = Contour()

    ctr.add_component(line1)
    ctr.add_component(arc2)
    ctr.add_component(line3)
    ctr.add_component(arc4)

    ctr.discretize_n_lines(1000)

    volume = int_volume_contour(ctr)
    cm = int_cm_contour(ctr)

    print("Without circle")
    print(volume)
    print(cm)

    circle1 = Circle(point(0.0,0.0), 1.0)
    obj_c = Contour()
    obj_c.add_component(circle1)
    obj_c.discretize_n_lines(1000, False)


    volume += int_volume_contour(obj_c)
    cm+= int_cm_contour(obj_c)
    print("With circle")
    print(volume)
    print(cm)

