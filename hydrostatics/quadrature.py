import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import objects as obj

class Gauss_Legendre:
    points_2 = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
    Ak_2 = np.array([1.0, 1.0])

    points_3 = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    Ak_3 = np.array([5/9, 8/9, 5/9])

    @classmethod
    def integrate(self, f: callable[float, float], n_points: int = 3) -> float:
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


def int_volume_line(l: obj.Line, n_points: int = 3) -> float:
    g = lambda xi: (l.param_func(xi)[0] * l.normal[0] + l.param_func(xi)[1] * l.normal[1]) * np.linalg.norm((l.x2 - l.x1)/2)
    return Gauss_Legendre.integrate(g, n_points)

def int_volume_contour(ctr: obj.Contour, n_points: int = 3) -> float:
    sum = 0.0
    for _, l in enumerate(ctr.lines):
        sum += int_volume_line(l, n_points)
    sum /= 2
    return sum


if __name__ == "__main__":
    p1 = obj.point(4.0, 2.0)
    p2 = obj.point(-4.0, 2.0)
    p3 = obj.point(-6.0, 0.0)
    p4 = obj.point(-4.0, -2.0)
    p5 = obj.point(4.0, -2.0)
    p6 = obj.point(6.0, 0.0)

    line1 = obj.Line(p1, p2)
    arc2 = obj.Arc.from_3_points(p2, p3, p4)
    line3 = obj.Line(p4, p5)
    arc4 = obj.Arc.from_3_points(p5, p6, p1)

    ctr = obj.Contour()

    ctr.add_component(line1)
    ctr.add_component(arc2)
    ctr.add_component(line3)
    ctr.add_component(arc4)

    ctr.discretize_n_lines(1000)

    volume = int_volume_contour(ctr)

    print(volume)