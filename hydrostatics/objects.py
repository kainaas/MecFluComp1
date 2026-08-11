import numpy as np
import matplotlib.pyplot as plt
import matplotlib

'''Represents a line in 2D. x1 and x2 represents the begin and end points of the line. The normal vector is considering that objects are ran counter-clockwise. The points list stores the points used in quadratures.'''
class Line:
    def __init__(self, x1: np.array, x2: np.array):
        self.x1 = x1
        self.x2 = x2
        self.vec = x2 - x1
        normal = np.array([self.vec[1],
                                -self.vec[0]
                                ])
        self.normal = normal/ np.linalg.norm(normal)
        self.points = []

    @classmethod
    def plot_line(cls, ax: matplotlib.axes, l: line, normal: bool=False):
        ax.plot([l.x1[0], l.x2[0]], [l.x1[1], l.x2[1]])
        if normal:
            midpoint = (l.x1 + l.x2)/2
            ax.quiver(midpoint[0], midpoint[1], l.normal[0], l.normal[1])


class Circle:
    def __init__(self, center: np.array, radius: float):
        self.c = center
        self.r = radius

class Arc(Circle):
    def __init__(self, x_start: np.array, x_end: np.array, center: float, counter_clockwise: bool):
        self.x1 = x_start
        self.x2 = x_end
        self.counter_clockwise = counter_clockwise
        r = np.linalg.norm(center - x_start)
        super().__init__(center, r)

    @classmethod
    def from_3_points(cls, x_start: np.array, x_middle: np.array, x_end: np.array):
        x1 = x_start[0]
        x2 = x_middle[0]
        x3 = x_end[0]

        y1 = x_start[1]
        y2 = x_middle[1]
        y3 = x_end[1]
        
        M = np.array([[2*(x2 - x1), 2*(y2 - y1)],
                      [2*(x3 - x1), 2*(y3 - y1)]
                    ])
        b = np.array([x2*x2 + y2*y2 - x1*x1 - y1*y1, x3*x3 + y3*y3 - x1*x1 - y1*y1])
        
        center = np.linalg.solve(M, b)
        r = np.linalg.norm(c - x1)

        #v1 = 


if __name__ == "__main__":
    p1 = np.array([0.0,0.0])
    p2 = np.array([1.0,1.0])
    line1 = Line(p1, p2)

    p3 = np.array([-1.0,-1.0])
    p4 = np.array([-1.0, 0.0])
    line2 = Line(p3, p4)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    Line.plot_line(ax, line1, True)
    Line.plot_line(ax, line2)
    plt.show()

    arc1 = arc(p1, p4, [-0.5, -0.5], True)