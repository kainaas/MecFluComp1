import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def cross(x1: np.array, x2: np.array):
    return x1[0]*x2[1] - x1[1]*x2[0]

def point(x1: float, x2: float) -> np.array:
    return np.array([x1, x2])


def angle_origin_vector(x: np.ndarray) -> float:
    theta = np.arctan2(x[1], x[0])
    if theta < 0:
        theta += 2*np.pi
    return theta


'''Represents a line in 2D. x1 and x2 represents the begin and end points of the line. The normal vector is considering that objects are ran counter-clockwise. The points list stores the points used in quadratures.'''
class Line:
    def __init__(self, x1: np.array, x2: np.array, normal_counter_clock: bool = True):
        self.x1 = x1
        self.x2 = x2
        self.vec = x2 - x1
        normal = np.array([self.vec[1],
                                -self.vec[0]
                                ])
        self.normal = normal/ np.linalg.norm(normal)
        if normal_counter_clock == False:
            self.invert_normal()

    def invert_normal(self): #When it's needed to invert the normal, i.e., points inside object. If the object is not simply conected
        for i in range(2):
            self.normal[i] = -self.normal[i]

    def plot(self, ax: matplotlib.axes, normal: bool=False):
        ax.plot([self.x1[0], self.x2[0]], [self.x1[1], self.x2[1]])
        if normal:
            midpoint = (self.x1 + self.x2)/2
            ax.quiver(midpoint[0], midpoint[1], self.normal[0], self.normal[1])

    #Parametrizes the line to the interval [-1, 1]
    def param_func(self, xi: float) -> np.array:
        return (1- xi)*self.x1 / 2 + (1 + xi)*self.x2 / 2






'''Represents a full circle in 2D. Has a lines list to store a discretized/segmented circle'''
class Circle:
    def __init__(self, center: np.array, radius: float):
        self.c = center
        self.r = radius
        self.lines = []

    def get_points_eq_space(self, n_segments: int):
        pace = 2*np.pi / (n_segments)
        return np.array([[self.c[0] + self.r*np.cos(i*pace), self.c[1] + self.r*np.sin(i*pace)] for i in range(n_segments+1)])

    def discretize(self, n_segments: int, counter_clockwise: bool = True):
        self.lines = []
        points = self.get_points_eq_space(n_segments)
        for i in range(n_segments):
            l = Line(points[i,:], points[i+1,:], counter_clockwise)
            self.lines.append(l)

    def get_angular_size(self):
        return 2* np.pi

    def get_arc_length(self):
        return self.r * self.get_angular_size()

    def plot(self, ax: matplotlib.axes, n_line_segments: int = 100):
        points = self.get_points_eq_space(n_line_segments)
        ax.plot(points[:,0], points[:,1])

    def plot_discretized(self, ax: matplotlib.axes, normal: bool = False):
        for i, l in enumerate(self.lines):
            l.plot(ax, normal)
    




'''Specialization of a Circle'''
class Arc(Circle):
    def __init__(self, x_start: np.array, x_end: np.array, center: np.array, counter_clockwise: bool):
        self.x1 = x_start
        self.x2 = x_end
        self.counter_clockwise = counter_clockwise
        r = np.linalg.norm(center - x_start)
        super().__init__(center, r)

    @classmethod
    def from_3_points(cls, x_start: np.array, x_middle: np.array, x_end: np.array):
        
        v1 = x_middle - x_start
        v2 = x_end - x_start
        cross_prod = cross(v1, v2)
        counter: bool = True
        if(np.abs(cross_prod) <= 1e-12): return None
        elif cross_prod < 0.0: counter = False
        
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

        return cls(x_start, x_end, center, counter)
    

    def get_angular_size(self):
        v1 = self.x1 - self.c
        v2 = self.x2 - self.c

        theta1 = angle_origin_vector(v1)
        theta2 = angle_origin_vector(v2)

        if self.counter_clockwise:
            diff = (theta2 - theta1) % (2*np.pi)
        else:
            diff = (theta1 - theta2) % (2*np.pi)
        return diff

    def get_points_eq_space(self, n_segments: int):
        diff = self.get_angular_size()

        if diff < 1e-12: 
            super().plot(ax, n_segments)
            return

        v1 = self.x1 - self.c
        theta1 = angle_origin_vector(v1)

        theta = np.linspace(0, diff, n_segments + 1)

        if self.counter_clockwise:
            theta += theta1
        else:
            theta = theta1 - theta

        return np.column_stack([
            self.c[0] + self.r*np.cos(theta),
            self.c[1] + self.r*np.sin(theta)
        ])

        
        

'''A set of lines and arcs'''
class Contour:
    def __init__(self):
        self.components = []
        self.lines = []

    def add_component(self, component):
        self.components.append(component)

    def discretize_n_lines(self, n_segments: int, counter_clockwise: bool = True):
        for _, comp in enumerate(self.components):
            if isinstance(comp, Circle):
                comp.discretize(n_segments, counter_clockwise)
                for _, l in enumerate(comp.lines):
                    self.lines.append(l)

            else: self.lines.append(comp)

    def discretize_size_lines(self, max_size: float, counter_clockwise: bool = True):
        for _, comp in enumerate(self.components):
            if isinstance(comp, Circle):
                arc_length = comp.get_arc_length()
                n_segments = arc_length / max_size
                if n_segments.is_integer():
                    n_segments = int(n_segments)
                else:    
                    n_segments = int(n_segments+1)
                comp.discretize(n_segments, counter_clockwise)
                for _, l in enumerate(comp.lines):
                    self.lines.append(l)

            else: self.lines.append(comp)

    def plot(self, ax: matplotlib.axes, n_line_segments: int = 100, normal:bool = False):
        for _, comp in enumerate(self.components):
            if isinstance(comp, Circle):
                comp.plot(ax, n_line_segments)
            else: comp.plot(ax, normal)

    def plot_discretized(self, ax: matplotlib.axes, normal: bool = False):
        for _, l in enumerate(self.lines):
            l.plot(ax, normal)





if __name__ == "__main__":
    p1 = point(4.0, 2.0)
    p2 = point(-4.0, 2.0)
    p3 = point(-6.0, 0.0)
    p4 = point(-4.0, -2.0)
    p5 = point(4.0, -2.0)
    p6 = point(6.0, 0.0)


    circle1 = Circle(point(0.0,0.0), 1.0)
    # circle1.discretize(6, False)

    line1 = Line(p1, p2)
    arc2 = Arc.from_3_points(p2, p3, p4)
    line3 = Line(p4, p5)
    arc4 = Arc.from_3_points(p5, p6, p1)

    obj = Contour()
    obj_c = Contour()

    obj.add_component(line1)
    obj.add_component(arc2)
    obj.add_component(line3)
    obj.add_component(arc4)

    obj_c.add_component(circle1)


    
    fig = plt.figure(figsize=(1000,1000,"px"))
    ax = fig.add_subplot(111)

    obj.discretize_size_lines(10.0)
    obj_c.discretize_n_lines(6, False)

    d = True
    if d:
        obj.plot_discretized(ax, True)
        obj_c.plot_discretized(ax, True)
    else:
        obj.plot(ax)
        obj_c.plot(ax)

    ax.set_ybound(-6.5, 6.5)
    ax.set_xbound(-6.5, 6.5)
    plt.show()
