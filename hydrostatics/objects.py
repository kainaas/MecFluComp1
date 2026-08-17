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

    def translate(self, x_step: float, y_step: float):
        p = point(x_step, y_step)
        self.x1 = self.x1 + p
        self.x2 = self.x2 + p

    def rotate(self, matrix):
        self.x1 = matrix @ self.x1
        self.x2 = matrix @ self.x2
        self.normal = matrix @ self.normal

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
    def __init__(self, center: np.array, radius: float, counter_clockwise: bool = True):
        self.c = center
        self.r = radius
        self.lines = []
        self.counter_clockwise = counter_clockwise

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

    def translate(self, x_step: float, y_step: float):
            p = point(x_step, y_step)
            self.c = self.c + p
            if len(self.lines) > 0:
                for l in self.lines:
                    l.translate(x_step, y_step)

    def rotate(self, matrix):
            self.c = matrix @ self.c
            if len(self.lines) > 0:
                for l in self.lines:
                    l.rotate(matrix)

    def plot(self, ax: matplotlib.axes, n_line_segments: int = 100):
        points = self.get_points_eq_space(n_line_segments)
        ax.plot(points[:,0], points[:,1])

    def plot_discretized(self, ax: matplotlib.axes, normal: bool = False):
        for i, l in enumerate(self.lines):
            l.plot(ax, normal)
    




'''Specialization of a Circle'''
class Arc(Circle):
    def __init__(self, x_start: np.array, x_end: np.array, center: np.array, counter_clockwise_normal: bool, counter_clockwise: bool = True):
        self.x1 = x_start
        self.x2 = x_end
        self.counter_clockwise_draw = counter_clockwise
        r = np.linalg.norm(center - x_start)
        super().__init__(center, r, counter_clockwise_normal)

    @classmethod
    def from_3_points(cls, x_start: np.array, x_middle: np.array, x_end: np.array, counter_clockwise_normal: bool = True):
        
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

        return cls(x_start, x_end, center, counter_clockwise_normal, counter)
    

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

        if self.counter_clockwise_draw:
            theta += theta1
        else:
            theta = theta1 - theta

        return np.column_stack([
            self.c[0] + self.r*np.cos(theta),
            self.c[1] + self.r*np.sin(theta)
        ])


    def translate(self, x_step: float, y_step: float):
        p = point(x_step, y_step)
        self.x1 = self.x1 + p
        self.x2 = self.x2 + p
        self.c = self.c + p
        if len(self.lines) > 0:
            for l in self.lines:
                l.translate(x_step, y_step)

    def rotate(self, matrix):
        self.c = matrix @ self.c
        self.x1 = matrix @ self.x1
        self.x2 = matrix @ self.x2
        if len(self.lines) > 0:
            for l in self.lines:
                l.rotate(matrix)

        
        

'''A set of lines and arcs'''
class Contour:
    def __init__(self):
        self.components = []
        self.lines = []

    def add_component(self, component):
        self.components.append(component)

    def discretize_n_lines(self, n_segments: int):
        for _, comp in enumerate(self.components):
            if isinstance(comp, Circle):
                comp.discretize(n_segments, comp.counter_clockwise)
                for _, l in enumerate(comp.lines):
                    self.lines.append(l)

            else: self.lines.append(comp)

    def discretize_size_lines(self, max_size: float):
        for _, comp in enumerate(self.components):
            if isinstance(comp, Circle):
                arc_length = comp.get_arc_length()
                n_segments = arc_length / max_size
                if n_segments.is_integer():
                    n_segments = int(n_segments)
                else:    
                    n_segments = int(n_segments+1)
                comp.discretize(n_segments, comp.counter_clockwise)
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

    def translate(self, x_step: float, y_step: float):
        for comp in self.components:
            comp.translate(x_step, y_step)

    def rotate(self, theta: float):
        s = np.sin(theta)
        c = np.cos(theta)
        mat = np.array([[c, -s], [s, c]])
        for comp in self.components:
            comp.rotate(mat)


    @classmethod
    def read_file(cls, file):
        ctr = cls()
        with open(file, 'r') as f:
            for line in f:
                counter_clockwise = 1
                l = line.split()
                if len(l) == 0:
                    pass
                elif l[0] == "#":
                    pass
                elif l[0] == "L":
                    x1, y1, x2, y2 = float(l[1]), float(l[2]), float(l[3]), float(l[4])
                    start = point(x1, y1)
                    end = point(x2, y2)
                    if len(l) > 5:
                        counter_clockwise = l[5] == "1"
                    ctr.add_component(Line(start, end, counter_clockwise))

                elif l[0] == "C":
                    c1, c2 = float(l[1]), float(l[2])
                    r = float(l[3])
                    if len(l) > 4:
                        counter_clockwise = l[4] == "1"
                    center = point(c1,c2)
                    ctr.add_component(Circle(center, r, counter_clockwise))

                elif l[0] == "Ac":
                    x1, y1, x2, y2, c1, c2 = float(l[1]), float(l[2]), float(l[3]), float(l[4]), float(l[5]), float(l[6])
                    if len(l) > 7:
                        counter_clockwise = l[7] == "1"
                    counter_clockwise_draw = True
                    if len(l) > 8:
                        counter_clockwise_draw = l[8] == "1"
                    start = point(x1, y1)
                    end = point(x2, y2)
                    center =point(c1, c2)
                    ctr.add_component(Arc(start, end, center, counter_clockwise, counter_clockwise_draw))

                elif l[0] == "A3":
                    x1, y1, x2, y2, x3, y3 = float(l[1]), float(l[2]), float(l[3]), float(l[4]), float(l[5]), float(l[6])
                    start = point(x1, y1)
                    mid = point(x2, y2)
                    end = point(x3, y3)
                    ctr.add_component(Arc.from_3_points(start, mid, end))
        return ctr






if __name__ == "__main__":
    obj = Contour.read_file("test.txt")


    fig = plt.figure(figsize=(10,10), dpi=100)
    ax = fig.add_subplot(111)

    obj.discretize_n_lines(10)

    obj.rotate(np.pi/2)
    obj.translate(1.0, 1.0)
    d = True
    if d:
        obj.plot_discretized(ax, True)
    else:
        obj.plot(ax)

    ax.set_ybound(-8, 8)
    ax.set_xbound(-8, 8)
    plt.show()
