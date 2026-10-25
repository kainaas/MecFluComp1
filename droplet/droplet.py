from sys import path

path.append("./../src")

from solve_PVI import *
import const
from objects import Circle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import itertools


class Droplet:
    def __init__(self, 
        r:np.array, 
        z:np.array, 
        V:float, 
        S:float,
        a: float,
        adimensional:bool = True
    ):
        self.r = r
        self.z = z
        self.V = V
        self.S = S
        self.a = a
        self.adimensional = adimensional

    def get_plot_vecs(self):
        x_plot = np.hstack([np.flip(-self.r), self.r])
        z_plot = np.hstack([np.flip(self.z), self.z])
        z_plot = z_plot - self.get_height()
        return x_plot, z_plot

    def plot(self, ax:plt.Axes, label: str|None = None):
        x_plot, z_plot = self.get_plot_vecs()
        if label is None:
            ax.plot(x_plot, z_plot)
        else:
            ax.plot(x_plot, z_plot, label=label)

    def dimensionalize(self):
        if self.adimensional:
            self.r = self.a * self.r
            self.z = self.a * self.z
            self.V = self.V * self.a**3
            self.S = self.S * self.a**2
            self.adimensional = False
        else:
            print("Already dimensionalized")

    def adimensionalize(self):
        if not self.adimensional:
            self.r = self.r / self.a
            self.z = self.z / self.a
            self.V = self.V / self.a**3
            self.S = self.S / self.a**2
            self.adimensional = True
        else:
            print("Already adimensionalized")
    
    def change_a(self, new_a:float):
        if new_a <= 0:
            raise ValueError("The nozzle radius must be positive.")
        if not self.adimensional:
            self.adimensionalize()
            self.a = new_a
            self.dimensionalize()
        else:
            self.a = new_a


    def get_height(self) -> float:
        return self.z[-1]



'''
u = r   =   u[0]
    z       u[1]
    theta   u[2]
    V       u[3]
    S       u[4]
'''

'''
f(u,s) = cos(theta)                                     =   cos(u[2])
         sin(theta)                                         sin(u[2])
         (p0 - density * g  * z) / gamma - sin(theta)/r     (p0 - density * g * u[1]) / gamma - sin(u[2])/u[0]
         pi * r * r * sin(theta)                            pi * u[0] * u[0] * sin(u[2])
         2 * pi * r                                         2 * pi * u[0]
'''

def f_aux_droplet(
    u: np.array,    #Last vector
    s: float,       #
    gamma: float,   #Superficial tension
    p0: float,      #Initial pressure
    density: float  #Liquid density
) -> np.array:
    udiv: float
    if np.abs(u[0]) < const.EPSILON: udiv = const.EPSILON
    else: udiv = u[0]
    return np.array([
        np.cos(u[2]),
        np.sin(u[2]),
        (p0 - density * const.g * u[1]) / gamma - np.sin(u[2])/udiv,
        np.pi * u[0] * u[0] * np.sin(u[2]),
        2 * np.pi * u[0]
    ])

'''
A = a * p0 / gamma
B = rho * g * a² / gamma
'''

def f_aux_droplet_adimensionalized(
    u: np.array,    #Last vector
    s: float,       #
    A: float,
    B: float
) -> np.array:
    return np.array([
        np.cos(u[2]),
        np.sin(u[2]),
        A - B * u[1] - np.sin(u[2])/u[0],
        np.pi * u[0] * u[0] * np.sin(u[2]),
        2 * np.pi * u[0]
    ])

def initialize_droplet(
    A: float,
    B: float,
    eps: float = 1.0e-7
) -> np.array:
    return np.array([
        eps - (A*A/24.0)* eps**3, #r
        (A/4.0)*eps*eps, #z
        (A/2.0)*eps - (A*B/16.0)* eps**3, #theta
        (np.pi * A /8.0)*eps**4, #V
        np.pi*eps*eps
    ])



def find_args_intersection(result: Result, a: float) -> list:
    '''
    Find the arguments of the result such that it has the same radius as a.
    The returned argument i is such that u[0, i] <= a <= u[0,i+1]
    '''
    tmp = np.vstack([result.u[0,:-1], result.u[0,1:]])
    args = np.argwhere(((tmp[0,:] <= a) & (tmp[1,:] > a)) | ((tmp[0,:] >= a) & (tmp[1,:] < a)) | (tmp[0,:] == a) & (tmp[1,:] == a))
    return list(itertools.chain.from_iterable(args))


def find_droplets(
    A: float,
    B: float,
    a: float,
    max_s: float,
    ds: float
) -> list[Droplet]:
    f = lambda u,s: f_aux_droplet_adimensionalized(u,s, A, B)
    eps = 1.0e-7
    u0 = initialize_droplet(A, B, eps)
    result = RK4(eps, max_s, ds, u0, f)
    args = find_args_intersection(result, 1.0)
    
    droplets = []
    if len(args) == 0:
        return droplets
    for arg in args:
        r = result.u[0,:arg]
        z = result.u[1,:arg]
        V = result.u[3, arg]
        S = result.u[4, arg]
        if arg < len(result.u[0,:]) - 1:
            u_arg = result.u[:, arg]
            u_rarg = result.u[:, arg+1]
            if np.abs(u_arg[0] - u_rarg[0]) < const.EPSILON:
                d = Droplet(r, z, V, S, a)
                droplets.append(d)
            else:
                xi = (1.0 - u_rarg[0]) / (u_arg[0] - u_rarg[0])
                u_interpol = xi * u_arg + (1.0 - xi) * u_rarg
                r = np.hstack([r, u_interpol[0]])
                z = np.hstack([z, u_interpol[1]])
                V = u_interpol[3]
                S = u_interpol[4]
                d = Droplet(r, z, V, S, a)
                droplets.append(d)
        else: print("Run with bigger max_s. Intersection on the edge of the solution vector")
    return droplets
        
    
def max_V_droplet(
    A: float,
    B: float,
    a: float,
    max_s: float,
    ds: float
) -> Droplet:
    ld = find_droplets(A, B, a, max_s, ds)
    max_V_droplet: Droplet = None
    for d in ld:
        if (max_V_droplet is None) or max_V_droplet.V < d.V:
            max_V_droplet = d
    return max_V_droplet


def min_V_droplet(
    A: float,
    B: float,
    a: float,
    max_s: float,
    ds: float
) -> Droplet:
    ld = find_droplets(A, B, a, max_s, ds)
    min_V_droplet: Droplet = None
    for d in ld:
        if (min_V_droplet is None) or min_V_droplet.V > d.V:
            min_V_droplet = d
    return min_V_droplet



def shooting(
    V: float,
    A0: float,
    A1: float,
    B: float,
    a: float,
    tol: float = 1.0e-10,
    max_iter: int = 100,
) -> float:
    def volume(A: float) -> float:
        droplets = find_droplets(A, B, a, np.pi * 5.0, 0.001)

        if not droplets:
            raise RuntimeError(f"No droplet found for A={A}")

        # Select the droplet whose volume is closest to the target.
        candidates = droplets[:2]
        droplet = min(candidates, key=lambda d: abs(d.V - V))
        return droplet.V

    V0 = volume(A0)
    V1 = volume(A1)

    for _ in range(max_iter):
        F0 = V0 - V
        F1 = V1 - V

        # Stop when the volume residual is sufficiently small.
        if abs(F1) <= tol:
            return A1

        denominator = V1 - V0

        if abs(denominator) <= const.EPSILON:
            raise RuntimeError(
                "Secant method failed: consecutive volume residuals "
                "are too close."
            )

        A_next = A1 - F1 * (A1 - A0) / denominator

        if abs(A_next - A1) <= tol:
            return A_next

        A0, V0 = A1, V1
        A1 = A_next
        V1 = volume(A1)

    raise RuntimeError(
        f"Shooting method did not converge after {max_iter} iterations."
    )



'''
Max volume droplet:
    A = 1.19721119513
    V* = 35.39486160126

Max pressure droplet:
    A* = 2.04614459832
    V = 2.17221828749
'''


if __name__ == "__main__":
    gamma = 0.0728      #N/m
    density = 1000.0    #kg/m³
    a = 0.001           #m

    ds = 0.001
    max_s = 5 * np.pi
    B = density * const.g * a * a / gamma


    plot_type = "various" #may be "Vxd", "VxA", "droplet", "droplets", "Vstar", "max_press", "no_g", "various", "anim"
    show_plot = True


    if plot_type == "Vxd":
        lA = np.linspace(0.0, 3.0, 500)
        lV = []
        ld = []
        for A in lA:
            droplets = find_droplets(A, B, a, 5*np.pi, 0.001)
            for d in droplets:
                lV.append(d.V)
                ld.append(d.get_height())
        

        fig = plt.figure(figsize = (1000, 1000, "px"))
        ax = fig.add_subplot(111)
        ax.scatter(lV, ld)
        ax.set_title(f"Volume x droplet height")
        ax.set_xlabel("Volume")
        ax.set_ylabel("d")
        
    elif plot_type == "VxA": 
        fig = plt.figure(figsize = (1000, 1000, "px"))
        ax = fig.add_subplot(111)
        ax.set_aspect("equal")

        lA = np.linspace(0.0, 3.0, 500)
        lV = []
        lA_plot = []
        for A in lA:
            droplets = find_droplets(A, B, a, 5*np.pi, 0.001)
            for d in droplets:
                lV.append(d.V)
                lA_plot.append(A)
        

        fig = plt.figure(figsize = (1000, 1000, "px"))
        ax = fig.add_subplot(111)
        ax.scatter(lV, lA_plot)
        ax.set_title(f"Volume x A")
        ax.set_xlabel("Volume")
        ax.set_ylabel("A")



    elif plot_type == "droplet":
        V = 7.824999999999999
        A = shooting(V, 1.5, 2.0, B, a)
        print(A, V)
        droplets = find_droplets(A, B, a, np.pi * 5, 0.001)


        p0 = (gamma/a) * A
        fig = plt.figure(figsize=(1000,1000, 'px'))
        ax = fig.add_subplot(111)
        for i, droplet in enumerate(droplets):
            if i > 1: break
            droplet.plot(ax, f"Intersection {i+1}, V = {droplet.V:.2f}")
            
        ax.set_title(f"Droplets A = {A}")
        ax.legend()
        ax.set_aspect("equal")




    elif plot_type == "Vstar":
        
        def V_droplet(A):
            return max_V_droplet(A, B, a, np.pi * 5.0, 0.001)
        
        tol = 1.0e-11
        
        A_min = 1.19
        A_max = 1.21
        while A_max - A_min > tol:
            print(A_min, A_max)
            A_l = A_min + (A_max - A_min) / 3
            A_r = A_max - (A_max - A_min) / 3

            V_l = V_droplet(A_l)
            V_r = V_droplet(A_r)

            if V_l.V < V_r.V:
                A_min = A_l
            else:
                A_max = A_r

        A_max_val = (A_min + A_max) / 2.0
        max_droplet= V_droplet(A_max_val)

        p0 = (gamma/a) * A_max_val
        fig = plt.figure(figsize=(1000,1000, 'px'))
        ax = fig.add_subplot(111)
        max_droplet.plot(ax)
        ax.set_title("Max volume droplet")
        ax.set_aspect("equal")
        ax.text(
            0.5, 0.5,
            f"p_0 ={p0}\n A={A_max_val}\nV*={max_droplet.V}",
            transform=fig.transFigure,
            fontsize=14, ha='center')


    elif plot_type == "max_press":

        def V_droplet(A):
            return min_V_droplet(A, B, a, np.pi * 5.0, 0.001)
        tol = 1.0e-11
        
        A_min = 2.0
        A_max = 2.2
        while A_max - A_min > tol:
            print(A_min, A_max)
            A_m = (A_max + A_min) / 2

            V_m = V_droplet(A_m)

            if V_m.V > 5.0:
                A_max = A_m
            else:
                A_min = A_m

        A_max_val = A_min
        max_droplet = V_droplet(A_max_val)

        p0 = (gamma/a) * A_max_val
        fig = plt.figure(figsize=(1000,1000, 'px'))
        ax = fig.add_subplot(111)
        max_droplet.plot(ax)
        ax.set_title("Max pressure droplet")
        ax.set_aspect("equal")
        ax.text(
            0.5, 0.5,
            f"p_0 ={p0}\n A={A_max_val}\nV*={max_droplet.V}",
            transform=fig.transFigure,
            fontsize=14, ha='center')


    elif plot_type == "no_g":
        g = 0.0
        B = 0.0
        A = 1.5
        droplets = find_droplets(A, B, a, np.pi * 5, 0.001)


        p0 = (gamma/a) * A
        fig = plt.figure(figsize=(1000,1000, 'px'))
        ax = fig.add_subplot(111)
        for i, droplet in enumerate(droplets):
            if i > 1: break
            droplet.plot(ax, f"Intersection {i+1}, V = {droplet.V:.2f}")
            if i == 1:
                height = droplet.get_height()
                radius = 2.0 / A 
                height -= radius
                c = Circle(point(0.0, -height), radius)
                c.plot(ax)

            
        ax.set_title(f"Droplets A = {A}")
        ax.legend()
        ax.set_aspect("equal")

    elif plot_type == "various" or plot_type == "anim":
        max_V = 31.0
        min_V = 0.1
        num_d = 19
        lV = np.linspace(min_V, max_V, num_d)
                
        fig = plt.figure(figsize=(1000, 1000, 'px'))
        ax = fig.add_subplot(111)
        ax.set_aspect("equal")
        droplets = []
        for V in lV:
            A: float
            print("V=", V)
            if V <= 2.17221828749:
                A = shooting(V, 0.0, 2.0, B, a)
            elif V > 20.0:
                A = shooting(V, 1.2, 1.4, B, a)
            elif V > 10.0:
                A = shooting(V, 1.3, 1.6, B, a)
            elif V > 5.0:
                A = shooting(V, 1.5, 1.9, B, a)
            elif V > 2.17221828749:
                A = shooting(V, 1.2, 1.9, B, a)
            
            ld = find_droplets(A, B, a, np.pi * 5.0, 0.001)
            if len(ld) >= 2:
                ld = ld[:2]
            d = sorted(ld, key=lambda d: abs(d.V - V))[0]
            droplets.append(d)

        droplets.append(find_droplets(1.19721119513, B, a, np.pi*5.0, 0.001)[1])

        if plot_type == "various":
            for d in droplets:
                d.plot(ax, label=f"V={d.V:.3f}")
            ax.legend()



        if plot_type == "anim":
            save = "imgs/b.gif"

            duration = 2.0 #in seconds
            interval = int(duration/len(droplets) * 1000) #in ms

            ax.set_xlim(-2.0, 2.0)
            ax.set_ylim(-5.5, 0.1)

            line, = ax.plot([], [], lw=2)
            line.set_color("black")


            def init():
                line.set_data([], [])
                return (line,)


            def update(frame):
                x, y = droplets[frame].get_plot_vecs()
                line.set_data(x, y)

                return (line,)



            anim = FuncAnimation(
                fig,
                update,
                frames=len(droplets),
                init_func=init,
                interval=interval,
                blit=True
            )
            if not (save is None):
                anim.save(save)

    if show_plot:
        plt.show()
