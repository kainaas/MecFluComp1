from sys import path

path.append("./../src")

from solve_PVI import *
import const

import numpy as np
import matplotlib.pyplot as plt
import itertools

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
):
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


def find_args_intersection(result: Result, a: float):
    tmp = np.vstack([result.u[0,:-1], result.u[0,1:]])
    args = np.argwhere(((tmp[0,:] <= a) & (tmp[1,:] > a)) | ((tmp[0,:] >= a) & (tmp[1,:] < a)))
    return list(itertools.chain.from_iterable(args))

def find_arg_intersection(result: Result, a: float):
    args = find_args_intersection(result, a)
    if len(args) <= 3 and len(args) > 0:
        return args[-1]
    elif len(args) == 0:
        print("Did not find any intersection")
        return None
    else:
        print("droplet is not viable")
        return None
    
def vec_for_plots(x, z):
    x_plot = np.hstack([np.flip(-x), x])
    z_plot = np.hstack([np.flip(z), z])
    return x_plot, z_plot


if __name__ == "__main__":
    gamma = 0.0728      #N/m
    density = 1000.0    #kg/m³
    a = 0.001           #m

    ds = 0.000001
    u0 = np.array([
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    ])
    make_Vxd = False
    show_plot = True

    if make_Vxd:
        lp0 = np.linspace(1.0, 120.0, 120 * 1)           #N/m²
        lp0 = [85.0]
        lV = []
        ld = []
        last_d = 0.0
        last_result: Result
        last_p0 = 1.0
        first_not_func = False
        p0_not_func: float = None
        result_not_func: Result = None
        for p0 in lp0:
            f = lambda u, s: f_aux_droplet(u, s, gamma, p0, density)

            result = RK4(0.0, 4.0*np.pi*a, ds, u0, f)
            # print(find_args_intersection(result, a))
            arg = find_arg_intersection(result, a)
            if not arg is None:
                d = result.u[1, arg]
                V = result.u[3, arg]
                lV.append(V)
                ld.append(d)
                if last_d > d and not first_not_func:
                    first_not_func = True
                    p0_not_func = last_p0
                    result_not_func = last_result
                    print("not function anymore. p0 =", last_p0)
                last_d = d
                last_p0 = p0
                last_result = result
            else:
                print(p0)
        
        

        fig = plt.figure(figsize = (2000, 1000, "px"))
        ax = fig.add_subplot(121)
        ax.scatter(lV, ld)
        ax.set_title(f"Volume x d")
        ax.set_xlabel("Volume")
        ax.set_ylabel("d")

        ax_drop = fig.add_subplot(122)
        ax_drop.set_aspect("equal")
        if not p0_not_func is None:
            ax_drop.set_title(rf"Maximun volume droplet $p_0 = {p0_not_func:.4f}$")
        
        x_plot ,y_plot = vec_for_plots(result.u[0,:], result.u[1,:])
        ax_drop.plot(x_plot, y_plot)
        ax_drop.plot([a, a], [0.0, np.max(y_plot)], 'k--')
        ax_drop.plot([-a, -a], [0.0, np.max(y_plot)], 'k--')
        
    else: 
        fig = plt.figure(figsize = (1000, 1000, "px"))
        ax = fig.add_subplot(111)
        ax.set_aspect("equal")
        
        
        max_d = 0.0
        lp0 = [1.0, 50.0, 85.0, 87.0125, 87.1, 87.5, 88.0, 90.0]
        for p0 in lp0:
            f = lambda u, s: f_aux_droplet(u, s, gamma, p0, density)

            result = RK4(0.0, 3.0*np.pi*a, ds, u0, f)
            # print(find_args_intersection(result, a))
            arg = find_arg_intersection(result, a)
            d = result.u[1, arg]
            if d > max_d:
                max_d = d
            z_plot = result.u[1, :arg] - d
            x_plot = result.u[0, :arg]
            x_plot, z_plot = vec_for_plots(x_plot, z_plot)
            ax.plot(x_plot, z_plot, label=rf"$p_0 = {p0}$")
        ax.plot([a, a], [0.0, -max_d], 'k--')
        ax.plot([-a, -a], [0.0, -max_d], 'k--')
        ax.legend()

    if show_plot:
        plt.show()
