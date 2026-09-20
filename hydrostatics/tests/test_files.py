import sys
sys.path.append("./..")

from objects import *
from quadrature import *
from liquid import *
from solve import *
from solve_PVI import *
from visualize import *
import const

import numpy as np
import matplotlib.pyplot as plt


obj_name = "lmacc"
density = 0.5
''' can be PVI, equilibrium or contour levels '''
what_to_do = "contour levels"
show_plot = True
water = Liquid()


obj: Contour
if obj_name == "manufactured_sol":
    obj = Contour.read_point_list("../objects/manufactured_sol.txt", density, False)
else:
    obj = Contour.read_file(f"../objects/{obj_name}.txt", density)
obj.discretize_n_lines(1000)

volume = int_volume_optm(obj)
cm = int_cm_optm(obj)

print("volume: ", volume)
print("mass center: ", cm)


if obj_name == "manufactured_sol":
    #Fully submerged
    obj.translate(0.0, -2.0, True)
    BF1 = int_bouyant_optm(obj, water)
    print("Bouyant force (fully submerged): ", BF1)
    #Partially submerged
    obj.translate(0.0, 1.0, True)
    BF2 = int_bouyant_optm(obj, water)
    print("Bouyant force (partially submerged): ", BF2)



move_cm_to_origin(obj, reset_pos=True)



if what_to_do == "PVI":
    gamma = 0.1
    u0 = np.array([
        6.0, #b2
        0.0, #y-speed
        0.0, #theta
        5.0 #angular speed
    ])

    t0 = 0.0
    t_final = 20.0
    h = 0.01
    
    volume = int_volume_optm(obj)
    mass = volume * obj.density
    weight = calc_weight_optm(obj)
    I = int_MI_optm(obj)

    f = lambda u, t: f_aux(u, t, obj, water, mass, weight, volume, I, gamma)

    result_e = euler_explicito(t0, t_final, h, u0, f)
    result_RK4 = RK4(t0, t_final, h, u0, f)

    fig_plot = plt.figure()
    ax1 = fig_plot.add_subplot(121)
    ax2 = fig_plot.add_subplot(122)

    ax1.plot(result_e.t, result_e.u[0], label=r"$b_2$")
    ax1.plot(result_e.t, result_e.u[2], label=r"$\theta$")
    ax1.set_title("Explicit Euler")
    ax2.plot(result_RK4.t, result_RK4.u[0], label=r"$b_2$")
    ax2.plot(result_RK4.t, result_RK4.u[2], label=r"$\theta$")
    ax2.set_title("RK4")

    ax1.legend()
    ax2.legend()
    make_gif(obj, water, result_RK4, save=f"../gifs/{obj_name}_animation.gif")


if what_to_do == "equilibrium":
    theta0 = - np.pi / 2
    b0 = 0.0
    db2 = 1e-4
    dtheta = 1e-4
    tol = 1e-11

    solution = Newton_FDM(obj, water, b0, theta0, dtheta, db2, tol, verbose=True)

    weight = calc_weight_optm(obj)
    BF = int_bouyant_optm(obj, water)
    torque = int_torque_optm(obj, water)
    I = int_MI_optm(obj)

    print("")
    print("Solution: theta = ", solution[0], "; b2 = ", solution[1])
    print("Weight: ", weight)
    print("Bouyant force: ", BF)
    print("Torque: ", torque)
    print("Moment of inertia: ", I)

    fig = plt.figure(figsize=(20,10), dpi=100)

    L = obj.find_biggest_dist()

    ax2 = fig.add_subplot(122)
    ax2.set_xlim(-L, L)
    ax2.set_ylim(-L, L)
    ax2.grid(True)
    obj.plot_discretized(ax2)
    ax2.scatter(0.0, solution[1])
    ax2.fill_between(
        [-2 * L, 2* L],
        -2* L,
        water.y_inter,
        color="blue",
        alpha=0.2
    )

    ax = fig.add_subplot(121)
    ax.set_xlim(cm[0] - L, cm[0] + L)
    ax.set_ylim(cm[1] - L, cm[1] + L)
    ax.grid(True)
    ax.scatter(cm[0], cm[1])
    rigid_transformation(obj, 0.0, cm[1])
    obj.translate(cm[0], 0.0)
    obj.plot_discretized(ax)

if what_to_do == "contour levels":
    beautiful_plot(obj, water, save=f"../imgs/{obj_name}_levels.pdf")

if show_plot:
    plt.show()