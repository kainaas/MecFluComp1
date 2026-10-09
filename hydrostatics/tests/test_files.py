import sys
sys.path.append("./../../src")

from objects import *
from quadrature import *
from liquid import *
from solve import *
from solve_PVI import *
from visualize import *
import const

import numpy as np
import matplotlib.pyplot as plt

#================================
#Parameters definition
#================================
obj_name = "titanic"
file_type = "point list" #can be "point list" or "components" 
density = 400.0
what_to_do = "PVI" #can be PVI, equilibrium or contour levels
show_plot = True
water = Liquid(1000.0)

#PVI parameters
gamma1 = 404.0
gamma2 = 1.0
u0 = np.array([
    0.0, #b2
    0.0, #y-speed
    0.0, #theta
    0.0 #angular speed
])

t0 = 0.0
t_final = 40.0
h = 0.03

speed_up = 4

#Newton parameters
theta0 = np.pi
b0 = 0.0
db2 = 1e-4
dtheta = 1e-4
tol = 1e-11


#================================
#Reading object
#================================
obj: Contour
if obj_name == "manufactured_sol":
    obj = Contour.read_point_list("../objects/manufactured_sol.txt", density, False)
else:
    if file_type == "point list":
        obj = Contour.read_point_list(f"../objects/{obj_name}.txt", density)
    else:
        obj = Contour.read_file(f"../objects/{obj_name}.txt", density)
obj.discretize_n_lines(1000)




#================================
#Calculating values
#================================
volume = int_volume_optm(obj)
cm = int_cm_optm(obj)
mass = volume * obj.density
weight = calc_weight_optm(obj)
I = int_MI_optm(obj)


print("volume: ", volume)
print("mass center: ", cm)
print("Weight:", weight)
print("Mass:", mass)
print("Moment of inertia:", I)


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



#================================
#PVI
#================================
if what_to_do == "PVI":
    f = lambda u, t: f_aux_float(u, t, obj, water, mass, weight, volume, I, gamma1, gamma2)

    result_RK4 = RK4(t0, t_final, h, u0, f)
    fig_plot = plt.figure()
    ax2 = fig_plot.add_subplot(111)

    ax2.plot(result_RK4.t, result_RK4.u[0], label=r"$b_2$")
    ax2.plot(result_RK4.t, result_RK4.u[2], label=r"$\theta$")
    ax2.set_title("RK4")

    ax2.legend()
    fig_plot.savefig(f"../imgs/{obj_name}_PVI_plots.pdf")

    make_gif(obj, water, result_RK4, save=f"../gifs/{obj_name}_anim.gif",speed_up=speed_up)




#================================
#Newton
#================================
if what_to_do == "equilibrium":
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


#================================
#Plots of level
#================================
if what_to_do == "contour levels":
    beautiful_plot(obj, water, save=f"../imgs/{obj_name}_levels.pdf")

if show_plot:
    plt.show()