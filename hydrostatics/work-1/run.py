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
#Choosing object
#================================

#from titanic import *
from triangulo_retangulo import *
#from iceberg import *

#================================
#Reading object
#================================
obj = Contour.read_point_list(f"../objects/{obj_name}.txt", density)
obj.discretize_n_lines(1)




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


move_cm_to_origin(obj, reset_pos=True)



#================================
#PVI
#================================
if what_to_do == "PVI":
    for h in h_l:
        for gamma1 in gamma1_l:
            for gamma2 in gamma2_l:
                f = lambda u, t: f_aux_float(u, t, obj, water, mass, weight, volume, I, gamma1, gamma2)

                result_RK4 = RK4(t0, t_final, h, u0, f)
                F1_vec = []
                F2_vec = []

                for i, t in enumerate(result_RK4.t):
                    theta = result_RK4.u[2,i]
                    b2 = result_RK4.u[0,i]
                    rigid_transformation(obj, theta, b2)
                    F1_vec.append(F1(obj, water, weight))
                    F2_vec.append(F2(obj, water))
                
                
                fig_plot = plt.figure(figsize=(1500,750,'px'))

                ax1 = fig_plot.add_subplot(121)
                ax1.set_xlabel("t (s)")
                ax1.set_ylabel("Forças em função de t")
                ax1.plot(result_RK4.t, F1_vec, label=r"$F_1$")
                ax1.plot(result_RK4.t, F2_vec, label=r"$F_2$")
                ax1.set_title("Forças - RK4")


                ax2 = fig_plot.add_subplot(122)
                ax2.set_xlabel("t (s)")
                ax2.set_ylabel("Posições em função de t")
                ax2.plot(result_RK4.t, result_RK4.u[0], label=r"$b_2$")
                ax2.plot(result_RK4.t, result_RK4.u[2], label=r"$\theta$")
                ax2.set_title("Posição - RK4")

                ax1.legend()
                ax2.legend()

                fig_plot.savefig(f"imgs/{obj_name}/PVI_h={h}_g1={gamma1}_g2={gamma2}.pdf")

                make_gif(obj, water, result_RK4, save=f"imgs/{obj_name}/anim_h={h}_g1={gamma1}_g2={gamma2}.gif",speed_up=speed_up)




#================================
#Newton
#================================
if what_to_do == "equilibrium":
    for i in range(len(theta0_l)):
        move_cm_to_origin(obj, reset_pos=True)

        solution, iterations = Newton_FDM(obj, water, b0_l[i], theta0_l[i], dtheta, db2, tol, verbose=True)

        print("")
        print("Solution: theta = ", solution[0], "; b2 = ", solution[1])


        fig = plt.figure(figsize=(20,10), dpi=100)

        L = obj.find_biggest_dist()

        ax = fig.add_subplot(111)
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)
        ax.grid(True)
        obj.plot_discretized(ax)
        ax.scatter(0.0, solution[1])
        ax.fill_between(
            [-2 * L, 2* L],
            -2* L,
            water.y_inter,
            color="blue",
            alpha=0.2
        )
        ax.text(
            0.02, 0.9,
            f"$b_2 = {solution[1]:.3f}$\n"
            f"$\\theta = {solution[0]:.3f}$\n"
            f"{iterations} iterations",
            transform=ax.transAxes
        )

        ax.set_aspect("equal")
        fig.savefig(f"imgs/{obj_name}/newton_b0={b0_l[i]:.3f}_theta0={theta0_l[i]:.3f}.pdf")


#================================
#Plots of level
#================================
if what_to_do == "contour levels":
    beautiful_plot(obj, water, b_points=100, theta_points=100, b_less=50, theta_less=50, save=f"imgs/{obj_name}/levels.pdf")



def calc_error(u:Result, u_fine:Result):
    '''
        u_fine must have values in the times u have value
    '''
    sum_b2 = 0.0
    sum_theta = 0.0
    for i, t in enumerate(u.t):
        j = np.argwhere(u_fine.t == t)
        sum_b2 += (u.u[0,i] - u_fine.u[0, j])**2
        sum_theta += (u.u[2,i] - u_fine.u[2, j])**2
    e_b2 = np.sqrt(sum_b2 * u.h)
    e_theta = np.sqrt(sum_theta * u.h)
    return e_theta[0][0], e_b2[0][0]

if what_to_do == "convergence":

    ldt = [0.5, 0.25, 0.125, 0.0625, 0.03125] #, 0.03125

    fig_plot = plt.figure(figsize=(2000, 1000, "px"))
    ax2 = fig_plot.add_subplot(111)
    ax2.set_title("RK4")

    fig_conv = plt.figure(figsize=(2000, 1000, "px"))
    ax_conv = fig_conv.add_subplot(111)
    f = lambda u, t: f_aux_float(u, t, obj, water, mass, weight, volume, I, gamma1_c, gamma2_c)

    errorsRK_theta = []
    errorsRK_b2 = []


    result_fine = RK4(t0, t_final, 0.00390625, u0, f)
    for h in ldt:
        result_RK4 = RK4(t0, t_final, h, u0, f)

        ax2.plot(result_RK4.t, result_RK4.u[0], label=fr"$b_2, dt = {h}$")
        ax2.plot(result_RK4.t, result_RK4.u[2], label=fr"$\theta, dt = {h}$")
        eRK_theta, eRK_b2 = calc_error(result_RK4, result_fine)
        errorsRK_theta.append(eRK_theta)
        errorsRK_b2.append(eRK_b2)

    print(errorsRK_theta)
    print(errorsRK_b2)

    ax_conv.loglog(ldt, errorsRK_b2, label=r"$b_2$ - RK4")
    ax_conv.loglog(ldt, errorsRK_theta, label=r"$\theta$ - RK4")

    ax_conv.loglog([0.3, 0.1], [10 * 0.3**4, 10 * 0.1**4], 'k--', label=r"$O(h^4)$")
    ax_conv.set_title("Erros na norma L2")
    ax_conv.set_xlabel("h")
    ax_conv.set_ylabel("error")
    ax_conv.legend()

    ax2.legend()
    fig_conv.savefig(f"imgs/{obj_name}/PVI_conv2.pdf")
    fig_plot.savefig(f"imgs/{obj_name}/PVI_conv.pdf")


if show_plot:
    plt.show()