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




obj_name = "cow"
file_type = "point list" #can be "point list" or "components" 
density = 0.5
show_plot = True
water = Liquid()


obj: Contour
if obj_name == "manufactured_sol":
    obj = Contour.read_point_list("../objects/manufactured_sol.txt", density, False)
else:
    if file_type == "point list":
        obj = Contour.read_point_list(f"../objects/{obj_name}.txt", density)
    else:
        obj = Contour.read_file(f"../objects/{obj_name}.txt", density)
obj.discretize_n_lines(1000)


ldt = [0.5, 0.25, 0.125, 0.0625, 0.03125] #, 0.03125

fig_plot = plt.figure(figsize=(2000, 1000, "px"))
ax1 = fig_plot.add_subplot(121)
ax2 = fig_plot.add_subplot(122)

ax1.set_title("Explicit Euler")
ax2.set_title("RK4")

fig_conv = plt.figure(figsize=(2000, 1000, "px"))
ax_conv = fig_conv.add_subplot(111)

gamma = 0.7
u0 = np.array([
    6.0, #b2
    0.0, #y-speed
    0.0, #theta
    3.0 #angular speed
])

t0 = 0.0
t_final = 20.0

volume = int_volume_optm(obj)
mass = volume * obj.density
weight = calc_weight_optm(obj)
I = int_MI_optm(obj)

f = lambda u, t: f_aux(u, t, obj, water, mass, weight, volume, I, gamma)


errorsEE_theta = []
errorsEE_b2 = []
errorsRK_theta = []
errorsRK_b2 = []


result_fine = RK4(t0, t_final, 0.00390625, u0, f)
for h in ldt:
    
    result_e = euler_explicito(t0, t_final, h, u0, f)
    result_RK4 = RK4(t0, t_final, h, u0, f)

    ax1.plot(result_e.t, result_e.u[0], label=fr"$b_2, dt = {h}$")
    ax1.plot(result_e.t, result_e.u[2], label=fr"$\theta, dt = {h}$")
    ax2.plot(result_RK4.t, result_RK4.u[0], label=fr"$b_2, dt = {h}$")
    ax2.plot(result_RK4.t, result_RK4.u[2], label=fr"$\theta, dt = {h}$")
    eEE_theta, eEE_b2 = calc_error(result_e, result_fine)
    eRK_theta, eRK_b2 = calc_error(result_RK4, result_fine)
    errorsEE_theta.append(eEE_theta)
    errorsEE_b2.append(eEE_b2)
    errorsRK_theta.append(eRK_theta)
    errorsRK_b2.append(eRK_b2)

print(errorsEE_theta)
print(errorsEE_b2)
print(errorsRK_theta)
print(errorsRK_b2)

ax_conv.loglog(ldt, errorsEE_b2, label=r"$b_2$ - EE")
ax_conv.loglog(ldt, errorsEE_theta, label=r"$\theta$ - EE")
ax_conv.loglog(ldt, errorsRK_b2, label=r"$b_2$ - RK4")
ax_conv.loglog(ldt, errorsRK_theta, label=r"$\theta$ - RK4")

ax_conv.loglog([0.3, 0.1], [10 * 0.3**4, 10 * 0.1**4], 'k--', label=r"$O(h^4)$")
ax_conv.loglog([0.3, 0.1], [5* 0.3, 5 * 0.1], 'k--', label=r"$O(h)$")
ax_conv.set_title("Erros na norma L2")
ax_conv.set_xlabel("h")
ax_conv.set_ylabel("error")
ax_conv.legend()

ax1.legend()
ax2.legend()
fig_conv.savefig(f"../imgs/{obj_name}_PVI_conv2.pdf")
fig_plot.savefig(f"../imgs/{obj_name}_PVI_conv.pdf")

if show_plot:
    plt.show()