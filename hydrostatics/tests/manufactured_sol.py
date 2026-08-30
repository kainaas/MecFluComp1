import sys
sys.path.append("./..")

from objects import *
from quadrature import *
from liquid import *
from solve import *
import const

import numpy as np
import matplotlib.pyplot as plt

show_plot = False

water = Liquid()

obj = Contour.read_point_list("../objects/manufactured_sol.txt", 0.5, False)

obj.discretize_n_lines(1000)

volume = int_volume_contour(obj)
cm = int_cm_contour(obj)

print("volume: ", volume)
print("mass center: ", cm)

#Fully submerged
obj.translate(0.0, -2.0)
BF1 = BF = int_bouyant_contour(obj, water)
print("Bouyant force (fully submerged): ", BF1)
#Partially submerged
obj.translate(0.0, 1.0)
BF2 = BF = int_bouyant_contour(obj, water)
print("Bouyant force (partially submerged): ", BF2)


move_cm_to_origin(obj, reset_pos=True)



theta0 = - np.pi / 2
b0 = 0.0
db2 = 1e-4
dtheta = 1e-4
tol = 1e-14

solution = Newton_FDM(obj, water, b0, theta0, dtheta, db2, tol, verbose=True)

weight = calc_weight(obj)
BF = int_bouyant_contour(obj, water)
torque = int_torque_contour(obj, water)

print("")
print("Solution: theta = ", solution[0], "; b2 = ", solution[1])
print("Weight: ", weight)
print("Bouyant force: ", BF)
print("Torque: ", torque)


fig = plt.figure(figsize=(20,10), dpi=100)


ax2 = fig.add_subplot(122)
ax2.set_xlim(-3.0, 3.0)
ax2.set_ylim(-3.0, 3.0)
ax2.grid(True)
obj.plot(ax2)
ax2.scatter(0.0, solution[1])

ax = fig.add_subplot(121)
ax.set_xlim(-0.5, 5.5)
ax.set_ylim(-0.5, 5.5)
ax.grid(True)
ax.scatter(cm[0], cm[1])
rigid_transformation(obj, 0.0, cm[1])
obj.translate(cm[0], 0.0)
obj.plot(ax)

if show_plot:
    plt.show()