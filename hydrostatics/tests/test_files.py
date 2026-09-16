import sys
sys.path.append("./..")

from objects import *
from quadrature import *
from liquid import *
from solve import *
import const

import numpy as np
import matplotlib.pyplot as plt

water = Liquid()

obj = Contour.read_file("../objects/lmacc.txt", 0.5)

obj.discretize_n_lines(10000)

volume = int_volume_optm(obj)
cm = int_cm_optm(obj)

move_cm_to_origin(obj, reset_pos=True)


print("volume: ", volume)
print("mass center: ", cm)


theta0 = - np.pi 
b0 = 0.0
db2 = 1e-4
dtheta = 1e-4
tol = 1e-11

solution = Newton_FDM(obj, water, b0, theta0, dtheta, db2, tol, verbose=True)

weight = calc_weight_optm(obj)
BF = int_bouyant_optm(obj, water)
torque = int_torque_optm(obj, water)

print("")
print("Solution: theta = ", solution[0], "; b2 = ", solution[1])
print("Weight: ", weight)
print("Bouyant force: ", BF)
print("Torque: ", torque)

fig = plt.figure(figsize=(20,10), dpi=100)


ax2 = fig.add_subplot(122)
ax2.set_aspect('equal', adjustable='box')
ax2.grid(True)
obj.plot_discretized(ax2)
ax2.scatter(0.0, solution[1])

ax = fig.add_subplot(121)
ax.set_aspect('equal', adjustable='box')
ax.grid(True)
ax.scatter(cm[0], cm[1])
rigid_transformation(obj, 0.0, cm[1])
obj.translate(cm[0], 0.0)
obj.plot_discretized(ax)


plt.show()