from objects import *
from quadrature import *
from liquid import *
from solve import *
import const

import numpy as np
import matplotlib.pyplot as plt

water = Liquid()

obj = Contour.read_file("objects/lmacc.txt", 0.5)

obj.discretize_n_lines(1000)

volume = int_volume_contour(obj)
cm = int_cm_contour(obj)


move_cm_to_origin(obj, reset_pos=True)


print("volume: ", volume)
print("mass center: ", cm)


solution = Newton_FDM(obj, water, 0.0, - np.pi / 4, 0.001, 0.001, 1e-4)

weight = calc_weight(obj)
BF = int_bouyant_contour(obj, water)
torque = int_torque_contour(obj, water)

print("Solution:", solution)
print("Weight: ", weight)
print("Bouyant force: ", BF)
print("Torque: ", torque)

obj.discretize_n_lines(10)


fig = plt.figure(figsize=(20,10), dpi=100)
ax = fig.add_subplot(121)
# ax.set_xlim(-15.0, 15.0)
# ax.set_ylim(-15.0, 15.0)
ax.grid(True)
obj.plot_discretized(ax, True)

ax2 = fig.add_subplot(122)
# ax2.set_xlim(-15.0, 15.0)
# ax2.set_ylim(-15.0, 15.0)
ax2.grid(True)
obj.plot(ax2)


plt.show()