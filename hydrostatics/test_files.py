from objects import *
from quadrature import *
from liquid import *
import const

import numpy as np
import matplotlib.pyplot as plt

water = Liquid()

obj = Contour.read_file("objects/test.txt")

obj.discretize_n_lines(1000)

volume = int_volume_contour(obj)
cm = int_cm_contour(obj)
weight = calc_weight(obj)


print("volume: ", volume)
print("mass center: ", cm)
print("Weight: ", weight)


obj.translate(-cm[0], -cm[1])

obj.reset_position()

BF = int_bouyant_contour(obj, water)
print("Bouyant force: ", BF)

cm = point(0.0, 0.0)

torque = int_torque_contour(obj, water)
print("Torque: ", torque)

obj.discretize_n_lines(6)


fig = plt.figure(figsize=(20,10), dpi=100)
ax = fig.add_subplot(121)
# ax.set_xlim(-15.0, 15.0)
# ax.set_ylim(-15.0, 15.0)
ax.grid(True)
obj.plot_discretized(ax, True)

ax2 = fig.add_subplot(122)
ax2.set_xlim(-15.0, 15.0)
ax2.set_ylim(-15.0, 15.0)
ax2.grid(True)
obj.plot(ax2)


#plt.show()