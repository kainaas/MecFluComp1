from objects import *
from quadrature import *
import numpy as np
import matplotlib.pyplot as plt

obj = Contour.read_file("circle.txt")

obj.discretize_n_lines(1000)

volume = int_volume_contour(obj)
cm = int_cm_contour(obj)
print("volume: ", volume)
print("mass center: ", cm)

obj.translate(-cm[0], -cm[1])

obj.discretize_n_lines(6)


fig = plt.figure(figsize=(20,10), dpi=100)
ax = fig.add_subplot(121)
ax.set_xlim(-15.0, 15.0)
ax.set_ylim(-15.0, 15.0)
ax.grid(True)
obj.plot_discretized(ax, True)

ax2 = fig.add_subplot(122)
ax2.set_xlim(-15.0, 15.0)
ax2.set_ylim(-15.0, 15.0)
ax2.grid(True)
obj.plot(ax2)


plt.show()