from objects import *
from quadrature import *
import numpy as np
import matplotlib.pyplot as plt

obj = Contour.read_file("lmacc.txt")

fig = plt.figure(figsize=(10,10), dpi=100)
ax = fig.add_subplot(111)
obj.discretize_n_lines(5)

obj.plot_discretized(ax, True)

plt.show()