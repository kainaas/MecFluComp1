import sys
sys.path.append("./..")

from objects import *
from quadrature import *
from liquid import *
from solve import *
import const
from tabulate import tabulate

import numpy as np
import matplotlib.pyplot as plt

water = Liquid()




cm_true = point(1.0, 1.0)
volume_true = np.pi
weight_true = - volume_true * const.g

#Torque and Bouyante force when cm = (0.0, 0.0)
BF_true = - weight_true/2
torque_true = 0.0



num_lines = [2**i for i in range(2,15)]

volume_diff = []
cm_diff = []
weight_diff = []
torque_diff = []
BF_diff = []

table = []
table_factors = []
for i, n in enumerate(num_lines):
    obj = Contour.read_file("../objects/circle.txt")
    obj.discretize_n_lines(n)
    volume_diff.append(np.abs(volume_true - int_volume_optm(obj)))
    cm_diff.append(np.linalg.norm( cm_true - int_cm_optm(obj)))
    weight_diff.append(np.abs(weight_true - calc_weight_optm(obj)))
    
    move_cm_to_origin(obj, point(1.0,1.0), True)
    torque_diff.append(np.abs(torque_true - int_torque_optm(obj, water)))
    BF_diff.append(np.abs(BF_true - int_bouyant_optm(obj, water)))

    table.append([n, volume_diff[i], cm_diff[i], weight_diff[i], torque_diff[i], BF_diff[i]])

    if i != 0:
        volume_factor = volume_diff[i] / volume_diff[i-1]
        cm_factor = cm_diff[i] / cm_diff[i-1]
        weight_factor = weight_diff[i] / weight_diff[i-1]
        torque_factor = torque_diff[i] / torque_diff[i-1]
        BF_factor = BF_diff[i] / BF_diff[i-1]

        table_factors.append([n, volume_factor, cm_factor, weight_factor, torque_factor, BF_factor])
    else:
        table_factors.append([n, 0.0, 0.0, 0.0, 0.0, 0.0])


print(tabulate(table, headers=["num lines","volume error", "cm error", "weight error", "torque error", "BF error"], tablefmt="grid", numalign="left", stralign="left"))
print(tabulate(table_factors, headers=["num lines","volume factor", "cm factor", "weight factor", "torque factor", "BF factor"], tablefmt="grid", numalign="left", stralign="left"))
