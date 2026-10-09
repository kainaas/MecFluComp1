import sys
sys.path.append("./../../src")

from liquid import *
import numpy as np

#================================
#Parameters definition
#================================
obj_name = "triangulo_retangulo"
density = 500.0
what_to_do = "PVI" #can be PVI, equilibrium, contour levels or convergence
show_plot = False
water = Liquid(1000.0)

#PVI parameters
gamma1_l = [1.0, 500.0, 5000.0]
gamma2_l = [0.5, 1.0, 10.0]
u0 = np.array([
    0.0, #b2
    0.0, #y-speed
    0.0, #theta
    0.0 #angular speed
])

t0 = 0.0
t_final = 40.0
h_l = [0.03]

speed_up = 2

#Convergence parameters
gamma1_c = 400.0
gamma2_c = 1.0

#Newton parameters
theta0_l = [0.0, 1.8, 3.2, 3*np.pi/2]
b0_l = [0.1, 0.0, 0.0, 0.0]
db2 = 1e-4
dtheta = 1e-4
tol = 1e-11