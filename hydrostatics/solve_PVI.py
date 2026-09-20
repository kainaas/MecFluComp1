import sys
sys.path.append("./..")

from objects import *
from quadrature import *
from liquid import *
from solve import *
import const

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import floor

#Classe para retornar o resultado dos métodos
class Resultado:
    def __init__(self, qnt_passos, h, u, t):
        self.qnt_passos = qnt_passos #Quantidade de passos executada
        self.h = h #Passo de t
        self.u = u #Vetor com a solução
        self.t = t #Vetor com os valores do tempo utilizados. Note que  t_final <= t[-1] < t_final + h

def calcula_qnt_passos(t0, t_final, h):
    tam_intervalo = t_final - t0

    qnt_passos = tam_intervalo/h
    if qnt_passos != floor(qnt_passos): qnt_passos = floor(qnt_passos) + 1
    else: qnt_passos = int(qnt_passos)
    qnt_passos += 1 #Acrescentar u0 no vetor solução
    return qnt_passos

def inicializa_metodo(qnt_passos, t0, u0):
    u = np.zeros((len(u0), qnt_passos)) #Cria uma matriz em que as colunas são a solução em um determinado tempo e as linhas são a solução de uma variável no tempo todo
    t = np.zeros(qnt_passos)

    u[:, 0] = u0
    t[0] = t0
    return u, t

def euler_explicito(t0, t_final, h, u0: np.array, f):
    qnt_passos = calcula_qnt_passos(t0, t_final, h)
    u, t = inicializa_metodo(qnt_passos, t0, u0)

    for i in range(1,qnt_passos):
        t[i] = t[i-1] + h
        u[:, i] = u[:, i-1] + h*f(u[:, i-1], t[i-1])

    return Resultado(qnt_passos, h, u, t)
     

def RK4(t0, t_final, h, u0: np.array, f):
    qnt_passos = calcula_qnt_passos(t0, t_final, h)
    u, t = inicializa_metodo(qnt_passos, t0, u0)

    for i in range(1,qnt_passos):
        t[i] = t[i-1] + h
        K1 = f(u[:, i-1], t[i-1])
        K2 = f(u[:, i-1] + (h/2)*K1, t[i-1] + h/2)
        K3 = f(u[:, i-1] + (h/2)*K2, t[i-1] + h/2)
        K4 = f(u[:, i-1] + h*K3, t[i-1] + h)
        u[:,i] = u[:,i-1] + (h/6)*(K1 + 2*K2 + 2*K3 + K4)

    return Resultado(qnt_passos, h, u, t)


def f_aux(
    u: np.array, 
    t:float, 
    ctr: Contour, 
    liquid: Liquid, 
    mass: float, 
    weight: float, 
    volume:float, 
    moment_inertia: float,
    gamma:float
):
    rigid_transformation(ctr, u[2], u[0])
    return np.array([
        u[1],
        (1/mass)*(F1(ctr, liquid, weight) - gamma*volume*u[1]),
        u[3],
        (1/moment_inertia) * F2(ctr, liquid) -  gamma*u[3]
    ])



if __name__ == "__main__":
    #obj = Contour.read_point_list("objects/manufactured_sol.txt", 0.5, False)
    obj = Contour.read_file("objects/test.txt", 0.5)
    obj.discretize_n_lines(100)
    water = Liquid()
    gamma = 0.1

    move_cm_to_origin(obj, reset_pos=True)
    volume = int_volume_optm(obj)
    mass = volume * obj.density
    weight = calc_weight_optm(obj)
    I = int_MI_optm(obj)

    f = lambda u, t: f_aux(u, t, obj, water, mass, weight, volume, I, gamma)
        
    u0 = np.array([
        6.0, #b2
        0.0, #y-speed
        0.0, #theta
        1.0 #angular speed
    ])

    t0 = 0.0
    t_final = 20.0
    h = 0.01

    result_e = euler_explicito(t0, t_final, h, u0, f)
    result_RK4 = RK4(t0, t_final, h, u0, f)

    fig_plot = plt.figure()
    ax1 = fig_plot.add_subplot(121)
    ax2 = fig_plot.add_subplot(122)

    ax1.plot(result_e.t, result_e.u[0], label=r"$b_2$")
    ax1.plot(result_e.t, result_e.u[2], label=r"$\theta$")
    ax1.set_title("Explicit Euler")
    ax2.plot(result_RK4.t, result_RK4.u[0], label=r"$b_2$")
    ax2.plot(result_RK4.t, result_RK4.u[2], label=r"$\theta$")
    ax2.set_title("RK4")

    ax1.legend()
    ax2.legend()
    plt.show()

    