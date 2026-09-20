import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.lines import Line2D
from solve import *
from objects import *
from liquid import *
from quadrature import *
from rigid_transformation import *
from solve_PVI import *


def calc_F_output(
    ctr: Contour,
    liquid: Liquid,
    type: str,
    n_points_b: int = 100,
    n_points_theta: int = 100
):
    '''
        Returns a matrix suited to be plotted in a colorbar. This matrix have values of F1 or F2 in function of theta and b2
    '''

    t = False
    if type == "F1":
        t = 0
    elif type == "F2":
        t = 1
    
    max_dist = ctr.find_biggest_dist()
    
    move_cm_to_origin(ctr, reset_pos=True)
    y = np.linspace(
        -max_dist / 2.0,
        max_dist / 2.0,
        n_points_b
    )

    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        n_points_theta,
        endpoint=False
    )
    output = np.zeros((n_points_b, n_points_theta))

    weight = calc_weight_optm(ctr)

    for j in range(n_points_b):
        move_cm_to_origin(ctr, reset_pos=True)
        rigid_transformation(ctr, 0.0, y[j])
        cm = int_cm_optm(ctr)

        for i in range(n_points_theta):
            rigid_transformation(ctr, theta[i], y[j],cm)
            if t == 0:
                output[j, i] = F1(ctr, liquid, weight)
            elif t == 1:
                output[j, i] = F2(ctr, liquid, cm)
    return output





def map_F(
    ctr: Contour,
    type: str,
    fig,
    ax,
    F1: np.array | None = None,
    F2: np.array | None = None,
    n_points_b: int = 100,
    n_points_theta: int = 100
):
    ''' 
        Type may be "F1", "F2", "level", "quiver" or "combined". 
        F1 and F2 plots these functions according to theta and b2
        level plots the contour levels of both functions
        quiver plots the vector field (F1, F2)
        combined plots both the vector field and the contour levels.
    '''
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$b_2$")
    #ax.set_aspect('equal', adjustable='box')
    t = False
    if type == "F1":
        t = 0
    elif type == "F2":
        t = 1
    elif type == "level":
        t = 2
    elif type == "quiver":
        t = 3
    elif type == "combined":
        t = 4
    max_dist = ctr.find_biggest_dist()

    y = np.linspace(
        -max_dist / 2.0,
        max_dist / 2.0,
        n_points_b
    )

    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        n_points_theta,
        endpoint=False
    )

    x_grid, y_grid = np.meshgrid(theta, y)

    output: np.array
    if t == 0: output = F1
    elif t == 1: output = F2

    if (t == 0) or (t == 1): 
        if t ==0:
            ax.set_title(r"$F_1(\theta, b_2)$")
        else:
            ax.set_title(r"$F_2(\theta, b_2)$")
        epsilon = 1e-6
        c: plt.QuadMesh
        if t == 0 or t == 1:
            vmax = np.max(np.abs(output))

            norm = SymLogNorm(
                linthresh=epsilon,
                vmin=-vmax,
                vmax=vmax
            )

            c = ax.pcolormesh(
                x_grid,
                y_grid,
                output,
                cmap="seismic",
                norm=norm,
                shading="auto"
            )
        fig.colorbar(c, ax=ax, label=type)

    if t == 2 or t == 4:

        if t == 2:
            ax.set_title("Contour lines")
        else:
            ax.set_title("Contour lines and arrows")
        c1 = ax.contour(x_grid, y_grid, F1, levels=[0], colors="blue")
        c2 = ax.contour(x_grid, y_grid, F2, levels=[0], colors="red")
        legend_lines = [
            Line2D([0], [0], color="blue", label=r"$F_1=0$"),
            Line2D([0], [0], color="red", label=r"$F_2=0$")
        ]

        ax.legend(handles=legend_lines)

    if t == 3 or t == 4:
        if t == 3:
            ax.set_title(r"$(F_1, F_2)$")
        magnitude = np.sqrt(F1**2 + F2**2)

        U = np.divide(
            F1,
            magnitude,
            out=np.zeros_like(F1),
            where=magnitude > 0
        )

        V = np.divide(
            F2,
            magnitude,
            out=np.zeros_like(F2),
            where=magnitude > 0
        )
        ax.quiver(
            x_grid,
            y_grid,
            U,
            V
        )

def beautiful_plot(
    ctr: Contour, 
    liquid: Liquid, 
    b_points: int = 200, 
    theta_points: int = 200,
    b_less: int = 20, 
    theta_less:int = 20,
    show: bool = False, 
    save: str | None = None
):
    '''
        Plots a figure with 4 subplots containing a plot of F1, F2, contour level and vector field.
    '''
    fig = plt.figure(figsize=(2000, 1000, "px"))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    F1_map = calc_F_output(ctr, liquid, "F1", b_points, theta_points)
    F2_map = calc_F_output(ctr, liquid, "F2",  b_points, theta_points)
    map_F(ctr, "F1", fig, ax1, F1_map, None,  b_points, theta_points)
    map_F(ctr, "F2", fig, ax2, None, F2_map,  b_points, theta_points)
    map_F(ctr, "level", fig, ax3, F1_map, F2_map, b_points, theta_points)

    F1_less = calc_F_output(ctr, liquid, "F1", b_less, theta_less)
    F2_less = calc_F_output(ctr, liquid, "F2", b_less, theta_less)
    map_F(ctr, "quiver", fig, ax4, F1_less, F2_less, b_less, theta_less)
    if not (save is None):
        fig.savefig(save)
    if show:
        plt.show()

    

def make_gif(ctr: Contour, liquid: Liquid, result: Result, interval: int = 33, show: bool = False, save:str | None = None):
    '''
        Given a PVI solution, makes a gif of this solution. interval is the period of time in ms between each frame (33 is arround 30 fps).
        Not recommended going aboe 30 fps
    '''
    fig, ax = plt.subplots()
    fig.set_size_inches((10,10))

    # Limites fixos do gráfico
    ax.set_aspect("equal")

    L = ctr.find_biggest_dist()

    ax.set_xlim(-2 * L, 2 * L)
    ax.set_ylim(-2 * L, 2 * L)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    line, = ax.plot([], [], lw=2)
    line.set_color("black")

    # Linha representando a superfície da água
    water_line, = ax.plot(
        [-2 * L, 2 * L],
        [liquid.y_inter, liquid.y_inter],
        color="blue",
        lw=1
    )
    blup = ax.fill_between(
        [-2 * L, 2* L],
        -2* L,
        liquid.y_inter,
        color="blue",
        alpha=0.3
    )

    time_text = ax.text(
        0.02, 0.9,
        "",
        transform=ax.transAxes
    )


    def init():
        line.set_data([], [])
        time_text.set_text("")
        return line, time_text


    def update(frame):
        time = int((interval/(result.h*1000))*frame)
        if time >= result.qnt_passos:
            time = result.qnt_passos - 1
        # Estado da solução neste instante
        b2 = result.u[0, time]
        theta = result.u[2, time]

        # Atualiza a posição/orientação do objeto
        rigid_transformation(
            ctr,
            theta,
            b2
        )

        # Coordenadas dos segmentos
        x = np.vstack((
            ctr.Lmatrix[0, :],
            ctr.Lmatrix[3, :]
        )).ravel(order='F')

        y = np.vstack((
            ctr.Lmatrix[1, :],
            ctr.Lmatrix[4, :]
        )).ravel(order='F')

        line.set_data(x, y)

        time_text.set_text(
            f"$t = {result.t[time]:.2f}$ s\n"
            f"$b_2 = {b2:.3f}$\n"
            f"$\\theta = {theta:.3f}$"
        )

        return line, time_text



    anim = FuncAnimation(
        fig,
        update,
        frames=int((1/(interval/1000))*(np.shape(result.t)[0] * result.h)),
        init_func=init,
        interval=interval,
        blit=True
    )
    if not (save is None):
        anim.save(save)
    if show:
        plt.show()


if __name__ == "__main__":
    obj_name = "manufactured_sol"
    if obj_name == "manufactured_sol":
        obj = Contour.read_point_list("objects/manufactured_sol.txt", 0.5, False)
    else:
        obj = Contour.read_file(f"objects/{obj_name}.txt", 0.5)
    obj.discretize_n_lines(1000)
    water = Liquid()

    beautiful_plot(obj, water, save=f"imgs/{obj_name}_levels.pdf")

    