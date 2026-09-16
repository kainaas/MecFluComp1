import numpy as np
from enum import Flag, auto

from objects import *
from quadrature import *
from liquid import *
from rigid_transformation import *
import const


class Stop_Criteria(Flag):
    ITERATIONS = auto()
    FUNC_TOLERANCE = auto()
    STEP_TOLERANCE = auto()






#F1 = E + W
#F2 = Tau

def F1(ctr: Contour, liquid: Liquid, weight: float) ->float:
    return weight + int_bouyant_optm(ctr, liquid)

def F2(ctr: Contour, liquid: Liquid, cm: np.ndarray | None = None)  -> float:
    if cm is None:
        cm = int_cm_optm(ctr)
    return int_torque_optm(ctr, liquid, cm)



def calc_jacobian(ctr: Contour, liquid: Liquid, dtheta: float, db2: float, weight: float, cm: np.ndarray | None = None) -> np.array:
    if cm is None:
        cm = int_cm_contour(ctr)

    original_theta = ctr.theta
    original_b2 = ctr.pos[1]

    #del F / del theta
    rigid_transformation(ctr, original_theta + dtheta, original_b2, cm)
    F1_dtheta_p = F1(ctr, liquid, weight)
    F2_dtheta_p = F2(ctr, liquid, cm)
    
    rigid_transformation(ctr, original_theta -  dtheta, original_b2, cm)
    F1_dtheta_m = F1(ctr, liquid,weight)
    F2_dtheta_m = F2(ctr, liquid, cm)

    delF1_deltheta = (F1_dtheta_p - F1_dtheta_m) / (2* dtheta)
    delF2_deltheta = (F2_dtheta_p - F2_dtheta_m) / (2* dtheta)

    #del F / del b2
    rigid_transformation(ctr, original_theta, original_b2 + db2, cm)
    F1_db2_p = F1(ctr, liquid, weight)
    F2_db2_p = F2(ctr, liquid)

    rigid_transformation(ctr, original_theta, original_b2 - db2)
    F1_db2_m = F1(ctr, liquid, weight)
    F2_db2_m = F2(ctr, liquid)

    delF1_delb2 = (F1_db2_p - F1_db2_m) / (2* db2)
    delF2_delb2 = (F2_db2_p - F2_db2_m) / (2* db2)

    rigid_transformation(ctr, original_theta, original_b2)

    return np.array([[delF1_deltheta, delF1_delb2], [delF2_deltheta, delF2_delb2]])



#Newton method with finite differences
def Newton_FDM(
        ctr: Contour, liquid: Liquid, 
        b0: float, theta0: float, 
        dtheta: float, db2: float, 
        tol: float, max_it: int = 100,
        verbose: bool = False
) -> np.array:

    weight = calc_weight_optm(ctr)
    x_last = point(theta0, b0)
    rigid_transformation(ctr, theta0, b0)
    cm_last = int_cm_optm(ctr)
    F_last = point(F1(ctr, liquid, weight), F2(ctr, liquid, cm_last))

    if verbose:
        print("\n==================================================")
        print("Initiating Newton iteration with finite differences")
        print("tol:", tol, "; max iterations:", max_it)
        print("theta0 =", theta0, "; b0 =", b0, "; dtheta =", dtheta, "; db2 =", db2)
        print("==================================================")

    for i in range(max_it):
        J = calc_jacobian(ctr, liquid, dtheta, db2, weight, cm_last)

        delta = np.linalg.solve(J, -F_last)
        x_now = x_last + delta
        
        rigid_transformation(ctr, x_now[0], x_now[1], cm_last)
        cm_now = int_cm_optm(ctr)

        F_now = point(F1(ctr, liquid, weight), F2(ctr, liquid, cm_now))
        
        if verbose:
            print("Iteration =", i)
            print("theta =", x_now[0], "; b2 =", x_now[1])
            print("F values:", F_now)
            print("==================================================")


        if (np.linalg.norm(delta) < tol and np.linalg.norm(F_now) < tol) or i == max_it - 1:
            if verbose:
                print("\n==================================================")
                if i == max_it - 1:
                    print("Newton iteration terminated by reaching max iterations (", max_it, ")")
                else:    
                    print("Newton iteration terminated in", i, "iterations")
                print("==================================================")
                print("Solution: theta = ", x_now[0], "; b2 = ", x_now[1])
                print("delta norm:", np.linalg.norm(delta))
                print("F norm: ", np.linalg.norm(F_now))
                print("F values: ", F_now)
                print("==================================================")
            return x_now

        F_last = F_now
        x_last = x_now
        cm_last = cm_now


    