import numpy as np
import numba


def update_pos_v(position, vitesse, acceleration, delta_t):
    position = position + delta_t*vitesse + 0.5*(delta_t)**2*acceleration
    vitesse = vitesse + delta_t*acceleration
    return position, vitesse


@numba.njit(parallel=True)
def force_attraction(masse, position): #renvoie l'accelération
    n = len(masse)
    f : np.ndarray = np.empty((n,3),dtype=np.float64) #array non initialisé
    G : numba.float64 = 1.560339*1e-13
    #G : np.double = 1.560339*1e-13

    for i in numba.prange(n):
        force = np.zeros(3,dtype=np.float64)
        for j in numba.prange(n):
           if j != i:
                diff_x = position[j, 0] - position[i, 0]
                diff_y = position[j, 1] - position[i, 1]
                diff_z =  position[j, 2] - position[i, 2]
                norm = diff_x**2 + diff_y**2 + diff_z**2
                force[0] += (masse[j]*diff_x)/(norm**3)
                force[1] += (masse[j]*diff_y)/(norm**3)
                force[2] += (masse[j]*diff_z)/(norm**3)
        f[i]=G*force
    return f #np.ndarray
