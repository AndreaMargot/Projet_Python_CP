import numpy as np
import numba


def update_pos_v(positions, vitesses, accelerations, delta_t):
    positions = positions + delta_t*vitesses + 0.5*(delta_t)**2*accelerations
    vitesses = vitesses + delta_t*accelerations
    return positions, vitesses


@numba.njit(parallel=True)
def acc(masses, positions):
    """
    renvoie un tableau qui contient l'accélération de
    chaque corps de la galaxie 
    
    """
    nb_corps = len(masses)
    a : np.ndarray = np.empty((nb_corps,3),dtype=np.float64) 
    G : numba.float64 = 1.560339*1e-13


    for i in numba.prange(nb_corps):
        a_i = np.zeros(3,dtype=np.float64)
        for j in numba.prange(nb_corps):
           if j != i:
                diff_x = positions[j, 0] - positions[i, 0]
                diff_y = positions[j, 1] - positions[i, 1]
                diff_z =  positions[j, 2] - positions[i, 2]
                norm = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
                a_i[0] += (masses[j]*diff_x)/(norm**3)
                a_i[1] += (masses[j]*diff_y)/(norm**3)
                a_i[2] += (masses[j]*diff_z)/(norm**3)
        a[i]=G*a_i
    return a #np.ndarray
