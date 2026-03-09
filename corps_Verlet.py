import numpy as np 
import numba

@numba.njit(parallel=True)
def compute_acceleration(positions,masses):
    nb_corps = len(positions)
    a : np.ndarray = np.empty((nb_corps,3), dtype = np.float64)
    G : numba.float64 = 1.560339*1e-13

    for i in numba.prange(nb_corps):
        a_i = np.zeros(3, dtype = np.float64)
        for j in numba.prange(nb_corps):
            if j != i:
                diff_x = positions[j, 0] - positions[i, 0]
                diff_y = positions[j, 1] - positions[i, 1]
                diff_z =  positions[j, 2] - positions[i, 2]
                norm = diff_x**2 + diff_y**2 + diff_z**2
                a_i[0] += (masses[j]*diff_x)/(norm**3)
                a_i[1] += (masses[j]*diff_y)/(norm**3)
                a_i[2] += (masses[j]*diff_z)/(norm**3)
        a[i]=G*a_i
    return a

def update_pos_v(positions, vitesses, masses, dt):
    """
    Utilise la méthode de Verlet vectorisée pour mettre 
    à jour la position et la vitesse de chaque corps de 
    la galaxie

    """
    
    a = compute_acceleration(positions, masses)
    positions += vitesses*dt + 0.5*a*dt*dt
    a_new = compute_acceleration(positions, masses)
    vitesses += 0.5*(a + a_new)*dt
    return positions, vitesses

