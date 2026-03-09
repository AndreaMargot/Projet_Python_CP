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
                norm = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
                a_i[0] += (masses[j]*diff_x)/(norm**3)
                a_i[1] += (masses[j]*diff_y)/(norm**3)
                a_i[2] += (masses[j]*diff_z)/(norm**3)
        a[i]=G*a_i
    return a

def update_pos_v(positions, vitesses, masses, dt):
    """
    Utilise un schéma de Runge-Kutta d'ordre 4 pour mettre 
    à jour la position et la vitesse de chaque corps de la 
    galaxie
    
    """

    # Etape 1 
    a1 = compute_acceleration(positions,masses)
    v1 = vitesses
    p1 = positions

    # Etape 2 
    a2 = compute_acceleration(p1 + 0.5*dt*v1,masses)
    v2 = v1 + 0.5*dt*a1
    p2 = p1 + 0.5*dt*v1

    # Etape 3 
    a3 = compute_acceleration(p2 + 0.5*dt*v2,masses)
    v3 = v1 + 0.5*dt*a2
    p3 = p1 + 0.5*dt*v2

    # Etape 4
    a4 = compute_acceleration(p3 + dt*v3,masses)
    v4 = v1 + dt*a3
    p4 = p1 + dt*v3

    # Mise à jour finale des positions et vitesses
    positions += (dt/6.0)*(v1 + 2.0*v2 + 2.0*v3 + v4)
    vitesses += (dt/6.0)*(a1 + 2.0*a2 + 2.0*a3 + a4)

    return positions, vitesses 
