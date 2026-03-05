import numpy as np

def maj_pos_v(positions, vitesses, a, dt): 
    """
    mise à jour de la position et de la vitesse de
    chaque corps de la galaxie 
    
    """
    positions = positions + dt*vitesses + (dt**2)*a/2
    vitesses = vitesses + dt*a
    return positions, vitesses

def acc(masses, positions):
    G = 1.560339*1e-13
    diff = positions[: , np.newaxis, :] - positions[np.newaxis, :, :] #cube
    a = -G * masses[np.newaxis,:, np.newaxis] * diff / (np.linalg.norm(diff, axis = -1)**3+1e-15)[:, :, np.newaxis]
    a_i = np.sum(a, axis = 1)

    return a_i
