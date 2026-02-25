import numpy as np

def maj_v_a(position, vitesse, a, dt): #maj de la position et vitesse à partir de l'accélération et un pas de temps
    position = position + dt*np.array(vitesse) + (dt**2)*a/2
    vitesse = vitesse + dt*a
    return position, vitesse

def force_attraction(masse, position):
    G = 1.560339*1e-13
    masse = np.array(masse)
    position = np.array(position)
    diff = position[: , np.newaxis, :] - position[np.newaxis, :, :] #cube
    f = -G * masse[np.newaxis,:, np.newaxis] * diff / (np.linalg.norm(diff, axis = -1)**3+1e-15)[:, :, np.newaxis]
    f_i = np.sum( f, axis = 1)

    return f_i
