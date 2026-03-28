import visualizer3d_vbo as visual
import galaxy_generator as generator
import numpy as np
import time
import numba

n_stars = 200
masses, positions, vitesses, colors = generator.generate_galaxy(n_stars = n_stars)

def acceleration(n_stars, positions, masses ):

    boxes, l, L = box(positions, n_stars)
    G = 1.560339 * 1e-13
    acc = np.zeros_like(positions)
    list_boxes = [b for rangee in boxes for b in rangee if len(b) > 0]

    all_cgrav, m_boxes = centres_gravite(boxes, positions, masses)
    seuil = np.sqrt(l**2 + L**2) / 10

    for i in range(n_stars):
        pos_i = positions[i]
        diff_boxes = all_cgrav - pos_i
        dist = np.sqrt(np.sum(diff_boxes**2, axis=1)) 

        loin = dist >= seuil 
        inv_dist3 = 1 / (dist[loin]**3)
        acc[i] += G * np.sum(m_boxes[loin][:, np.newaxis] * diff_boxes[loin] * inv_dist3[:, np.newaxis], axis=0)

        proche = np.where(~loin)[0] 
        if proche.size > 0 :

            for ind_boite in proche:

                ind_etoiles_vois = list_boxes[ind_boite]

                for j in ind_etoiles_vois:

                    if i != j :
                        diff = positions[j] - pos_i
                        d = np.sqrt(np.sum(diff**2))

                        if d > 1e-13:
                            acc[i] += G*masses[j]*diff/d**3
    return acc



def update_pos_v(position, vitesse, acceleration, delta_t):

    position = position + delta_t*vitesse + 0.5*(delta_t)**2*acceleration
    vitesse = vitesse + delta_t*acceleration

    return position, vitesse



def update(dt):

    global positions
    global vitesses
    global masses

    masses = np.array(masses, dtype=np.float64)
    positions = np.array(positions, dtype=np.float64) 
    vitesses = np.array(vitesses, dtype=np.float64)

    t1 = time.time()

    f = acceleration(n_stars, positions, masses)

    t2 = time.time()

    print(f"temps calcul accelération : {t2-t1} s")

    positions, vitesses = update_pos_v(positions,vitesses,f,dt)

    t3 = time.time()

    print(f"maj : {t3 - t2} s")
    print(f" Execution time : {t3 - t1} s")
    return positions


def centres_gravite(boxes, positions, masses):

    flat_boxes = [b for rangee in boxes for b in rangee if len(b) > 0]

    num_boxes = len(flat_boxes)
    cgravs = np.zeros((num_boxes, 3))
    m_totales = np.zeros(num_boxes)

    for i, oboite in enumerate(flat_boxes):

        m_boite = masses[oboite]
        p_boite = positions[oboite]

        m_somme = np.sum(m_boite)
        m_totales[i] = m_somme

        if m_somme > 0:
            cgravs[i] = np.sum(p_boite * m_boite[:, np.newaxis], axis=0) / m_somme

    return cgravs, m_totales


def box(positions, n_stars):

    x_min = min([positions[i][0] for i in range(n_stars)])
    x_max = max([positions[i][0] for i in range(n_stars)])

    y_min = min([positions[i][1] for i in range(n_stars)])
    y_max = max([positions[i][1] for i in range(n_stars)])

    l = y_max - y_min 
    L = x_max - x_min 

    interv_x = L/10 
    interv_y = l/10 

    C=[[[] for _ in range(10)] for _ in range(10)] 
    
    for i in range(n_stars):
        rest_x = int((positions[i][0]-x_min )// interv_x)  
        rest_y = int((positions[i][1] - y_min) // interv_y)
        C[rest_y-1][rest_x-1].append(i)

    return C, l/10, L/10  

luminosities = np.random.uniform(0.3, 1.0, n_stars+1).astype(np.float32)
bounds = ((-3, 3), (-3, 3), (-3, 3))
visualizer = visual.Visualizer3D(positions, colors, luminosities, bounds)
visualizer.run(update)
