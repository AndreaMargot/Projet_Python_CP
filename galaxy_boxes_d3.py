import visualizer3d_vbo as visual 
import galaxy_generator as generator
import numpy as np 
import time

n_stars = 200
masses, positions, vitesses, colors = generator.generate_galaxy(n_stars = n_stars)
#on enlève le trou noir?
#masses = masses[1:]
#positions = positions[1:]
#vitesses = vitesses[1:]
#colors = colors[1:]

def acceleration(n_stars, positions, masses ):
    acc = np.zeros((n_stars+1,3)) #prendre les coordonnées en z ou pas?
    boxes, l, L = box(positions, n_stars) 
    for i in range(n_stars): #on parcours toutes les étoiles
        for rangee in boxes:    #toutes les "lignes"  du cadrillage de la galaxies
            for oboite in rangee: #chaque boîte de la ligne de la galaxie
                cgrav = centre_gravité(oboite, positions, masses) #centre de gravité de la boîte (x, y, masse)
                diff = cgrav[:-1] - positions[i+1] #taille : (1,2)
                if np.linalg.norm(diff) >= np.sqrt(l**2+L**2)/10: #si la distance entre le centre de gravité de la boite et l'étoile 
                    #est >= diamètre de la boîte: on considère la galaxie
                    
                    acc[i+1] += cgrav[-1]*diff/np.linalg.norm(diff)**3
                
                else : #on considère l'étoile
                    for etoile in oboite:
                        diff = positions[etoile]-positions[i+1] #taille : (1,3)
                        norm_diff = np.linalg.norm(diff)
                        if norm_diff > 1e-3: 
                            acc[i+1] += masses[etoile]*np.array(diff)/norm_diff**3 
    G = 1.560339*1e-13
    acc *= G
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
    positions = np.array(positions, dtype=np.float64) # Doit être de forme (N, 3)
    vitesses = np.array(vitesses, dtype=np.float64)

    t1 = time.time()
    #print(f"positions : {positions}")
    #print(f"vitesse : {vitesses}")

    f = acceleration(n_stars, positions, masses) 
    #print(f"acceleration : {f}")

    t2 = time.time()
    print(f"temps calcul accelération : {t2-t1} s")

    positions, vitesses = update_pos_v(positions,vitesses,f,dt)

    t3 = time.time()
    print(f"maj : {t3 - t2} s")
    print(f" Execution time : {t3 - t1} s")

    return positions

def centre_gravité(indices, positions, masses):
    coord_x = 0
    coord_y = 0
    coord_z = 0
    m=0
    x = 0
    y = 0
    z = 0
    for ind in indices : 
        x += masses[ind]*positions[ind][0]
        y += masses[ind]*positions[ind][1]
        z += masses[ind]*positions[ind][2]
        m += masses[ind]
    if m!= 0:
        x /= m
        y /= m
        z /= m
    return np.array([x, y, z, m])


def box(positions, n_stars):
    #on définit les bords de notre grille
    x_min = min([positions[i][0] for i in range(n_stars)])
    x_max = max([positions[i][0] for i in range(n_stars)])

    #print(f"taille positions : {len(positions)}, n_stars : {n_stars}")

    y_min = min([positions[i][1] for i in range(n_stars)])
    y_max = max([positions[i][1] for i in range(n_stars)])

    l = y_max - y_min #largeur de la boîte
    L = x_max - x_min #longueur de la boîte

    #on divise en 10 intervalles 
    interv_x = L/10 #longueur d'un intervalle en x
    interv_y = l/10 #longueur d'un intervalle en y

    C=[[[] for _ in range(10)] for _ in range(10)] #faire une matrice 10*10 de listes vides

    for i in range(n_stars): #pour chaque étoile on détermine sa sous-boite
        rest_x = int((positions[i][0]-x_min )// interv_x)  #indice de la position en x de l'étoile d'indice i
        rest_y = int((positions[i][1] - y_min) // interv_y)
        #print(f"rest_x : {rest_x}")
        #print(f"rest_y : {rest_y}")
        C[rest_y-1][rest_x-1].append(i)
        #C=np.array(C)
    return C, l/10, L/10  #C : liste de listes de listes







luminosities = np.random.uniform(0.3, 1.0, n_stars+1).astype(np.float32)
bounds = ((-3, 3), (-3, 3), (-3, 3))
visualizer = visual.Visualizer3D(positions, colors, luminosities, bounds)
visualizer.run(update)

#temps d'execution : 
#maj :  
#temps calcul accelération : 
