import corps_vectorise as galaxyv2
import visualizer3d_vbo as vz 
import galaxy_generator as gg
import numpy as np 
import time

n_stars = 100
masses, positions, vitesses, colors = gg.generate_galaxy(n_stars = n_stars)

def update(dt):

    global positions
    global vitesses
    global masses

    masses = np.array(masses)
    positions = np.array(positions) 
    vitesses = np.array(vitesses)
  
    t1 = time.time()
    acc = galaxyv2.acc(masses, positions)
    t2 = time.time()
    positions, vitesses = galaxyv2.maj_pos_v(positions,vitesses,acc,dt)
    t3 = time.time()
    print(f"Temps de calcul accélération : {t2 - t1} s ")
    print(f"Temps de calcul maj positions, vitesses : {t3 - t2} s")
    print(f"Execution time : {t3 - t1} s")
    return positions

luminosities = np.random.uniform(0.3, 1.0, n_stars+1).astype(np.float32)
bounds = ((-3, 3), (-3, 3), (-3, 3))
visualizer = vz.Visualizer3D(positions, colors, luminosities, bounds)
visualizer.run(update)

#temps d'éxécution : ~ 0.002 s
#temps de calcul maj positions, vitesses : ~ 0 s
#temps de calcul accélération : ~ 0.002 s
