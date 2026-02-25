import corps_numba as galaxyv2
import visualizer3d_vbo as visual 
import galaxy_generator as generator
import numpy as np 
import time

n_stars = 100
masses, positions, vitesses, colors = generator.generate_galaxy(n_stars = n_stars)

def update(dt):

    global positions
    global vitesses
    global masses
    masses = np.array(masses, dtype=np.float64)
    positions = np.array(positions, dtype=np.float64) # Doit être de forme (N, 3)
    vitesses = np.array(vitesses, dtype=np.float64)
    t1 = time.time()
    f = galaxyv2.force_attraction(masses, positions) 
    t2 = time.time()
    print(f"temps calcul accelération : {t2-t1} s")
    positions, vitesses = galaxyv2.update_pos_v(positions,vitesses,f,dt)
    t3 = time.time()
    print(f"maj : {t3 - t2} s")
    print(f" Execution time : {t3 - t1} s")
    return positions

luminosities = np.random.uniform(0.3, 1.0, n_stars+1).astype(np.float32)
bounds = ((-3, 3), (-3, 3), (-3, 3))
visualizer = visual.Visualizer3D(positions, colors, luminosities, bounds)
visualizer.run(update)

#temps d'execution : ~ 0 s ou 0.0009 s
#maj :  ~ 0 s
#temps calcul accelération : ~ 0 s
