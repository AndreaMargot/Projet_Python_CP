import corps_Verlet as gVerlet
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

    masses = np.array(masses, dtype=np.float64)
    positions = np.array(positions, dtype=np.float64)
    vitesses = np.array(vitesses, dtype=np.float64)

    t1 = time.time()
    positions, vitesses = gVerlet.update_pos_v(positions,vitesses,masses,dt)
    t2 = time.time()
    print(f"Execution time : {t2 - t1} s")
    return positions

luminosities = np.random.uniform(0.3, 1.0, n_stars+1).astype(np.float32)
bounds = ((-3, 3), (-3, 3), (-3, 3))
visualizer = vz.Visualizer3D(positions, colors, luminosities, bounds)
visualizer.run(update)