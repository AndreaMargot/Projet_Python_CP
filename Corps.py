import numpy as np
import galaxy_generator as gg
import time

class Corps():
    def __init__(self, masse, position, vitesse, couleur):
        self.masse = masse
        self.couleur = couleur
        self.position = np.array(position, dtype=np.float32)
        self.vitesse = np.array(vitesse, dtype=np.float32)
    
    def maj_v_a(self, a, dt): #maj de la position et vitesse à partir de l'accélération et un pas de temps
        self.position = self.position + dt*self.vitesse + (dt**2)*np.array(a)/2
        self.vitesse = self.vitesse + dt*np.array(a)
        return self.position, self.vitesse

    def distance(self, other):
        return np.linalg.norm(self.position - other.position)
   
        
class NCorps():
    def __init__(self, collect_corps):
        self.collect_corps = collect_corps

    def f_att(self): #renvoie l'accelération
        G = 1.560339*1e-13
        f = []
        for self_corps in self.collect_corps: #force d'attraction du corps envers lui-même
            f_i = 0
            for self_corps2 in self.collect_corps:
                diff = self_corps2.position - self_corps.position
                norm_diff = np.linalg.norm(diff)
                if norm_diff > 1e-10: 
                    f_i += G * self_corps2.masse * diff / norm_diff**3
            f.append(f_i) 
        return np.array(f) 

def step(dt):
        t1 = time.time()
        acc = Ncorps.f_att()
        new_pos=[]
        for i in range(len(Ncorps.collect_corps)):
            position, vitesse = Ncorps.collect_corps[i].maj_v_a(acc[i], dt)
            new_pos.append(position)
        t2 = time.time()
        print(f"Execution time : {t2 - t1} s")
        return new_pos


if __name__ == '__main__':

    import visualizer3d_vbo as vz  
    n_stars = 100 #nb d'étoiles
    masses, positions, velocities, colors = gg.generate_galaxy(n_stars, 
                   black_hole_mass=None,
                   star_mass_range=(0.5, 10.0),
                   min_orbital_radius=0.001,
                   max_orbital_radius=1.0,
                   output_file=None) #on génère n étoiles
    
    Ncorps = NCorps([Corps(masses[i], np.array(positions[i]), np.array(velocities[i]), colors[i]) for i in range(n_stars)]) #collection de corps à partir des données générées
    
    luminosites = np.random.uniform(0.3, 1.0, n_stars+1).astype(np.float32)
    bounds = ((-3, 3), (-3, 3), (-3, 3)) #bornes...
    a = Ncorps.f_att() #accelération
    visualizer = vz.Visualizer3D(positions, colors, luminosites, bounds)
    visualizer.run(step)

#temps d'execution : ~ 0.1 s
    


