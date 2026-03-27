import numpy as np
import galaxy_generator as gg
import visualizer3d_vbo as vz
import time

class Corps():
    def __init__(self, masse, position, vitesse, couleur):
        self.masse = masse
        self.couleur = couleur
        self.position = np.array(position, dtype=np.float32)
        self.vitesse = np.array(vitesse, dtype=np.float32)
    
    def maj_pos_v(self, a, dt): 
        """
        mise à jour de la position et de la vitesse à partir de 
        l'accélération et d'un pas de temps

        """
        self.position = self.position + dt*self.vitesse + (dt**2)*a/2
        self.vitesse = self.vitesse + dt*a
        return self.position, self.vitesse

    def distance(self, other): 
        """
        calcule la distance entre deux corps

        """
        return np.linalg.norm(self.position - other.position)
   
        
class NCorps():
    def __init__(self, collect_corps):
        self.collect_corps = collect_corps

    def acc(self):
        """
        renvoie un tableau qui contient l'accélération de chaque corps 
        de la galaxie
        
        """
        G = 1.560339*1e-13
        a = []
        for self_corps in self.collect_corps: 
            a_i = np.zeros(3)
            for self_corps2 in self.collect_corps:
                diff = self_corps2.position - self_corps.position
                norm_diff = np.linalg.norm(diff)
                if norm_diff > 1e-10: 
                    a_i += G * self_corps2.masse * diff / norm_diff**3
            a.append(a_i) 
        return np.array(a) 

def step(dt):
        t1 = time.time()
        acc = Ncorps.acc()
        t2 = time.time()
        new_pos=[]
        nb_corps = len(Ncorps.collect_corps)
        for i in range(nb_corps):
            position, vitesse = Ncorps.collect_corps[i].maj_pos_v(acc[i], dt)
            new_pos.append(position)
        t3 = time.time()
        print(f"Temps de calcul accélération : {t2 - t1} s")
        print(f"Temps de calcul maj positions, vitesses : {t3 - t2} s")
        print(f"Execution time : {t3 - t1} s")
        return new_pos


if __name__ == '__main__':
  
    n_stars = 100 #nb d'étoiles
    masses, positions, velocities, colors = gg.generate_galaxy(n_stars, 
                   black_hole_mass=None,
                   star_mass_range=(0.5, 10.0),
                   min_orbital_radius=0.001,
                   max_orbital_radius=1.0,
                   output_file=None) #on génère n_stars étoiles
    
    Ncorps = NCorps([Corps(masses[i], positions[i], velocities[i], colors[i]) for i in range(n_stars + 1)]) #collection de corps à partir des données générées
    
    luminosities = np.random.uniform(0.3, 1.0, n_stars+1).astype(np.float32)
    bounds = ((-3, 3), (-3, 3), (-3, 3)) #définition des limites de l'espace 
    visualizer = vz.Visualizer3D(positions, colors, luminosities, bounds)
    visualizer.run(step)

#temps d'éxécution : ~ 0.1 s
#temps de calcul maj positions, vitesses : ~ 0.002 s
#temps de calcul accélération : ~ 0.1 s


