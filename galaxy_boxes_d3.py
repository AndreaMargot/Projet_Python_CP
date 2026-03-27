import visualizer3d_vbo as visual
import galaxy_generator as generator
import numpy as np
import numba
import time

N_BOXES = 10
THETA = 0.5

n_stars = 200
masses, positions, vitesses, colors = generator.generate_galaxy(n_stars=n_stars)


def build_grid(positions, masses, n_boxes=N_BOXES):
    """
    Partitionne l'espace en une grille 2D de n_boxes x n_boxes.
    Retourne les structures de données nécessaires au calcul de l'accélération.
    """
    n_bodies = len(positions)

    x_min = np.min(positions[:, 0])
    x_max = np.max(positions[:, 0])
    y_min = np.min(positions[:, 1])
    y_max = np.max(positions[:, 1])

    Lx = x_max - x_min
    Ly = y_max - y_min

    dx = Lx / n_boxes if Lx > 1e-15 else 1.0
    dy = Ly / n_boxes if Ly > 1e-15 else 1.0

    # Assigner chaque corps à une boîte (vectorisé)
    ix = np.clip(((positions[:, 0] - x_min) / dx).astype(np.int64), 0, n_boxes - 1)
    iy = np.clip(((positions[:, 1] - y_min) / dy).astype(np.int64), 0, n_boxes - 1)

    # Compter le nombre de corps par boîte
    box_counts = np.zeros((n_boxes, n_boxes), dtype=np.int64)
    for b in range(n_bodies):
        box_counts[iy[b], ix[b]] += 1

    # Calculer les offsets (début de chaque boîte dans le tableau plat)
    box_offsets = np.zeros((n_boxes, n_boxes), dtype=np.int64)
    offset = 0
    for by in range(n_boxes):
        for bx in range(n_boxes):
            box_offsets[by, bx] = offset
            offset += box_counts[by, bx]

    # Remplir le tableau des indices de corps par boîte
    box_contents = np.empty(n_bodies, dtype=np.int64)
    temp_offsets = box_offsets.copy()
    for b in range(n_bodies):
        by, bx = iy[b], ix[b]
        box_contents[temp_offsets[by, bx]] = b
        temp_offsets[by, bx] += 1

    # Calculer le centre de gravité de chaque boîte
    box_centers = np.zeros((n_boxes, n_boxes, 3), dtype=np.float64)
    box_masses = np.zeros((n_boxes, n_boxes), dtype=np.float64)

    for by in range(n_boxes):
        for bx in range(n_boxes):
            start = box_offsets[by, bx]
            count = box_counts[by, bx]
            if count > 0:
                idx = box_contents[start:start + count]
                m_total = np.sum(masses[idx])
                box_centers[by, bx] = np.sum(
                    masses[idx, np.newaxis] * positions[idx], axis=0
                ) / m_total
                box_masses[by, bx] = m_total

    box_size = np.sqrt(dx**2 + dy**2)

    return box_contents, box_offsets, box_counts, box_centers, box_masses, box_size


@numba.njit(parallel=True)
def acceleration(positions, masses, box_contents, box_offsets, box_counts,
                 box_centers, box_masses, box_size, theta, n_boxes):
    """
    Calcule l'accélération gravitationnelle avec la méthode en boîtes
    (Barnes-Hut simplifié sur grille 2D).

    - Champ lointain : approximation par le centre de gravité de la boîte
    - Champ proche : calcul direct étoile par étoile
    """
    n_bodies = len(positions)
    acc = np.zeros((n_bodies, 3), dtype=np.float64)
    G = 1.560339e-13

    for i in numba.prange(n_bodies):
        for by in range(n_boxes):
            for bx in range(n_boxes):
                cm = box_masses[by, bx]
                if cm == 0.0:
                    continue

                # Distance au centre de gravité de la boîte
                dx = box_centers[by, bx, 0] - positions[i, 0]
                dy = box_centers[by, bx, 1] - positions[i, 1]
                dz = box_centers[by, bx, 2] - positions[i, 2]
                dist = np.sqrt(dx * dx + dy * dy + dz * dz)

                if dist > 1e-10 and box_size / dist < theta:
                    # Champ lointain : approximation centre de gravité
                    inv_dist3 = 1.0 / (dist * dist * dist)
                    acc[i, 0] += G * cm * dx * inv_dist3
                    acc[i, 1] += G * cm * dy * inv_dist3
                    acc[i, 2] += G * cm * dz * inv_dist3
                else:
                    # Champ proche : calcul direct
                    start = box_offsets[by, bx]
                    count = box_counts[by, bx]
                    for k in range(count):
                        j = box_contents[start + k]
                        if j != i:
                            djx = positions[j, 0] - positions[i, 0]
                            djy = positions[j, 1] - positions[i, 1]
                            djz = positions[j, 2] - positions[i, 2]
                            norm = np.sqrt(djx * djx + djy * djy + djz * djz)
                            if norm > 1e-10:
                                inv_norm3 = 1.0 / (norm * norm * norm)
                                acc[i, 0] += G * masses[j] * djx * inv_norm3
                                acc[i, 1] += G * masses[j] * djy * inv_norm3
                                acc[i, 2] += G * masses[j] * djz * inv_norm3

    return acc


def update_pos_v(positions, vitesses, acc, dt):
    """Mise à jour des positions et vitesses (schéma d'Euler)."""
    positions = positions + dt * vitesses + 0.5 * dt**2 * acc
    vitesses = vitesses + dt * acc
    return positions, vitesses


def update(dt):
    global positions, vitesses, masses

    masses = np.array(masses, dtype=np.float64)
    positions = np.array(positions, dtype=np.float64)
    vitesses = np.array(vitesses, dtype=np.float64)

    t1 = time.time()

    box_contents, box_offsets, box_counts, box_centers, box_masses, box_size = \
        build_grid(positions, masses, N_BOXES)

    acc = acceleration(positions, masses, box_contents, box_offsets, box_counts,
                       box_centers, box_masses, box_size, THETA, N_BOXES)

    t2 = time.time()

    positions, vitesses = update_pos_v(positions, vitesses, acc, dt)

    t3 = time.time()
    print(f"temps calcul accélération : {t2 - t1:.4f} s")
    print(f"maj : {t3 - t2:.4f} s")
    print(f"Execution time : {t3 - t1:.4f} s")

    return positions


luminosities = np.random.uniform(0.3, 1.0, n_stars + 1).astype(np.float32)
bounds = ((-3, 3), (-3, 3), (-3, 3))
visualizer = visual.Visualizer3D(positions, colors, luminosities, bounds)
visualizer.run(update)
