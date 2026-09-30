"""
PSO for chord optimization using rational Bézier controls.

We reuse the same PSO skeleton as in basic_pso.py, but optimize the
six chord design variables defined in para_control_bez.py:
  [p1y, p4x, d1, y4, w23, w56]

Objective (problem 12 in get_test_problem.py):
  Minimize sum((Chord(R) - Chord_original(R))^2) over R in [0.17, 1]
"""

import numpy as np
import matplotlib.pyplot as plt
from get_test_problem import get_test_problem
from para_control_bez import para_control_bez


# Design variable counts
num_var = 5  # chord variables: [p1y, p4x, d1, y4, w23, w56]

# Bounds for chord variables
# p1y, y4 are chord magnitudes (normalized) → assume [0.05, 0.6]
# p4x is a radius location between root and tip → [0.25, 0.9]
# d1 controls spacing (distance along R) → [0.05, 0.5]
# w23, w56 are rational weights → [0, 1]
bounds = np.array([
    [0.2, 0.3],   # p1y
    [0.45, 0.85],   # p4x
    # [0.05, 0.45],   # d1
    [0.15, 0.5],
    [0.4, 0.7],   # y4
    # [0.0, 1.0],    # w23
    [0.1, 0.3],    # w56
])

# Empty constraints (we use bounds only)
constrain_func = []
constrain_bound = []

# PSO hyperparameters
swarm_size = 100
num_iter = 150
C1 = 2.05
C2 = 2.05
C = C1 + C2
PSO_variant = 1  # 1: constriction factor, 2: inertia + Vmax
w_max = 0.9
w_min = 0.5
Particle_Gen_method = 1
spacing_per_dim = 5
swarm_size_eq_space = spacing_per_dim ** num_var
Contraint_handling = 2  # 0: none, 1: rejection, 2: penalty
craziness = 0

# Helpers
Vmax = 0.2 * (bounds[:, 1] - bounds[:, 0])
Vmax_col = Vmax[:, None]
Vmax_1d = Vmax

if PSO_variant == 1:
    X = 2 / (C - 2 + np.sqrt(C**2 - 4 * C))
    w = 1
    vel = -Vmax_col + 2 * Vmax_col * np.random.rand(num_var, swarm_size)
else:
    X = 1
    vel = -Vmax_col + 2 * Vmax_col * np.random.rand(num_var, swarm_size)

pitch_con = np.zeros(7)  # pitch is not optimized here

# Prepare overlay figure: original once, then overlay optimized curves across runs
Pitch_original = lambda x: 1 * (19344.5071 * x**12 + -114044.8587 * x**11 + 280789.2801 * x**10 + -357377.6146 * x**9 + 207947.2705 * x**8 + 43330.8173 * x**7 + -173099.4797 * x**6 + 143116.5772 * x**5 + -65570.9523 * x**4 + 18410.2055 * x**3 + -3128.7530 * x**2 + 294.0671 * x + -10.4419)
ChordLength_original = lambda x: 1 * (-143202.4761 * x**12 + 978274.9902 * x**11 + -2992184.0323 * x**10 + 5408923.9625 * x**9 + -6424276.8851 * x**8 + 5271614.6993 * x**7 + -3058632.5267 * x**6 + 1261908.7720 * x**5 + -366745.1423 * x**4 + 73096.4455 * x**3 + -9470.1908 * x**2 + 716.1619 * x + -23.7157)
R_values = np.concatenate([np.arange(0.17, 0.98, 0.018), [0.99, 0.999]])
R_plot = np.linspace(R_values[0], R_values[-1], 200)
plt.figure()
plt.plot(R_plot, ChordLength_original(R_plot), 'b', linewidth=1.4, label='Original chord')

for k in range(3):
    # fresh initialization per run
    vel = -Vmax_col + 2 * Vmax_col * np.random.rand(num_var, swarm_size)
    lower_col = bounds[:, 0][:, None]
    range_col = (bounds[:, 1] - bounds[:, 0])[:, None]
    pos = lower_col + range_col * np.random.rand(num_var, swarm_size)
    new_vel = vel.copy()
    t = 1

    # Containers
    pbest = np.zeros(swarm_size)
    pbest_pos = np.zeros((num_var, swarm_size))
    gbest_track = np.zeros(num_iter)
    avg_fitness_per_iter = np.zeros(num_iter)

    while t <= num_iter:
        # Penalty setup
        penalty_val = np.zeros(swarm_size)
        
        if Contraint_handling == 2:
            for i in range(swarm_size):
                var = pos[:, i]
                def penalty_func(v):
                    return sum(max(0, constrain_func[j](v) - constrain_bound[j]) for j in range(len(constrain_func)))
                penalty_val[i] = penalty_func(var)

        # Evaluate fitness for each particle using problem 12 (chord)
        fitness = np.zeros(swarm_size)
        for i in range(swarm_size):
            var = pos[:, i]
            fitness[i] = get_test_problem(12, 2, pitch_con, var)

        avg_fitness = np.sum(fitness) / swarm_size
        avg_fitness_per_iter[t - 1] = avg_fitness

        for i in range(swarm_size):
            fitness[i] = fitness[i] + abs(5 * avg_fitness * penalty_val[i])

        # Iteration best
        iter_best = np.min(fitness)
        iter_indx = np.argmin(fitness)

        # Update pbest
        for i in range(swarm_size):
            if t == 1:
                pbest[i] = fitness[i]
                pbest_pos[:, i] = pos[:, i]
            else:
                if fitness[i] < pbest[i]:
                    pbest[i] = fitness[i]
                    pbest_pos[:, i] = pos[:, i]

        # Update gbest
        if t == 1:
            gbest = iter_best
            gbest_pos = pos[:, iter_indx]
            counter = 0
        else:
            if iter_best < gbest:
                gbest = iter_best
                gbest_pos = pos[:, iter_indx]
                counter = 0
            else:
                counter += 1
                if counter > 50:
                    print(f'No improvement for 50 iters; stopping at t = {t}')
                    gbest_track = gbest_track[: t - 1]
                    avg_fitness_per_iter = avg_fitness_per_iter[: t - 1]
                    break

        gbest_track[t - 1] = gbest

        # Update velocity/position
        t += 1
        if PSO_variant == 2:
            w = w_max - w_min * (t / num_iter)

        r1 = np.random.rand(num_var, swarm_size)
        r2 = np.random.rand(num_var, swarm_size)
        if t < num_iter:
            for i in range(swarm_size):
                new_vel[:, i] = X * (
                    w * vel[:, i]
                    + (C1 * r1[:, i]) * (pbest_pos[:, i] - pos[:, i])
                    + (C2 * r2[:, i]) * (gbest_pos - pos[:, i])
                )
                if craziness == 1:
                    new_vel[:, i] = X * (
                        w * vel[:, i]
                        + (C1 * r1[:, i]) * (pbest_pos[:, i] - pos[:, i])
                        + (C2 * r2[:, i]) * (gbest_pos - pos[:, i])
                        + vel[:, i] * np.random.rand(num_var)
                    )

                # Clamp velocities and positions
                new_vel[:, i] = np.maximum(np.minimum(new_vel[:, i], Vmax_1d), -Vmax_1d)
                pos[:, i] = pos[:, i] + new_vel[:, i]
                pos[:, i] = np.maximum(np.minimum(pos[:, i], bounds[:, 1]), bounds[:, 0])

            vel = new_vel.copy()

    # After run k: print and overlay optimized chord
    print(f'Best Function Value Found: {gbest}')
    print('Best Position Found:', end=' ')
    print(*gbest_pos, sep=' ')
    if gbest_pos.shape[0] == 5:
        p1y, p4x, d1, y4, w56 = gbest_pos
        print(f'  p1y = {p1y:.6f}, p4x = {p4x:.6f}, d1 = {d1:.6f}, y4 = {y4:.6f}, w56 = {w56:.6f}')
        # store
        with open('chord_opt_bez_final2.txt', 'a') as f:
            f.write(f'{p1y:.6f} {p4x:.6f} {d1:.6f} {y4:.6f} {w56:.6f}\n')
    optimized_pitch, optimized_chord, chord_con_points, pitch_con_points = para_control_bez(Pitch_original, ChordLength_original, R_values, 2, pitch_con, gbest_pos)
    plt.plot(R_plot, optimized_chord(R_plot), '-', linewidth=1.0, alpha=0.7, label=f'Opt run {k}')

# Finalize overlay
plt.xlabel('R')
plt.ylabel('Chord')
plt.grid(True)
plt.legend()
plt.title('Chord: Original vs optimized (Bézier) across runs')
plt.show()


