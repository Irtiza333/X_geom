"""
PSO optimization of cubic clamped B-spline control for Pitch using para_control_cub

Decision vector (size = 6): pitch_con = [
  r2,            # absolute radius of control point 2
  frac23,        # fraction in (0,1) to place r3 between r2 and tip
  pitch0,        # pitch at root control point (baseline)
  d1,            # increment for pitch at r2 (pitch0 + d1)
  d2,            # increment for pitch at r3 (pitch0 + d2)
  tip_frac       # tip pitch = pitch0 - pitch0 * tip_frac
]

Objective (problem_id = 13 in get_test_problem.py):
  Minimize sum((Pitch_spline(R) - Pitch_original(R))^2) over R in sampled domain
"""

import numpy as np
import matplotlib.pyplot as plt
from get_test_problem import get_test_problem
from para_control_cub import para_control_cub


# PSO configuration
num_var = 6
bounds = np.array([
    [0.20, 0.8],  # r2
    [0.05, 0.95],  # frac23
    [0.50, 1.3],  # pitch0
    [0.1, 0.70],  # d1
    [0.1, 0.80],  # d2
    [0.3, 0.95],  # tip_frac
])

swarm_size = 50
num_iter = 100
C1, C2 = 2.05, 2.05
C = C1 + C2
PSO_variant = 1
w_max, w_min = 0.9, 0.5
Contraint_handling = 2
craziness = 0


# Derived helpers
Vmax = 0.2 * (bounds[:, 1] - bounds[:, 0])
Vmax_col = Vmax[:, None]
Vmax_1d = Vmax

if PSO_variant == 1:
    X = 2 / (C - 2 + np.sqrt(C**2 - 4 * C))
    w = 1.0
else:
    X = 1.0

# (moved into per-run loop below)

# Fixed arrays for API compatibility
chord_con = np.zeros(5)  # chord not optimized here

# Prepare overlay figure: plot original once, then add all optimized runs
Pitch_original = lambda x: 1 * (19344.5071 * x**12 + -114044.8587 * x**11 + 280789.2801 * x**10 + -357377.6146 * x**9 + 207947.2705 * x**8 + 43330.8173 * x**7 + -173099.4797 * x**6 + 143116.5772 * x**5 + -65570.9523 * x**4 + 18410.2055 * x**3 + -3128.7530 * x**2 + 294.0671 * x + -10.4419)
ChordLength_original = lambda x: 1 * (-143202.4761 * x**12 + 978274.9902 * x**11 + -2992184.0323 * x**10 + 5408923.9625 * x**9 + -6424276.8851 * x**8 + 5271614.6993 * x**7 + -3058632.5267 * x**6 + 1261908.7720 * x**5 + -366745.1423 * x**4 + 73096.4455 * x**3 + -9470.1908 * x**2 + 716.1619 * x + -23.7157)
R_values = np.concatenate([np.arange(0.17, 0.98, 0.018), [0.99, 0.999]])
R_plot = np.linspace(R_values[0], R_values[-1], 200)
plt.figure()
plt.plot(R_plot, Pitch_original(R_plot), 'b', linewidth=1.4, label='Original pitch')

for k in range(10):
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
    counter = 0

    while t <= num_iter:
        penalty_val = np.zeros(swarm_size)

        fitness = np.zeros(swarm_size)
        for i in range(swarm_size):
            var = pos[:, i]
            fitness[i] = get_test_problem(13, 3, var, chord_con)

        avg_fitness = float(np.mean(fitness))
        avg_fitness_per_iter[t - 1] = avg_fitness
        fitness = fitness + np.abs(5 * avg_fitness * penalty_val)

        iter_indx = int(np.argmin(fitness))
        iter_best = float(fitness[iter_indx])

        for i in range(swarm_size):
            if t == 1 or fitness[i] < pbest[i]:
                pbest[i] = fitness[i]
                pbest_pos[:, i] = pos[:, i]

        if t == 1:
            gbest = iter_best
            gbest_pos = pos[:, iter_indx].copy()
            counter = 0
        else:
            if iter_best < gbest:
                gbest = iter_best
                gbest_pos = pos[:, iter_indx].copy()
                counter = 0
            else:
                counter += 1
                if counter > 50:
                    print(f'No improvement for 50 iters; stopping at t = {t}')
                    gbest_track = gbest_track[: t - 1]
                    avg_fitness_per_iter = avg_fitness_per_iter[: t - 1]
                    break

        gbest_track[t - 1] = gbest

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
                new_vel[:, i] = np.maximum(np.minimum(new_vel[:, i], Vmax_1d), -Vmax_1d)
                pos[:, i] = pos[:, i] + new_vel[:, i]
                pos[:, i] = np.maximum(np.minimum(pos[:, i], bounds[:, 1]), bounds[:, 0])
            vel = new_vel.copy()

    print(f'Best Function Value Found: {gbest}')
    print('Best Position Found:', end=' ')
    print(*gbest_pos, sep=' ')
    if gbest_pos.shape[0] == 6:
        r2, frac23, pitch0, d1, d2, tip_frac = gbest_pos
        print(f'  r2 = {r2:.6f}, frac23 = {frac23:.6f}, pitch0 = {pitch0:.6f}, d1 = {d1:.6f}, d2 = {d2:.6f}, tip_frac = {tip_frac:.6f}')

        # store result from all 10 iterations in a file
        with open('pitch_opt_cub.txt', 'a') as f:
            f.write(f'{r2:.6f} {frac23:.6f} {pitch0:.6f} {d1:.6f} {d2:.6f} {tip_frac:.6f}\n')
        print(f'Iteration {k} completed')

        # Add this run to the overlay plot
        optimized_pitch, _ = para_control_cub(Pitch_original, ChordLength_original, R_values, 3, gbest_pos, np.zeros(5))
        plt.plot(R_plot, optimized_pitch(R_plot), '-', linewidth=1.0, alpha=0.7, label=f'Opt run {k}')

# Finalize combined plot
plt.xlabel('R')
plt.ylabel('Pitch')
plt.grid(True)
plt.legend()
plt.title('Pitch: Original vs optimized (cubic B-spline) across runs')
plt.show()


