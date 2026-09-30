"""
PSO optimization of cubic clamped B-spline control for Chord using para_control_cub

Decision vector (size = 5 control parameters): chord_con = [
  chord_con[0],  # R position of 2nd control point (absolute radius)
  chord_con[1],  # fraction for 3rd control point: R3 = R2 + chord_con[1]*(R7 - R2)
  chord_con[2],  # chord value at root (first control point)
  chord_con[3],  # delta from root chord for 2nd control point
  chord_con[4]   # delta from root chord for 3rd control point
]

Objective (problem_id = 14 in get_test_problem.py):
  Minimize sum((Chord_spline(R) - Chord_original(R))^2) over R in sampled domain
"""

import numpy as np
import matplotlib.pyplot as plt
from get_test_problem import get_test_problem
from para_control_cub import para_control_cub
from get_test_problem import get_test_problem

Pitch_original = lambda x: 1 * (19344.5071 * x**12 + -114044.8587 * x**11 + 280789.2801 * x**10 + -357377.6146 * x**9 + 207947.2705 * x**8 + 43330.8173 * x**7 + -173099.4797 * x**6 + 143116.5772 * x**5 + -65570.9523 * x**4 + 18410.2055 * x**3 + -3128.7530 * x**2 + 294.0671 * x + -10.4419)
ChordLength_original = lambda x: 1 * (-143202.4761 * x**12 + 978274.9902 * x**11 + -2992184.0323 * x**10 + 5408923.9625 * x**9 + -6424276.8851 * x**8 + 5271614.6993 * x**7 + -3058632.5267 * x**6 + 1261908.7720 * x**5 + -366745.1423 * x**4 + 73096.4455 * x**3 + -9470.1908 * x**2 + 716.1619 * x + -23.7157)
R_values = np.concatenate([np.arange(0.17, 0.98, 0.018), [0.99, 0.999]])

# PSO configuration: 5 control parameters for chord
num_var = 5

# Build bounds informed by domain and original chord
R1 = float(R_values[0]); R7 = float(R_values[-1])
root_chord = float(ChordLength_original(R1))

bounds = np.array([
    [0.3, 0.9],  # r2_abs
    [0.05, 0.95],            # frac23
    [0.2, 0.4],  # chord0
    [0.0, 0.8],               # d1
    [0.0, 0.8 ],               # d2
])

swarm_size = 50
num_iter = 100
C1, C2 = 2.05, 2.05
C = C1 + C2
PSO_variant = 1  # 1: constriction factor; 2: inertia + Vmax
w_max, w_min = 0.9, 0.5
Contraint_handling = 2  # penalty
craziness = 0


# Derived helpers
Vmax = 0.2 * (bounds[:, 1] - bounds[:, 0])
Vmax_col = Vmax[:, None]
Vmax_1d = Vmax

if PSO_variant == 1:
    X = 2 / (C - 2 + np.sqrt(C**2 - 4 * C))
    w = 1.0
    vel = -Vmax_col + 2 * Vmax_col * np.random.rand(num_var, swarm_size)
else:
    X = 1.0
    vel = -Vmax_col + 2 * Vmax_col * np.random.rand(num_var, swarm_size)

# Initialize positions
lower_col = bounds[:, 0][:, None]
range_col = (bounds[:, 1] - bounds[:, 0])[:, None]
pos = lower_col + range_col * np.random.rand(num_var, swarm_size)

pitch_con = np.zeros(6)  # pitch not optimized here

R_plot = np.linspace(R_values[0], R_values[-1], 200)
plt.figure()
plt.plot(R_plot, ChordLength_original(R_plot), 'b', linewidth=1.4, label='Original chord')

for k in range(5):
    # Fresh RNG and initial state for each run
    # np.random.seed(None)  # optional: rely on default entropy
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
        # Penalty (no inequality constraints provided; keep structure)
        penalty_val = np.zeros(swarm_size)

        # Evaluate fitness for each particle using problem 14 (chord cubic B-spline)
        fitness = np.zeros(swarm_size)
        for i in range(swarm_size):
            var = pos[:, i]
            # Delegate objective to centralized problem definition
            fitness[i] = get_test_problem(14, 2, pitch_con, var)

        avg_fitness = float(np.mean(fitness))
        avg_fitness_per_iter[t - 1] = avg_fitness

        # Apply penalty (zeros by default)
        fitness = fitness + np.abs(5 * avg_fitness * penalty_val)

        # Iteration best
        iter_indx = int(np.argmin(fitness))
        iter_best = float(fitness[iter_indx])

        # Update pbest
        for i in range(swarm_size):
            if t == 1 or fitness[i] < pbest[i]:
                pbest[i] = fitness[i]
                pbest_pos[:, i] = pos[:, i]

        # Update gbest
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

        # Velocity/position update
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

    # Results
    print(f'Best Function Value Found: {gbest}')
    print('Best Position Found (5 chord control params):', end=' ')
    print(*gbest_pos, sep=' ')
    if gbest_pos.shape[0] == 5:
        r2, frac_r3, chord_root, delta1, delta2 = gbest_pos
        print(f'  R2={r2:.6f}, frac_R3={frac_r3:.6f}, chord_root={chord_root:.6f}, delta1={delta1:.6f}, delta2={delta2:.6f}')
        # store result from all 10 iterations in a file
        with open('chord_opt_cub.txt', 'a') as f:
            f.write(f'{r2:.6f} {frac_r3:.6f} {chord_root:.6f} {delta1:.6f} {delta2:.6f}\n')
        print(f'Iteration {k} completed')

    # Plot this run's optimized curve on the same figure
    optimized_pitch, optimized_chord = para_control_cub(
        Pitch_original,
        ChordLength_original,
        R_values,
        2,
        np.zeros(6),
        gbest_pos,   # chord_con is 5 control parameters expected by para_control_cub
    )
    plt.plot(R_plot, optimized_chord(R_plot), '-', linewidth=1.0, alpha=0.7, label=f'Opt run {k}')

# Finalize combined plot
plt.xlabel('R')
plt.ylabel('Chord')
plt.grid(True)
plt.legend()
plt.title('Chord: Original vs optimized (cubic B-spline) across runs')
plt.show()

