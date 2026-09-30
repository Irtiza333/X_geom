"""
This is the interpretation of my basic understanding of PSO. 
There are confusion with the basic version including contriction factor 
and the inertia. This also involves the concept of Vmax, how to handle 
and bound it.
                                                Irtiza Khan
                                                Date started: May 05, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from get_test_problem import get_test_problem
from para_control_bez import para_control_bez 

# Clear workspace equivalent
# rng default equivalent
# np.random.seed(42)

# Get test problem info
chord_con = np.zeros(5)
pitch_con = np.zeros(6)
# obj_func, num_var, bounds, constrain_func, constrain_bound, best_known_sol, best_known_val = get_test_problem(11,3,pitch_con,chord_con)
num_var = 6                             # number of variables
bounds = np.array([[0.8, 1.25], #p1y 
                    [0.05, 1.0], #y4
                    [0.35, 0.75], #p4x
                    [0.15, 0.5], #d1
                    [0.15, 0.5], #d2
                    [0.4, 0.7]])  #p7y
best_known_sol = []                        # location of the global maximum
best_known_val = 0                             # maximum value of func
constrain_func = []                            # no constraints
constrain_bound = []    
                       # location of the global maximum
swarm_size = 100 
                                                           # number of particles in swarm
num_iter = 200                                                              # number of iteration                                                                 
C1 = 2.05                                                                   # Pbest coefficient 
C2 = 2.05                                                                   # Gbest coefficient
C = C1 + C2
PSO_variant = 1                                                             # PSO variant => 
                                                                             # 1: constriction factor 
                                                                             # 2: intertia & Vmax

w_max = 0.9                                                                 # inertia range for dynamic inertia                                                   
w_min = 0.5


Contraint_handling = 2                                                      # Contraint handling technique 
                                                                             # 0 => no constraints 
                                                                             # 1 => rejecttion 
                                                                             # 2 => penalty
craziness = 0

# Initialize velocity, constriction factor, & intertia 
Vmax = 0.2 * (bounds[:, 1] - bounds[:, 0])            # shape: (num_var,)
Vmax_col = Vmax[:, None]                               # shape: (num_var, 1) for broadcasting
Vmax_1d = Vmax                                         # ensure 1-D vector for per-particle clamping

# Prepare overlay figure: plot original once, then add all optimized runs
Pitch_original = lambda x: 1 * (19344.5071 * x**12 + -114044.8587 * x**11 + 280789.2801 * x**10 + -357377.6146 * x**9 + 207947.2705 * x**8 + 43330.8173 * x**7 + -173099.4797 * x**6 + 143116.5772 * x**5 + -65570.9523 * x**4 + 18410.2055 * x**3 + -3128.7530 * x**2 + 294.0671 * x + -10.4419)
ChordLength_original = lambda x: 1 * (-143202.4761 * x**12 + 978274.9902 * x**11 + -2992184.0323 * x**10 + 5408923.9625 * x**9 + -6424276.8851 * x**8 + 5271614.6993 * x**7 + -3058632.5267 * x**6 + 1261908.7720 * x**5 + -366745.1423 * x**4 + 73096.4455 * x**3 + -9470.1908 * x**2 + 716.1619 * x + -23.7157)
R_values = np.concatenate([np.arange(0.17, 0.98, 0.018), [0.99, 0.999]])
R_plot = np.linspace(R_values[0], R_values[-1], 200)
plt.figure()
plt.plot(R_plot, Pitch_original(R_plot), 'b', linewidth=1.4, label='Original pitch')


if PSO_variant == 1:                                                        # with the constriction factor effect                                                                
    X = 2/(C-2+np.sqrt(C**2-4*C))                                          # contriction factor
    w = 1                                                                   # inertia

    # Initialize random velocities within variable limits
         # velocity initialized with range a of 20% of the variable dynamic range

elif PSO_variant == 2:                                                      # without the constriction factor effect 
    X = 1
  
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
    counter = 0                                # track the average func. val. per iter.

    # Main loop
    while t <= num_iter:

        # REJECTION 
        if Contraint_handling == 1:
            for i in range(swarm_size):
                var = pos[:, i]

                # check if particle violates any boundary
                while not all(constrain_func[j](var) <= constrain_bound[j] for j in range(len(constrain_func))):

                    # generate new particle within bounds
                    var = (bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * \
                        np.random.rand(num_var))

                # accept the newly found valid point as part of the swarm
                pos[:, i] = var

        # PENALTY
        penalty_val = np.zeros(swarm_size)                                      # container for penalty values

        if Contraint_handling == 2:
            for i in range(swarm_size):
                var = pos[:, i]

                # Calculate penalty value
                def penalty_func(var):
                    return sum(max(0, constrain_func[j](var) - constrain_bound[j]) for j in range(len(constrain_func)))
                
                penalty_val[i] = penalty_func(var)                              # assign the value as penalty value for that particle

        # loop to find function/ fitness value
        fitness = np.zeros(swarm_size)                                          # container for fitness values

        for i in range(swarm_size):
            var = pos[:, i]  # evaluate current particle
            fitness[i] = get_test_problem(11, 3, var, chord_con)                                    # evaluate fitness values

        # Calculate avg fitness for the iteration & store
        avg_fitness = np.sum(fitness)/swarm_size                                # average function values for this iteration

        avg_fitness_per_iter[t-1] = np.sum(fitness)/swarm_size                  # store avearge func. value per iteration

        for i in range(swarm_size):
            fitness[i] = fitness[i] + abs(5*avg_fitness * penalty_val[i])      # evaluate modified fitness values based on penalty

        # Calculate best fitness for the iteration                      
        iter_best = np.min(fitness)                                             # calculate best value and index of this iteration
        iter_indx = np.argmin(fitness)

        # loop to update personal best value and position
        for i in range(swarm_size):
            # intially set pbest to the 1st iteration values
            if t == 1:
                pbest[i] = fitness[i]                                           
                pbest_pos[:, i] = pos[:, i]

            # for next iterations check if value gets better
            else:
                if fitness[i] < pbest[i]:                                       # if gets better (lower)
                    pbest[i] = fitness[i]                                       # update pbest for each particle
                    pbest_pos[:, i] = pos[:, i]                                 # update pbest position for each particle

        # finding and updating global best value and position
        if t == 1:
            gbest = iter_best                                           
            gbest_pos = pos[:, iter_indx]                                       # set gbest position to the best particle's position
            counter = 0

        # for next iterations check if value gets better
        else:
            if iter_best < gbest:                                         
                gbest = iter_best                                       
                gbest_pos = pos[:, iter_indx]                                   # set gbest position to the best particle's position
                counter = 0
            else: 

                # stopping criteria based on improvement  
                counter = counter + 1
                if counter > 50:
                    print(f'No improvement for 50 iters; stopping at t = {t}')
                    gbest_track = gbest_track[:t-1]                               # trim the unused entries 
                    avg_fitness_per_iter = avg_fitness_per_iter[:t-1]              # trim the unused entries
                    break

        gbest_track[t-1] = gbest

        # update time
        t = t + 1                                                                 

        # calculate new velocity and new position
        # if inertia effect is being used, then reduce intertia with iteration
        if PSO_variant == 2:
            w = w_max - w_min*(t/num_iter)                                      # update inertia with time

        r1 = np.random.rand(num_var, swarm_size)                                # random factor for C1
        r2 = np.random.rand(num_var, swarm_size)                                # random factor for C2

        if t < num_iter:                                                        # only update for the number of iteration
            for i in range(swarm_size): 
                new_vel[:, i] = X*(w*vel[:, i] + (C1*r1[:, i]) * \
                    (pbest_pos[:, i] - pos[:, i]) + (C2*r2[:, i]) * \
                    (gbest_pos - pos[:, i]))                                     # update velocity

                # Add craziness effect based on settings
                if craziness == 1:
                    new_vel[:, i] = X*(w*vel[:, i] + (C1*r1[:, i]) * \
                    (pbest_pos[:, i] - pos[:, i]) + (C2*r2[:, i]) * \
                    (gbest_pos - pos[:, i]) + vel[:, i] * np.random.rand(num_var))

                # Clamp velocities using Vmax (1-D per-dimension limits)
                new_vel[:, i] = np.maximum(np.minimum(new_vel[:, i], Vmax_1d), -Vmax_1d)
        
                pos[:, i] = pos[:, i] + new_vel[:, i]                           # update position

                # Clamp particle positions using bounds
                pos[:, i] = np.maximum(np.minimum(pos[:, i], bounds[:, 1]), bounds[:, 0])

            vel = new_vel.copy()                                                 # setting new velocity as the prev. velocity for the next iter.

    # Plots and Display

    print(f'Best Function Value Found: {gbest}')
    print('Best Position Found:', end=' ')
    print(*gbest_pos, sep=' ')
    if gbest_pos.shape[0] == 6:
        p1y, y4, p4x, d1, d2, p7y = gbest_pos
        print(f'  p1y = {p1y:.6f}, y4 = {y4:.6f}, p4x = {p4x:.6f}, d1 = {d1:.6f}, d2 = {d2:.6f}, p7y = {p7y:.6f}')

        # store result from all 10 iterations in a file
        with open('pitch_opt_bez_final.txt', 'a') as f:
            f.write(f'{p1y:.6f} {y4:.6f} {p4x:.6f} {d1:.6f} {d2:.6f} {p7y:.6f}\n')
        print(f'Iteration {k} completed')

        # Add this run to the overlay plot
        optimized_pitch, _, *_ = para_control_bez(Pitch_original, ChordLength_original, R_values, 3, gbest_pos, np.zeros(6))
        plt.plot(R_plot, optimized_pitch(R_plot), '-', linewidth=1.0, alpha=0.7, label=f'Opt run {k}')

# Finalize combined plot
plt.xlabel('R')
plt.ylabel('Pitch')
plt.grid(True)
plt.legend()
plt.title('Pitch: Original vs optimized (Bézier) across runs')
plt.show()

"""
    Modifications

    May 07, 2025 - Completed basic code: contriction factor or Vmax, 
                & inertia, Particle generation - random or equally 
                distributed in all dimensions. 
                - Clamping or not? - need to test, graphs for visulization 
                - tracking Global best (good for now)

    May 08, 2025 - Added in constraint handling: rejection & penalty. 
                Rejection struggles for problem 3.1 (too many constraints)  
                but other problems work. 
                - Penalty works for all the functions pretty 
                well. 3.1 is sometimes struggling with just avg_fitness *
                penalty. 5 * avg_fitness * penalty seems to work
                consistantly.

    May 09, 2025 - Added in velocity and position clamping, otherwise particle
                leave bounds for concave functions.
                - Added average fitness value per iteration tracking so see
                if optimizer is actually converging the particles. 
                - What other tracking should be included for the parametric
                analysis 
                - Need to add more comments to make code clearer.

    May 12, 2025 - Added in comments, should be enough. 
                - Added craziness
    """
