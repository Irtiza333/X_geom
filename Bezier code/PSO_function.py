"""
This is the implementation of the PSO optimizer in the function form. 
This has the same structure as the basic_pso.m  
                                                Irtiza Khan
                                                Date started: May 05, 2025
"""

import numpy as np
from get_test_problem import get_test_problem

def PSO_function(swarm_size, num_iter, C1, C2, PSO_variant, w_max, w_min,
                 Particle_Gen_method, spacing_per_dim, Contraint_handling, craziness, 
                 penalty_factor, problem_id):
    """
    PSO optimizer function
    
    Returns:
        gbest: global best value
        gbest_pos: global best position
        iterations: number of iterations completed
        gbest_track: tracking of global best over iterations
        avg_fitness_per_iter: average fitness per iteration
    """
    
    obj_func, num_var, bounds, constrain_func, constrain_bound, best_known_sol, best_known_val = get_test_problem(problem_id)

    swarm_size_eq_space = spacing_per_dim**num_var
    C = C1 + C2

    # Initialize velocity, constriction factor, & intertia 
    Vmax = 0.2 * (bounds[:, 1] - bounds[:, 0])          # shape: (num_var,)
    Vmax_col = Vmax[:, None]                             # shape: (num_var,1)
    Vmax_1d = Vmax                                       # for clamping per-dimension
    if PSO_variant == 1:                                                        # with the constriction factor effect                                                                
        X = 2/(C-2+np.sqrt(C**2-4*C))                                          # contriction factor
        w = 1                                                                   # inertia

        # Initialize random velocities within variable limits
        vel = -Vmax_col + 2*Vmax_col * np.random.rand(num_var, swarm_size)      # velocity initialized with range a of 20% of the variable dynamic range

    elif PSO_variant == 2:                                                      # without the constriction factor effect 
        X = 1
        vel = -Vmax_col + 2*Vmax_col * np.random.rand(num_var, swarm_size) 
      
    # Initialize random position within variable limits
    if Particle_Gen_method == 1:
        lower_col = bounds[:, 0][:, None]
        range_col = (bounds[:, 1] - bounds[:, 0])[:, None]
        pos = lower_col + range_col * np.random.rand(num_var, swarm_size)

    # distribute particles equally on the each dimension 
    # (only set up for 2 dimnesions for testing purposes) 
    # elif Particle_Gen_method==2:
    #     pos=zeros(num_var,swarm_size_eq_space);
    #     for i=1:spacing_per_dim
    #         increment=(bounds(1,2)-bounds(1,1))/(spacing_per_dim-1);
    #         pos(:,spacing_per_dim*(i-1)+1:spacing_per_dim*i)=...
    #             [ones(1,spacing_per_dim)*(bounds(1,1)+increment*(i-1));...
    #             bounds(2,1)+(bounds(2,2)-bounds(2,1)).* ...
    #             (ones(1,spacing_per_dim).*(0:1/(spacing_per_dim-1):1))];
    #     end
    elif Particle_Gen_method == 2:
        # number of points per dimension
        m = spacing_per_dim
        # precompute 1×m vectors for each var
        gridAxes = []
        for d in range(num_var):
            gridAxes.append(np.linspace(bounds[d, 0], bounds[d, 1], m))
        
        # build the full n-D grid
        grid = np.meshgrid(*gridAxes, indexing='ij')
        
        # flatten into pos: each column is one particle
        swarm_size_eq_space = m**num_var
        pos = np.zeros((num_var, swarm_size_eq_space))
        for d in range(num_var):
            pos[d, :] = grid[d].flatten()

    new_vel = vel.copy()                                                        # set the randomly generated velocity to new velocity
    t = 1                                                                       # iteration

    # Generate data containers
    pbest = np.zeros(swarm_size)                                                # container for pbest values
    pbest_pos = np.zeros((num_var, swarm_size))                                # container for pbest position
    gbest_track = np.zeros(num_iter)                                            # track the global best
    avg_fitness_per_iter = np.zeros(num_iter)                                   # track the average func. val. per iter.

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
            fitness[i] = obj_func(pos[:, i])                                    # evaluate fitness values

        # Calculate avg fitness for the iteration & store
        avg_fitness = np.sum(fitness)/swarm_size                                # average function values for this iteration

        avg_fitness_per_iter[t-1] = np.sum(fitness)/swarm_size                  # store avearge func. value per iteration

        # Adjust function value based on penalty
        for i in range(swarm_size):
            fitness[i] = fitness[i] + abs(penalty_factor*avg_fitness * penalty_val[i])     # evaluate modified fitness values based on penalty

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
                    # print(f'No improvement for 50 iters; stopping at t = {t}')
                    gbest_track[t-1:] = np.ones(num_iter-t+1) * gbest_track[t-2]                                # Fill the rest of the last found val 
                    avg_fitness_per_iter[t-1:] = np.ones(num_iter-t+1) * avg_fitness_per_iter[t-2]                       # Fill the rest of the last found val
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

    iterations = t - 1
    
    return gbest, gbest_pos, iterations, gbest_track, avg_fitness_per_iter
