"""
This is the interpretation of my basic understanding of PSO. 
There are confusion with the basic version including contriction factor 
and the inertia. This also involves the concept of Vmax, how to handle 
and bound it.
                                                Irtiza Khan
                                                Date started: May 05, 2025
"""

from dataclasses import dataclass

import numpy as np

from bound_check import bound_check
from penalty import penalty_check
from repair import repair


@dataclass
class PSOResult:
    best_pos: np.ndarray            # global best position
    best_val: float                 # objective at best_pos (no penalty)
    best_violation: float           # constraint violation at best_pos (0 = feasible)
    top_pop: np.ndarray             # top 10 % of the last swarm, best first
    top_fitness: np.ndarray         # objective of those particles
    gbest_track: np.ndarray         # global best (with penalty) per iteration
    avg_fitness: np.ndarray         # average objective of the swarm per iteration
    infeasible_fraction: np.ndarray # fraction of the swarm outside the constraints
    n_iter: int                     # iterations run (fewer after a stall stop)
    n_eval: int                     # objective evaluations


def PSO_main(gp, num_var, bounds, constrain_func=None, constrain_bound=0.0,
             type_of_PSO="minimization", *, swarm_size=50, num_iter=100,
             C1=2.05, C2=2.05, PSO_variant=1, w_max=0.9, w_min=0.5,
             vmax_frac=0.2, constraint_handling=2, penalty_factor=5.0,
             craziness=0, stall_iter=15, seed=42, top_frac=0.1, verbose=True):
    """
    Particle swarm optimisation inside the box `bounds`.

    gp                   objective: a surrogate with .predict(X) (e.g. a
                         scikit-learn GP) or a plain function f(X). Either way
                         it gets the whole swarm, X of shape (swarm_size,
                         num_var), and returns one value per particle.
    num_var              number of design variables
    bounds               (num_var, 2): [low, high] for each variable
    constrain_func       optional inequality constraints g(X) <= constrain_bound;
                         g returns (swarm_size,) or (swarm_size, n_constraints)
    type_of_PSO          'minimization' or 'maximization'

    PSO_variant          1 => constriction factor (w = 1)
                         2 => inertia, from w_max down to w_min over the run
    vmax_frac            Vmax of each variable as a fraction of its range; the
                         starting velocities are drawn within +-Vmax
    constraint_handling  0 => none
                         1 => repair: infeasible particles are re-drawn inside
                              the bounds
                         2 => penalty: fitness worsened by
                              penalty_factor * |average fitness| * violation,
                              with a repair when more than 80 % of the swarm is
                              infeasible
                         3 => clip: positions held inside the bounds (as in
                              ESPSOLS); constrain_func, if given, by penalty
    craziness            1 adds a random share of the old velocity
    stall_iter           stop after more than this many iterations without a
                         better global best (0 => always run num_iter)
    seed                 seeds the Latin hypercube start and every random draw,
                         so the same inputs give the same answer
    top_frac             share of the last swarm returned in top_pop

    Returns a PSOResult.
    """
    if num_iter < 1:
        raise ValueError("num_iter must be at least 1")
    search_bounds = np.asarray(bounds, dtype=float).reshape(num_var, 2)
    low, high = search_bounds[:, 0], search_bounds[:, 1]
    predict = gp.predict if hasattr(gp, "predict") else gp
    rng = np.random.default_rng(seed)

    kind = str(type_of_PSO).lower()
    if kind.startswith("min"):
        sign = 1.0                          # the swarm always minimises sign * fitness
    elif kind.startswith("max"):
        sign = -1.0
    else:
        raise ValueError("type_of_PSO must be 'minimization' or 'maximization'")

    def violation(pop):
        viol = penalty_check(pop, search_bounds)
        if constrain_func is not None:
            g = np.asarray(constrain_func(pop), dtype=float).reshape(len(pop), -1)
            viol = viol + np.maximum(g - constrain_bound, 0.0).sum(axis=1)
        return viol

    C = C1 + C2
    if PSO_variant == 1:                    # with the constriction factor effect
        X = 2 / (C - 2 + np.sqrt(C**2 - 4 * C))     # constriction factor (needs C1 + C2 > 4)
        w = 1.0                             # inertia
    elif PSO_variant == 2:                  # without the constriction factor effect
        X = 1.0
        w = w_max
    else:
        raise ValueError("PSO_variant must be 1 or 2")

    # Initialize velocity (within +-Vmax) and positions (Latin hypercube)
    Vmax = vmax_frac * (high - low)
    vel = rng.uniform(-Vmax, Vmax, (swarm_size, num_var))
    pop = bound_check(seed, search_bounds, num_var, swarm_size)

    # Containers
    pbest = np.full(swarm_size, np.inf)
    pbest_pos = pop.copy()
    gbest, gbest_pos = np.inf, pop[0].copy()
    gbest_fit = gbest_viol = np.nan
    gbest_track = np.full(num_iter, np.nan)
    avg_fitness_per_iter = np.full(num_iter, np.nan)
    infeasible_fraction = np.full(num_iter, np.nan)
    counter = 0                             # iterations since the global best last improved
    n_eval = 0

    # Main loop
    for t in range(num_iter):

        # Constraint handling
        viol = violation(pop)
        bad = viol > 0
        infeasible_fraction[t] = bad.mean()
        if (constraint_handling == 1 and bad.any()) or \
           (constraint_handling == 2 and bad.sum() > 0.8 * swarm_size):
            pop[bad] = repair(pop[bad], search_bounds, rng)
            viol[bad] = violation(pop[bad])

        fitness = np.asarray(predict(pop), dtype=float).reshape(swarm_size)
        n_eval += swarm_size
        avg_fitness = fitness.mean()
        avg_fitness_per_iter[t] = avg_fitness

        f = sign * fitness                  # modified fitness, lower is better
        if constraint_handling in (2, 3):
            f = f + abs(penalty_factor * avg_fitness) * viol

        # personal bests
        better = f < pbest
        pbest[better] = f[better]
        pbest_pos[better] = pop[better]

        # global best (best found so far)
        i = int(np.argmin(f))
        if f[i] < gbest:
            gbest, gbest_pos = f[i], pop[i].copy()
            gbest_fit, gbest_viol = fitness[i], viol[i]
            counter = 0
        else:
            counter += 1
        gbest_track[t] = sign * gbest
        last_pop, last_f, last_fitness = pop, f, fitness

        # stopping criteria based on improvement
        if stall_iter and counter > stall_iter:
            if verbose:
                print(f"No improvement for {stall_iter} iters; stopping at t = {t}")
            break

        # new velocity and position
        if PSO_variant == 2:
            w = w_max - (w_max - w_min) * t / num_iter
        r1 = rng.random((swarm_size, num_var))
        r2 = rng.random((swarm_size, num_var))
        new_vel = X * (w * vel + C1 * r1 * (pbest_pos - pop) + C2 * r2 * (gbest_pos - pop))
        if craziness == 1:
            new_vel = new_vel + X * vel * rng.random(num_var)
        new_vel = np.clip(new_vel, -Vmax, Vmax)
        pop = pop + new_vel
        if constraint_handling == 3:
            pop = np.clip(pop, low, high)
        vel = new_vel

    n_iter = t + 1
    top = np.argsort(last_f)[:max(1, int(top_frac * swarm_size))]
    return PSOResult(best_pos=gbest_pos, best_val=float(gbest_fit),
                     best_violation=float(gbest_viol),
                     top_pop=last_pop[top], top_fitness=last_fitness[top],
                     gbest_track=gbest_track[:n_iter],
                     avg_fitness=avg_fitness_per_iter[:n_iter],
                     infeasible_fraction=infeasible_fraction[:n_iter],
                     n_iter=n_iter, n_eval=n_eval)


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

    Sep 28, 2026 - Used for the rudder section fit (match_rudder_section.py).
                - The objective can be any function of the whole swarm, or a
                surrogate with .predict as before. Settings are keyword
                arguments with the old values as defaults. Returns a
                PSOResult (best position and value, top 10 %, tracking).
                - One seeded generator for every random draw, so a run can be
                repeated exactly. pbest update vectorised.
                - Fixed: the return sat inside the loop (one iteration only);
                argmin where argsort was meant for the top 10 %; bound_check
                returns one array, not two; bounds wrapped in an extra list;
                minimisation subtracted the penalty, so violations were
                rewarded; the no-improvement counter never counted; the
                penalty was computed before the repair; inertia now runs from
                w_max to w_min.
                - Added constraint_handling 3 (clip to the bounds, as in
                ESPSOLS) and optional inequality constraints (constrain_func).
"""
