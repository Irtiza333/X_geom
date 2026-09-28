import numpy as np
from numpy.random import random_sample as rand
from replica_funcs import *


def PSO_ctrl_pts(rud_x_shrp_norm, rud_y_shrp_norm, pso_params, foil_settings, bounds):

    nPop = pso_params.nPop
    pers_fac = pso_params.pers_fac
    soc_fac = pso_params.soc_fac
    t_max = pso_params.t_max
    VmaxPerc = pso_params.VmaxPerc
    BoundHandling = pso_params.BoundHandling 
    ConsHandling = pso_params.ConsHandling
    ConsOn = pso_params.ConsOn
    Obj = pso_params.Obj
    all_bounds = bounds

    n_bez_samples = foil_settings.n_bez_samples




    Dims = all_bounds.shape[0]
    Bounds = all_bounds
    lenDim = np.abs(Bounds[:, 0] - Bounds[:, 1])
    Vmax = lenDim * VmaxPerc / 100

    pos_history = np.zeros((nPop, Dims, t_max))
    velo_history = np.zeros((nPop, Dims, t_max))
    xo = np.zeros((nPop, Dims))
    vo = np.zeros((nPop, Dims))
    RMSo = np.zeros((nPop, 1))
    maxRSo = np.zeros((nPop, 1))
    Gbest = np.zeros((t_max, Dims))
    Gbest_val = np.zeros((t_max, 1))
    v_new = np.zeros((nPop, Dims))
    BoundTest = np.zeros((nPop, Dims))
    BoundSatisfy = np.zeros((nPop, 1))
    ConSatisfy = np.zeros((nPop, 1))
    Satisfy = np.zeros((nPop, 1))
    inert_term = np.zeros((nPop, Dims))
    pers_term = np.zeros((nPop, Dims))
    social_term = np.zeros((nPop, Dims))

    Gbest_aux = np.zeros((t_max, Dims))
    Gbest_val_aux = np.zeros((t_max, 1))

    low_bounds = Bounds[:, 0]
    up_bounds = Bounds[:, 1]

    # initialize population
    t = 0
    for i in range(nPop):
        while ConSatisfy[i] == 0:
            xo[i] = np.random.uniform(Bounds[:, 0], Bounds[:, 1])
            ConSatisfy[i] = CheckCons(xo[i, :], ConsHandling, ConsOn)
        RMSo[i], maxRSo[i] = EvalObj(xo[i, :], rud_x_shrp_norm, rud_y_shrp_norm, n_bez_samples)
    pos_history[:, :, 0] = xo[:, :]
    Pbest = xo.copy()
    if Obj == 0:
        Fo = RMSo
        sFo = maxRSo
    else:
        Fo = maxRSo
        sFo = RMSo
    Pbest_val = Fo.copy()

    # initialize velocities
    for j in range(Dims):
        vo[:, j] = np.reshape(Vmax[j] - 2 * Vmax[j] * rand((nPop, 1)), nPop)

    # find initial global best
    Gbest_val[0] = np.min(Fo)
    Gbest_ind = np.argmin(Fo)
    Gbest[0, :] = xo[Gbest_ind, :]

    Gbest_val_aux[0] = np.min(sFo)
    Gbest_ind_aux = np.argmin(sFo)
    Gbest_aux[0, :] = xo[Gbest_ind_aux, :]

    xi = xo.copy()
    vi = vo.copy()
    Fi = Fo.copy()
    sFi = sFo.copy()
    RMSi = RMSo.copy()
    maxRSi = maxRSo.copy()

    # optimization loop
    while t < t_max - 1:
        inert_fac = 1 - (1 - 0.7) / t_max * t
        inert_term[:, :] = inert_fac * vi[:, :]
        pers_term[:, :] = np.multiply(pers_fac * rand((nPop, Dims)), Pbest[:, :] - xi[:, :])
        social_term[:, :] = np.multiply(soc_fac * rand((nPop, Dims)), Gbest[t, :] - xi[:, :])
        v_new[:, :] = inert_term[:, :] + pers_term[:, :] + social_term[:, :]

        # check for too high velocities
        is_non_zero = (Vmax != 0).ravel()
        Vfac = np.zeros_like(v_new)
        Vfac[:, is_non_zero] = np.abs(v_new[:, is_non_zero] / Vmax[is_non_zero])
        vmax_factor = np.max(Vfac, axis=1)
        scale_mask = vmax_factor > 1
        vi = np.where(scale_mask[:, None], v_new / vmax_factor[:, None], v_new)
        xi = xi + vi

        # bound handling
        if BoundHandling == 0:  # fly over
            BoundTest = ((xi >= low_bounds) & (xi <= up_bounds)).astype(int)
        elif BoundHandling == 1:  # repair
            xi[:] = np.clip(xi, low_bounds, up_bounds)
            BoundTest = np.ones_like(xi, dtype=int)
        elif BoundHandling == 2:  # bounce
            upper_mask = xi > up_bounds
            lower_mask = xi < low_bounds
            vi[upper_mask | lower_mask] *= -1
            xi[:] = np.clip(xi, low_bounds, up_bounds)

        # for each individual in population
        for i in range(nPop):
            RMSi[i], maxRSi[i] = EvalObj(xi[i, :], rud_x_shrp_norm, rud_y_shrp_norm, n_bez_samples)
            if Obj == 0:
                Fi[i] = RMSi[i]
                sFi[i] = maxRSi[i]
            else:
                Fi[i] = maxRSi[i]
                sFi[i] = RMSi[i]
            BoundSatisfy[i] = all(BoundTest[i, :])
            ConSatisfy[i] = CheckCons(xi[i, :],  ConsHandling, ConsOn)
            Satisfy[i] = BoundSatisfy[i] and ConSatisfy[i]

            if Fi[i] <= Pbest_val[i]:
                if Satisfy[i] == 1 or ConsHandling == 1:
                    Pbest_val[i] = Fi[i]
                    Pbest[i, :] = xi[i, :]

        velo_history[:, :, t] = vi[:, :]
        pos_history[:, :, t + 1] = xi[:, :]
        t = t + 1

        if t % 10 == 0:
            print('time', t)

        Combined = np.concatenate((xi, Fi, Satisfy), axis=1)
        Combined_aux = np.concatenate((xi, sFi, Satisfy), axis=1)

        Sorted = Combined[Combined[:, Dims].argsort()]
        Sorted_aux = Combined_aux[Combined_aux[:, Dims].argsort()]

        for row in range(nPop):
            if Sorted[row, Dims + 1] == 1 or ConsHandling == 1:
                Gbest_val[t] = Sorted[row, Dims]
                Gbest[t, :] = Sorted[row, 0:-2]
                break
            if row == nPop - 1:
                # print(t)
                # print('All points non feasible, no global best')
                Gbest_val[t] = Gbest_val[t - 1]
                Gbest[t, :] = Gbest[t - 1, :]

        for row in range(nPop):
            if Sorted_aux[row, Dims + 1] == 1 or ConsHandling == 1:
                Gbest_val_aux[t] = Sorted_aux[row, Dims]
                Gbest_aux[t, :] = Sorted_aux[row, 0:-2]
                break
            if row == nPop - 1:
                # print(t)
                # print('All points non feasible, no global best')
                Gbest_val_aux[t] = Gbest_val_aux[t - 1]
                Gbest_aux[t, :] = Gbest_aux[t - 1, :]

    BestVal = np.min(Gbest_val)
    Gbest_time = np.argmin(Gbest_val)
    BestPt = Gbest[Gbest_time, :]
    LastBestVal = Gbest_val[t]
    LastBestPt = Gbest[t, :]

    BestVal_aux = np.min(Gbest_val_aux)
    Gbest_time_aux = np.argmin(Gbest_val_aux)
    BestPt_aux = Gbest_aux[Gbest_time_aux, :]
    LastBestVal_aux = Gbest_val_aux[t]
    LastBestPt_aux = Gbest_aux[t, :]
    # print(pos_history[:,:,2])
    # print(pos_history[:,:,80])


    return BestPt, BestVal, pos_history, velo_history







def EvalObj(xi, rud_x_shrp_norm, rud_y_shrp_norm, n_bez_samples):
    
    cand_x, cand_y = build_half_airfoil(xi, n_bez_samples)

    if cand_x.size < 2:
        return 10.0, 10.0  # degenerate candidate -- heavy penalty, no crash

    order = np.argsort(cand_x)
    cand_x, cand_y = cand_x[order], cand_y[order]
    cand_x, uniq_idx = np.unique(cand_x, return_index=True)
    cand_y = cand_y[uniq_idx]

    if cand_x.size < 2:
        return 10.0, 10.0

    # np.interp clamps to the end values outside [cand_x[0], cand_x[-1]], so a
    # candidate that doesn't fully span the target's x-range still gets a
    # (poor) score instead of an error.
    y_cand = np.interp(rud_x_shrp_norm, cand_x, cand_y)

    dev = rud_y_shrp_norm - y_cand
    square_dev = dev * dev
    RMS_dev = float(np.sqrt(np.mean(square_dev)))
    RSmax = float(np.sqrt(np.max(square_dev)))
    return RMS_dev, RSmax


def CheckCons(xi, ConsHandling, ConsOn):
    """Only inequality: X2 <= X3 (control points stay in x-order; X0=X1=0 and
    X4=chord are already locked, so this is the one ordering left to check)."""
    if not ConsOn:
        return True

    X2, X3 = xi[1], xi[2]
    satisfy = X2 <= X3

    if ConsHandling == 2 and not satisfy:
        repaired = min(X2, X3)
        xi[1] = repaired
        satisfy = True

    return bool(satisfy)
