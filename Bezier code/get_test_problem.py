"""
get_test_problem_analysis.py            Created by: Irtiza M. Khan.
                                        Created on: February 03 / 2025
                                        last modified: February 12 / 2025

This code returns function handle, number of variables, bounds, constraint functions, constraint limit, best known solution, and best known value
"""

import numpy as np
from para_control_bez import para_control_bez   
from para_control_cub import para_control_cub

Pitch_original = lambda x: 1*(19344.5071*x**12 + -114044.8587*x**11 + 280789.2801*x**10 + -357377.6146*x**9 + 207947.2705*x**8 + 43330.8173*x**7 + -173099.4797*x**6 + 143116.5772*x**5 + -65570.9523*x**4 + 18410.2055*x**3 + -3128.7530*x**2 + 294.0671*x + -10.4419)

ChordLength_original = lambda x: 1*(-143202.4761*x**12 + 978274.9902*x**11 + -2992184.0323*x**10 + 5408923.9625*x**9 + -6424276.8851*x**8 + 5271614.6993*x**7 + -3058632.5267*x**6 + 1261908.7720*x**5 + -366745.1423*x**4 + 73096.4455*x**3 + -9470.1908*x**2 + 716.1619*x + -23.7157)


def get_test_problem(problem_id, para_control, pitch_con, chord_con):
    """
    Returns function handle, number of variables, bounds, constraint functions, 
    constraint limit, best known solution, and best known value
    """
    
    if problem_id == 1:
        # F1
        func = lambda var: np.sum(var**2)  # objective function
        num_var = 2  # number of variables
        bounds = np.tile([-5.12, 5.12], (num_var, 1))  # upper and lower bound of variables 
        # bounds = np.array([[-5.12, 5.12], [-2, 2]])  # upper and lower bound of variables
        best_known_sol = np.array([0, 0]) 
        best_known_val = 0
        constrain_func = []  # constraint function
        constrain_bound = np.array([])  # constraint limit
    
    elif problem_id == 2:
        # F2
        func = lambda var: 100*(var[1] - var[0]**2)**2 + (1 - var[0])**2
        num_var = 2
        bounds = np.tile([-2.048, 2.048], (num_var, 1))
        constrain_func = []
        constrain_bound = np.array([])
        best_known_sol = np.array([1, 1])
        best_known_val = 0
    
    elif problem_id == 5:
        # F5 
        def func(var):
            return (0.002 + np.sum(1 / ((np.arange(1, 26) + np.sum((var.reshape(-1, 1) - \
                    np.array([np.tile([-32, -16, 0, 16, 32], 5), 
                             np.repeat([-32, -16, 0, 16, 32], 5)]))**6, axis=0))))**(-1))
        
        num_var = 2
        bounds = np.tile([-65.536, 65.536], (num_var, 1))
        constrain_func = []
        constrain_bound = np.array([])
        best_known_sol = np.array([-31.97833, -31.97833])
        best_known_val = 0.998003837794449325873406851315

    elif problem_id == 8:
        # F8
        def func(var):
            return 1 + np.sum(var**2 / 4000) - np.prod(np.cos(var / np.sqrt(np.arange(1, len(var)+1))))
        
        num_var = 10
        bounds = np.tile([-600, 600], (num_var, 1))
        constrain_func = []
        constrain_bound = np.array([])
        best_known_sol = np.zeros(num_var)
        best_known_val = 0
    
    elif problem_id == 2.1:  # Test Problem 2.1
        c = np.array([42, 44, 45, 47, 47.5])
        Q = 100 * np.eye(5)  # Identity matrix scaled by 100
        func = lambda var: c @ var - 0.5 * (var @ Q @ var)  # matrix multiplication
        num_var = 5
        bounds = np.tile([0, 1], (num_var, 1))
        constrain_func = [lambda var: 20*var[0] + 12*var[1] + 11*var[2] + 7*var[3] + 4*var[4]]
        constrain_bound = np.array([40])
        best_known_sol = np.array([1, 1, 0, 1, 0])
        best_known_val = -17

    elif problem_id == 3.1:  # Test Problem 3.1
        func = lambda var: var[0] + var[1] + var[2]
        num_var = 8
        bounds = np.array([[100, 10000], [1000, 10000], [1000, 10000], [10, 1000], 
                           [10, 1000], [10, 1000], [10, 1000], [10, 1000]])
        constrain_func = [
            lambda var: -1 + 0.0025 * (var[3] + var[5]),
            lambda var: -1 + 0.0025 * (-var[3] + var[4] + var[6]),
            lambda var: -1 + 0.01 * (-var[4] + var[7]),
            lambda var: 100 * var[0] - var[0] * var[5] + 833.33252 * var[3] - 83333.333,
            lambda var: var[1] * var[3] - var[1] * var[6] - 1250 * var[3] + 1250*var[4],
            lambda var: var[2] * var[4] - var[2] * var[7] - 2500 * var[4] + 1250000,
        ]
        constrain_bound = np.zeros(6)
        best_known_sol = np.array([579.31, 1359.97, 5109.97, 182.02, 295.6, 217.98, 286.42, 395.60])
        best_known_val = 7049.25
    
    elif problem_id == 4.6:  # Test Problem 4.6
        func = lambda var: -var[0] - var[1]
        num_var = 2
        bounds = np.array([[0, 3], [0, 4]])
        constrain_func = [
            lambda var: var[1] - (2*var[0]**4 - 8*var[0]**3 + 8*var[0]**2 + 2),
            lambda var: var[1] - (4*var[0]**4 - 32*var[0]**3 + 88*var[0]**2 - 96*var[0] + 36),
        ]
        constrain_bound = np.array([0, 0])
        best_known_sol = np.array([2.3295, 3.1783])
        best_known_val = -5.5079

   
    elif problem_id == 11:
        para_control = 3 
        R_values = np.concatenate([np.arange(0.17, 0.98, 0.018), [0.99, 0.999]])  # 100 points for smoother curves
        pitch, chord, *_ = para_control_bez(Pitch_original, ChordLength_original,R_values, para_control, pitch_con, chord_con)
        # Converging pitch distribution to the original pitch distribution
        R_values = np.concatenate([np.arange(0.17, 0.98, 0.018), [0.99, 0.999]])
        R_fine = np.linspace(R_values[0], R_values[-1], 100)
        func =   np.sum((pitch(R_fine)-Pitch_original(R_fine))**2)  # objective function (minimized at 0)
                                 # no contraint bounds
        return func

    elif problem_id == 12:
        para_control = 2 
        R_values = np.concatenate([np.arange(0.17, 0.98, 0.018), [0.99, 0.999]])  # 100 points for smoother curves
        pitch, chord, chord_con_points, pitch_con_points = para_control_bez(Pitch_original, ChordLength_original,R_values, para_control, pitch_con, chord_con)
        # Converging pitch distribution to the original pitch distribution
        R_values = np.concatenate([np.arange(0.17, 0.98, 0.018), [0.99, 0.999]])
        R_fine = np.linspace(R_values[0], R_values[-1], 100)
        func =   np.sum((chord(R_fine)-ChordLength_original(R_fine))**2)  # objective function (minimized at 0)
                                 # no contraint bounds
        return func

    elif problem_id == 13:
        para_control = 3 
        R_values = np.concatenate([np.arange(0.17, 0.98, 0.018), [0.99, 0.999]])  # 100 points for smoother curves
        pitch, chord = para_control_cub(Pitch_original, ChordLength_original,R_values, para_control, pitch_con, chord_con)
        # Converging pitch distribution to the original pitch distribution
        R_values = np.concatenate([np.arange(0.17, 0.98, 0.018), [0.99, 0.999]])
        R_fine = np.linspace(R_values[0], R_values[-1], 100)
        func =   np.sum((pitch(R_fine)-Pitch_original(R_fine))**2)  # objective function (minimized at 0)
                                 # no contraint bounds
        return func

    elif problem_id == 14:
        para_control = 2 
        R_values = np.concatenate([np.arange(0.17, 0.98, 0.018), [0.99, 0.999]])  # 100 points for smoother curves
        pitch, chord = para_control_cub(Pitch_original, ChordLength_original,R_values, para_control, pitch_con, chord_con)
        # Converging pitch distribution to the original pitch distribution
        R_values = np.concatenate([np.arange(0.17, 0.98, 0.018), [0.99, 0.999]])
        R_fine = np.linspace(R_values[0], R_values[-1], 100)
        func =   np.sum((chord(R_fine)-ChordLength_original(R_fine))**2)  # objective function (minimized at 0)
                                 # no contraint bounds
        return func
        
    else:
        raise ValueError('Invalid problem_id. Choose a valid problem number.')
    
    return func, num_var, bounds, constrain_func, constrain_bound, best_known_sol, best_known_val
