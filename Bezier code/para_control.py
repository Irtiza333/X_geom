import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from spmak1 import spmak1
from fnval1 import fnval1

def para_control(MaxCamber, Pitch, ChordLength, MaxThickness, SkewAngle, Rake, R_values, para_control_flag, x1):
    """
    Python equivalent of Para_control.m function
    """
    
    # Define the anonymous functions for each polynomial
    MaxCamber_origin = MaxCamber
    Pitch_origin = Pitch
    ChordLength_origin = ChordLength
    MaxThickness_chord_length = MaxThickness
    SkewAngle_origin = SkewAngle
    Rake_origin = Rake

    con_points_skew = 4
    # define radial positions
    R_new_skew = np.zeros(con_points_skew)
    skew_new = np.zeros(con_points_skew)

    con_points_chord = 4  # # control points 
    # --- 3) generate radial positions R_new_chord(1..4)
    R_new_chord = np.zeros(con_points_chord)
    chord_new = np.zeros(con_points_chord)

    con_points_thick = 4  # # control points
    # --- 2) generate radial positions R_new_thick(1..4)
    R_new_thick = np.zeros(con_points_thick)
    Thickness_new = np.zeros(con_points_thick)

    con_points_pitch = 4
    # --- 3) generate radial positions R_new_pitch(1..5)
    R_new_pitch = np.zeros(con_points_pitch)
    pitch_new = np.zeros(con_points_pitch)

    con_points_camber = 5  # # control points
    # --- 2) generate radial positions R_new_camber(1..4)
    R_new_camber = np.zeros(con_points_camber)
    camber_new = np.zeros(con_points_camber)

    con_points_rake = 4  # # control points
    # --- 2) generate radial positions R_new_rake(1..4)
    R_new_rake = np.zeros(con_points_rake)
    # --- 3) generate rake values rake_new(1..4)
    rake_new = np.zeros(con_points_rake)

    # Skew: para_control==1
    if para_control_flag == 0:
        pass  # break equivalent
    
    if para_control_flag == 1 or para_control_flag == 7:
        upper_bound = 50
        lower_bound = -10

        R_new_skew[0] = R_values[0]
        R_new_skew[-1] = R_values[-1]
        dis = R_values[0]
        
        for i in range(1, con_points_skew - 1):
            dis = dis + (R_values[-1] - R_values[0]) / (con_points_skew - 1)
            R_new_skew[i] = (dis + (R_values[-1] - R_values[0]) / (con_points_skew - 1) * 
                           0.45 * np.random.rand() * (2 * np.random.randint(0, 2) - 1))
            if i == con_points_skew - 2:  # equivalent to i==con_points_skew-1 in MATLAB
                R_new_skew[i] = dis + (R_values[-1] - R_values[0]) / (con_points_skew - 1) * 0.8 * np.random.rand()

        # define skew radial positions
        skew_new[0] = 0
        dis = skew_new[0]
        
        for i in range(1, con_points_skew):
            skew_new[i] = (dis + (upper_bound - lower_bound) / (con_points_skew - 2) * 
                          0.5 * np.random.rand() * (2 * np.random.randint(0, 2) - 1))
            dis = dis + ((upper_bound - lower_bound)) / (con_points_skew - 2) * 0.5

        ctrl_pts_S = np.vstack([R_new_skew, skew_new])

        r_knots = ctrl_pts_S[0, :]
        if np.any(np.diff(r_knots) <= 0):
            raise ValueError('The r-coordinates of control points must be strictly increasing.')

        degree = 3  # cubic
        order = degree + 1  # = 4
        N = ctrl_pts_S.shape[1]  # = 6 control points

        num_interior_knots = N - order
        # For N=6, order=4 => 6−4 = 2 interior knots

        if num_interior_knots > 0:
            # Place the interior knots uniformly between 0 and 1
            interior = np.linspace(0, 1, num_interior_knots + 2)
            # Remove the first/last (0 and 1), keep exactly (N−order) values:
            interior = interior[1:-1]
        else:
            interior = np.array([])

        t = np.concatenate([
            np.zeros(order),     # repeat 0 four times
            interior,            # e.g. two values: [0.3333, 0.6667]
            np.ones(order)       # repeat 1 four times
        ])

        spline_cam = spmak1(t, ctrl_pts_S)

        u = np.linspace(0, 1, 1000)  # dense sampling in the spline‐parameter
        xy = fnval1(spline_cam, u)   # 2×1000: row1 = r(u), row2 = skew(u)
        r_curve = xy[0, :]           # r(t_j)
        skew_curve = xy[1, :]        # skew(t_j)

        # R_values is your list of radii between 0.1 and 1.0
        SkewAngle = lambda x: interp1d(r_curve, skew_curve, kind='cubic', 
                                     bounds_error=False, fill_value='extrapolate')(x)

    # Chord Control
    if para_control_flag == 2 or para_control_flag == 7:
        # --- 1) bounds & number of points
        upper_bound = 0.65
        lower_bound = 0.15

        # --- 2) fix root & tip chords to original
        chord_tip = ChordLength_origin(R_values[-1])

        # --- 3) generate radial positions R_new_chord(1..4)
        R_new_chord = np.zeros(con_points_chord)
        R_new_chord[0] = R_values[0]
        R_new_chord[-1] = R_values[-1]
        stepR = (R_values[-1] - R_values[0]) / 3
        disR = R_values[0]
        
        for i in range(1, con_points_chord - 1):
            disR = disR + stepR
            R_new_chord[i] = disR + 0.95 * np.random.rand() * (R_new_chord[-1] - disR)
            if i > 1:  # i>2 in MATLAB (1-based)
                R_new_chord[i] = R_new_chord[i-1] + np.random.rand() * (R_new_chord[-1] - R_new_chord[i-1])

        # --- 4) generate chord values chord_new(1..4)
        chord_new = np.zeros(con_points_chord)
        chord_new[-1] = chord_tip
        stepC = (upper_bound - lower_bound) / (con_points_chord - 1)
        disC = 0
        
        for i in range(con_points_chord - 1):
            disC = disC + stepC
            # jitter ±50% of stepC
            if i == 0:
                chord_new[i] = disC + np.random.rand() * 0.5 * stepC
            elif i == 1:
                chord_new[i] = (upper_bound - lower_bound) + np.random.rand() * (upper_bound - (upper_bound - lower_bound))
            else:
                chord_new[i] = chord_new[i-1] + np.random.rand() * (upper_bound - chord_new[i-1])

        # --- 5) pack into ctrl_pts and check monotonicity
        ctrl_pts = np.vstack([R_new_chord, chord_new])
        if np.any(np.diff(ctrl_pts[0, :]) <= 0):
            raise ValueError('Chord-control radii must be strictly increasing.')

        # --- 6) build a clamped cubic B-spline in B-form
        degree = 3
        order = degree + 1  # =4
        N = ctrl_pts.shape[1]
        n_int = N - order  # interior knots
        
        if n_int > 0:
            interior = np.linspace(0, 1, n_int + 2)
            interior = interior[1:-1]
        else:
            interior = np.array([])
            
        t = np.concatenate([np.zeros(order), interior, np.ones(order)])
        spline_chord = spmak1(t, ctrl_pts)

        # --- 7) sample densely, then invert r→chord
        u_sample = np.linspace(0, 1, 1000)
        xy_sample = fnval1(spline_chord, u_sample)
        r_curve = xy_sample[0, :]
        chord_curve = xy_sample[1, :]
        ChordLength = lambda x: interp1d(r_curve, chord_curve, kind='cubic',
                                       bounds_error=False, fill_value='extrapolate')(x)
        chord_at_R = ChordLength(R_values)

        # --- 8) visualize
        R_plot = np.linspace(R_values[0], R_values[-1], 200)
        chord_plot = ChordLength(R_plot)

        plt.figure()
        plt.hold = True  # Note: hold is deprecated in newer matplotlib
        plt.plot(R_plot, chord_plot, 'k-', linewidth=2)
        plt.plot(R_new_chord, chord_new, 'ro', markerfacecolor='r', markersize=8)
        plt.plot(R_values, ChordLength_origin(R_values), 'b--', linewidth=1.5)
        plt.title('Chord Length Control via B-spline')
        plt.xlabel('R')
        plt.ylabel('Chord')
        plt.legend(['Modified chord', 'Control points', 'Original chord'], loc='best')
        plt.grid(True)

    # Thickness Control
    if para_control_flag == 3 or para_control_flag == 7:
        # --- 1) bounds & number of points
        upper_bound = 0.10
        lower_bound = 0.005

        R_new_thick[0] = R_values[0]
        R_new_thick[-1] = R_values[-1]
        stepR = (R_values[-1] - R_values[0]) / (con_points_thick - 1)
        disR = R_values[0]
        
        for i in range(1, con_points_thick - 1):
            disR = disR + stepR
            # ±45% jitter
            R_new_thick[i] = disR + (2 * np.random.rand() - 1) * 0.45 * stepR
            if i == con_points_thick - 2:  # equivalent to con_points_thick-1 in MATLAB
                # extra ±25% jitter on the penultimate
                R_new_thick[i] = disR + np.random.rand() * 0.45 * stepR

        # --- 3) generate thickness values Thickness_new(1..4)
        # lock root & tip to original
        Thickness_new[-1] = MaxThickness_chord_length(R_values[-1])

        for i in range(con_points_thick - 1):
            if i == 0:
                Thickness_new[i] = max((lower_bound + np.random.rand() * (upper_bound - lower_bound)), 0.15)
            elif i == con_points_thick - 2:  # equivalent to con_points_thick-1 in MATLAB
                Thickness_new[i] = max((Thickness_new[i-1] + (2 * np.random.rand() - 1) * 
                                      (upper_bound - Thickness_new[i-1])), Thickness_new[-1])
            else:
                Thickness_new[i] = max(Thickness_new[i-1] - np.random.rand() * 
                                     (upper_bound - Thickness_new[i-1]), 0.05)

        # --- 4) build a clamped cubic B-spline in B-form
        ctrl_pts_t = np.vstack([R_new_thick, Thickness_new])
        if np.any(np.diff(ctrl_pts_t[0, :]) <= 0):
            raise ValueError('Thickness-control radii must be strictly increasing.')
            
        degree = 3
        order = degree + 1
        N = ctrl_pts_t.shape[1]
        n_int = N - order
        
        if n_int > 0:
            interior = np.linspace(0, 1, n_int + 2)
            interior = interior[1:-1]
        else:
            interior = np.array([])
            
        t = np.concatenate([np.zeros(order), interior, np.ones(order)])
        spline_thick = spmak1(t, ctrl_pts_t)

        # --- 5) sample densely, then invert r→thickness
        u_sample = np.linspace(0, 1, 1000)
        xy_sample = fnval1(spline_thick, u_sample)
        r_curve_thick = xy_sample[0, :]
        thick_curve = xy_sample[1, :]
        MaxThickness = lambda x: interp1d(r_curve_thick, thick_curve, kind='cubic',
                                        bounds_error=False, fill_value='extrapolate')(x)

    # Pitch Control
    if para_control_flag == 4 or para_control_flag == 7:
        # --- 1) bounds & number of points
        upper_bound = 1.5
        lower_bound = 0.1

        R_new_pitch[0] = R_values[0]
        R_new_pitch[-1] = R_values[-1]
        stepR = (R_values[-1] - R_values[0]) / (con_points_pitch - 1)
        disR = R_values[0]
        
        for i in range(1, con_points_pitch - 1):
            disR = disR + stepR
            # ±45% jitter
            R_new_pitch[i] = disR + (2 * np.random.rand() - 1) * 0.45 * stepR
            if i == con_points_pitch - 2:  # equivalent to con_points_pitch-1 in MATLAB
                # additional up-to ±25% on the penultimate
                R_new_pitch[i] = R_new_pitch[i] + 0.25 * np.random.rand() * stepR

        # --- 4) generate pitch values pitch_new(1..5)
        for i in range(con_points_pitch):
            if i == 0:
                pitch_new[i] = lower_bound + np.random.rand() * (upper_bound - lower_bound)
            elif i == con_points_pitch - 2 or i == con_points_pitch - 1:  # equivalent to MATLAB conditions
                pitch_new[i] = lower_bound + np.random.rand() * (pitch_new[i-1] - lower_bound)
            else:
                pitch_new[i] = pitch_new[i-1] + np.random.rand() * (upper_bound - pitch_new[i-1])

        # --- 5) pack into ctrl_pts and check
        ctrl_pts_p = np.vstack([R_new_pitch, pitch_new])
        if np.any(np.diff(ctrl_pts_p[0, :]) <= 0):
            raise ValueError('Pitch-control radii must be strictly increasing.')

        # --- 6) build clamped cubic B-spline in B-form
        degree = 3
        order = degree + 1
        N = ctrl_pts_p.shape[1]
        n_int = N - order
        
        if n_int > 0:
            interior = np.linspace(0, 1, n_int + 2)
            interior = interior[1:-1]
        else:
            interior = np.array([])
            
        t = np.concatenate([np.zeros(order), interior, np.ones(order)])
        spline_pitch = spmak1(t, ctrl_pts_p)

        # --- 7) sample densely, invert r→pitch
        u_sample = np.linspace(0, 1, 1000)
        xy_sample = fnval1(spline_pitch, u_sample)
        r_curve = xy_sample[0, :]
        pitch_curve = xy_sample[1, :]
        Pitch = lambda x: interp1d(r_curve, pitch_curve, kind='cubic',
                                 bounds_error=False, fill_value='extrapolate')(x)

    # Camber Control
    if para_control_flag == 5 or para_control_flag == 7:
        # --- 1) bounds & number of points
        upper_bound = 0.07
        lower_bound = -0.07

        R_new_camber[0] = R_values[0]
        R_new_camber[-1] = R_values[-1]
        stepR = (R_values[-1] - R_values[0]) / (con_points_camber - 1)
        disR = R_values[0]
        
        for i in range(1, con_points_camber - 1):
            disR = disR + stepR
            # ±45% jitter
            R_new_camber[i] = disR + np.random.rand() * (2 * np.random.randint(0, 2) - 1) * 0.45 * stepR
            if i == con_points_camber - 2:  # equivalent to con_points_camber-1 in MATLAB
                # extra ±25% jitter on penultimate
                R_new_camber[i] = R_new_camber[i] + np.random.rand() * 0.5 * stepR

        # --- 3) generate camber values camber_new(1..4)
        # random root within bounds
        camber_new[0] = lower_bound + np.random.rand() * (upper_bound - lower_bound)
        camber_new[1] = camber_new[0] + np.random.rand() * (upper_bound - camber_new[0])
        # tip fixed to original
        camber_new[-1] = MaxCamber_origin(R_values[-1])

        for i in range(2, con_points_camber - 1):
            if i != con_points_camber - 2:  # equivalent to i~=con_points_camber-1 in MATLAB
                camber_new[i] = camber_new[i-1] + np.random.rand() * (upper_bound - camber_new[i-1])
            else:
                camber_new[i] = lower_bound + np.random.rand() * (upper_bound - lower_bound)

        # --- 4) pack into ctrl_pts and check monotonicity
        ctrl_pts_c = np.vstack([R_new_camber, camber_new])
        if np.any(np.diff(ctrl_pts_c[0, :]) <= 0):
            raise ValueError('Camber-control radii must be strictly increasing.')

        # --- 5) build clamped cubic B-spline in B-form
        degree = 3
        order = degree + 1
        N = ctrl_pts_c.shape[1]
        n_int = N - order
        
        if n_int > 0:
            interior = np.linspace(0, 1, n_int + 2)
            interior = interior[1:-1]
        else:
            interior = np.array([])
            
        t = np.concatenate([np.zeros(order), interior, np.ones(order)])
        spline_camber = spmak1(t, ctrl_pts_c)

        # --- 6) sample densely, invert r→camber
        u_sample = np.linspace(0, 1, 1000)
        xy_sample = fnval1(spline_camber, u_sample)
        r_curve = xy_sample[0, :]
        camber_curve = xy_sample[1, :]
        MaxCamber = lambda x: interp1d(r_curve, camber_curve, kind='cubic',
                                     bounds_error=False, fill_value='extrapolate')(x)

    # Rake Control
    if para_control_flag == 6 or para_control_flag == 7:
        # --- 1) bounds & number of points
        upper_bound = 0.2
        lower_bound = -0.1

        R_new_rake[0] = R_values[0]
        R_new_rake[-1] = R_values[-1]
        stepR = (R_values[-1] - R_values[0]) / (con_points_rake - 1)
        disR = R_values[0]
        
        for i in range(1, con_points_rake - 1):
            disR = disR + stepR
            # ±45% jitter
            R_new_rake[i] = disR + (2 * np.random.rand() - 1) * 0.45 * stepR
            if i == con_points_rake - 2:  # equivalent to con_points_rake-1 in MATLAB
                # extra ±25% jitter on the penultimate
                R_new_rake[i] = R_new_rake[i] + np.random.rand() * 0.5 * stepR

        # lock root & tip to original
        rake_new[0] = Rake_origin(R_values[0])

        for i in range(1, con_points_rake):
            rake_new[i] = lower_bound + np.random.rand() * (upper_bound - lower_bound)

        # --- 4) build B-spline in B-form
        ctrl_pts_r = np.vstack([R_new_rake, rake_new])
        if np.any(np.diff(ctrl_pts_r[0, :]) <= 0):
            raise ValueError('Rake-control radii must be strictly increasing.')
            
        degree = 3
        order = degree + 1
        N = ctrl_pts_r.shape[1]
        n_int = N - order
        
        if n_int > 0:
            interior = np.linspace(0, 1, n_int + 2)
            interior = interior[1:-1]
        else:
            interior = np.array([])
            
        t = np.concatenate([np.zeros(order), interior, np.ones(order)])
        spline_rake = spmak1(t, ctrl_pts_r)

        # --- 5) sample densely and invert r→rake
        u_sample = np.linspace(0, 1, 1000)
        xy_sample = fnval1(spline_rake, u_sample)
        r_curve = xy_sample[0, :]
        rake_curve = xy_sample[1, :]
        Rake = lambda x: interp1d(r_curve, rake_curve, kind='cubic',
                                bounds_error=False, fill_value='extrapolate')(x)

    # Calculate lengths and prepare control points matrix
    lengths = [
        len(R_new_skew), len(skew_new),
        len(R_new_chord), len(chord_new),
        len(R_new_thick), len(Thickness_new),
        len(R_new_pitch), len(pitch_new),
        len(R_new_camber), len(camber_new),
        len(R_new_rake), len(rake_new)
    ]
    maxLen = max(lengths)

    # Preallocate with NaNs
    control_points = np.full((12, maxLen), np.nan)

    # Fill each row, padding shorter rows automatically
    control_points[0, :len(R_new_skew)] = R_new_skew
    control_points[1, :len(skew_new)] = skew_new
    control_points[2, :len(R_new_chord)] = R_new_chord
    control_points[3, :len(chord_new)] = chord_new
    control_points[4, :len(R_new_thick)] = R_new_thick
    control_points[5, :len(Thickness_new)] = Thickness_new
    control_points[6, :len(R_new_pitch)] = R_new_pitch
    control_points[7, :len(pitch_new)] = pitch_new
    control_points[8, :len(R_new_camber)] = R_new_camber
    control_points[9, :len(camber_new)] = camber_new
    control_points[10, :len(R_new_rake)] = R_new_rake
    control_points[11, :len(rake_new)] = rake_new

    # Open file
    filename = f'control_points_{x1}.dat'
    with open(filename, 'w') as fid:
        # Write the 12×N control_points matrix, one row per line
        nRows, nCols = control_points.shape
        for i in range(nRows):
            for j in range(nCols):
                fid.write(f'{control_points[i, j]}\t')
            fid.write('\n')

    print(f'Wrote {filename} ({nRows} rows × {nCols} cols)')

    return MaxCamber, Pitch, ChordLength, MaxThickness, SkewAngle, Rake 