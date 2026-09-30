import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from spmak1 import spmak1
from fnval1 import fnval1

def para_control_cub(Pitch, ChordLength, R_values, para_control, pitch_con, chord_con):
    """
    Parametric control using cubic clamped B-splines
    
    Parameters:
    MaxCamber, Pitch, ChordLength, MaxThickness, SkewAngle, Rake: original functions
    R_values: radial positions
    para_control: control parameter (2=chord, 4=pitch, 7=both)
    x1: additional parameter (not used in current implementation)
    con_points_chord: number of control points for chord spline (default: 4)
    con_points_pitch: number of control points for pitch spline (default: 5)
    
    Returns:
    Modified MaxCamber, Pitch, ChordLength, MaxThickness, SkewAngle, Rake functions
    
    Note: This implementation uses clamped cubic B-splines with:
    - Degree 3 (order 4) splines
    - Knot vectors with full multiplicity at endpoints
    - Random control point generation within specified bounds
    """
    
    # Store original functions
    Pitch_origin = Pitch
    ChordLength_origin = ChordLength
    
    
    # Skew: para_control==1 (currently not implemented)
    if para_control == 0:
        return Pitch, ChordLength
    
    # Chord Control
    if para_control == 2 or para_control == 7: 
        
        # 2) fix root & tip chords to original
        chord_tip = ChordLength_origin(R_values[-1])
        
        # 3) generate radial positions R_new_chord
        R_new_chord = np.zeros(5)
        chord_step = (R_values[-1] - R_values[0]) / (len(chord_con))
        R_new_chord[0]= R_values[0]
        for i in range(len(chord_con)):
            R_new_chord[i+1] = R_new_chord[i] + chord_step
               
        
        # 4) generate chord values chord_new
        chord_new = np.zeros(5)
        chord_new[0] = chord_con[0]
        chord_new[1] = chord_new[0]+ chord_con[1]
        chord_new[2] = chord_new[0]+ chord_con[2]

        if chord_new[2] > chord_new[1]:
            chord_new[3] = chord_new[1]+ chord_con[3]
        else:
            chord_new[3] = chord_new[2]- chord_con[3]*(chord_new[2]-chord_tip)

        chord_new[-1] = chord_tip
    
       
        # 5) pack into ctrl_pts and check monotonicity
        ctrl_pts = np.array([R_new_chord, chord_new])
        if np.any(np.diff(ctrl_pts[0, :]) <= 0):
            raise ValueError('Chord-control radii must be strictly increasing.')
        
        # 6) build a clamped cubic B-spline in B-form
        degree = 3
        order = degree + 1  # =4
        N = ctrl_pts.shape[1]
        n_int = N - order  # interior knots
        
        if n_int > 0:
            interior = np.linspace(0, 1, n_int + 2)[1:-1]
        else:
            interior = np.array([])
        
        t = np.concatenate([np.zeros(order), interior, np.ones(order)])
        spline_chord = spmak1(t, ctrl_pts)
        
        # 7) sample densely, then invert r→chord
        u_sample = np.linspace(0, 1, 1000)
        xy_sample = fnval1(spline_chord, u_sample)
        r_curve = xy_sample[0, :]
        chord_curve = xy_sample[1, :]
        
        # Create interpolation function
        ChordLength = lambda x: interp1d(r_curve, chord_curve, kind='cubic', 
                                       bounds_error=False, fill_value='extrapolate')(x)
        
        # # 8) visualize
        # R_plot = np.linspace(R_values[0], R_values[-1], 200)
        # chord_plot = ChordLength(R_plot)
        
        # plt.figure()
        # plt.plot(R_plot, chord_plot, 'k-', linewidth=2, label='Modified chord')
        # plt.plot(R_new_chord, chord_new, 'ro', markersize=8, markerfacecolor='r', label='Control points')
        # plt.plot(R_values, ChordLength_origin(R_values), 'b--', linewidth=1.5, label='Original chord')
        # plt.title('Chord Length Control via B-spline')
        # plt.xlabel('R')
        # plt.ylabel('Chord')
        # plt.legend()
        # plt.grid(True)
        # plt.show()
    
    # Pitch Control
    if para_control == 3 or para_control == 7:
    
        
        R_new_pitch = np.zeros(4)
        R_new_pitch[0] = R_values[0]
        R_new_pitch[1] = pitch_con[0]
        R_new_pitch[2] =  R_new_pitch[1] + (R_values[-1] -  R_new_pitch[1]) * pitch_con[1]
        R_new_pitch[-1] = R_values[-1]
        
        # 4) generate pitch values pitch_new
        pitch_new = np.zeros(4)
        pitch_new[0] = pitch_con[2]
        pitch_new[1] = pitch_new[0]+ pitch_con[3]
        pitch_new[2] = pitch_new[0]+ pitch_con[4]
        pitch_new[-1] = pitch_new[0]- pitch_new[0] * pitch_con[5]
        
        # 5) pack into ctrl_pts and check
        ctrl_pts_p = np.array([R_new_pitch, pitch_new])
        if np.any(np.diff(ctrl_pts_p[0, :]) <= 0):
            raise ValueError('Pitch-control radii must be strictly increasing.')
        
        # 6) build clamped cubic B-spline in B-form
        degree = 3
        order = degree + 1
        N = ctrl_pts_p.shape[1]
        n_int = N - order
        
        if n_int > 0:
            interior = np.linspace(0, 1, n_int + 2)[1:-1]
        else:
            interior = np.array([])
        
        t = np.concatenate([np.zeros(order), interior, np.ones(order)])
        spline_pitch = spmak1(t, ctrl_pts_p)
        
        # 7) sample densely, invert r→pitch
        u_sample = np.linspace(0, 1, 1000)
        xy_sample = fnval1(spline_pitch, u_sample)
        r_curve = xy_sample[0, :]
        pitch_curve = xy_sample[1, :]
        
        Pitch = lambda x: interp1d(r_curve, pitch_curve, kind='cubic',
                                 bounds_error=False, fill_value='extrapolate')(x)
        
        # # 8) visualize
        # R_plot = np.linspace(R_values[0], R_values[-1], 200)
        # pitch_dis = Pitch(R_plot)
        
        # plt.figure()
        # plt.plot(R_plot, pitch_dis, 'k-', linewidth=2, label='Modified Pitch')
        # plt.plot(R_new_pitch, pitch_new, 'ro', markersize=8, markerfacecolor='r', label='Control points')
        # plt.plot(R_values, Pitch_origin(R_values), 'b--', linewidth=1.5, label='Original Pitch')
        # plt.title('Pitch Control via B-spline')
        # plt.xlabel('R')
        # plt.ylabel('Pitch')  # Fixed: was 'Chord' in MATLAB
        # plt.legend()
        # plt.grid(True)
        # plt.show()
    
    return  Pitch, ChordLength