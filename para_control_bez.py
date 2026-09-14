import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def para_control_bez(Pitch, ChordLength,R_values, para_control, pitch_con, chord_con):
    """
    Parametric control using rational Bezier curves
    
    Parameters:
    MaxCamber, Pitch, ChordLength, MaxThickness, SkewAngle, Rake: original functions
    R_values: radial positions
    para_control: control parameter (2=chord, 3=pitch, 7=both)
    x1: additional parameter (not used in current implementation)
    
    Returns:
    Modified MaxCamber, Pitch, ChordLength, MaxThickness, SkewAngle, Rake functions
    """
    
    # Store original functions
   
    Pitch_origin = Pitch
    ChordLength_origin = ChordLength
    chord_con_points = np.zeros(5)
    pitch_con_points = np.zeros(6)
    
    # Chord Control
    if para_control == 2 or para_control == 7:
        chord_R1 = R_values[0]
        chord_R7 = R_values[-1]
        chord_p1x = R_values[0]  # chord P1.x
        
        # Variables (chord)
        chord_p1y = chord_con[0]
        chord_p4x = chord_con[1]
        chord_d1 = chord_con[2]
        chord_y4 = chord_con[3]  # chord P4.y

        chord_w23 = 0.4
        chord_w56 = chord_con[4]  # weight at P5=P6 (chord)
        
        # Build the seven Bézier anchors (with the required coincidences)
        chord_p2x = chord_p4x - (chord_p4x - chord_R1) * chord_d1  # P2.x = P3.x (chord)
        chord_P1 = np.array([chord_p1x, chord_p1y])
        chord_P2 = np.array([chord_p2x, chord_y4])
        chord_P3 = chord_P2  # coincide
        chord_P4 = np.array([chord_p4x, chord_y4])
        chord_P5 = np.array([R_values[-1], chord_y4])
        chord_P6 = chord_P5  # coincide
        chord_P7 = np.array([R_values[-1], ChordLength_origin(R_values[-1])])
        
        chord_con_points = np.array([chord_p1y, chord_p4x, chord_d1, chord_y4, chord_w56])
        # Homogeneous weights for each 4-point segment
        w_seg1 = np.array([1, chord_w23, chord_w23, 1])  # for [chord_P1..P4]
        # w_seg1 = np.array([1, 1, 1, 1])  # for [chord_P1..P4]
        w_seg2 = np.array([1, chord_w56, chord_w56, 1])  # for [chord_P4..P7]
        
        # Sample each rational Bézier
        u = np.linspace(0, 1, 600)
        R1_s, C1_s = eval_rational_bezier(np.column_stack([chord_P1, chord_P2, chord_P3, chord_P4]), w_seg1, u)
        R2_s, C2_s = eval_rational_bezier(np.column_stack([chord_P4, chord_P5, chord_P6, chord_P7]), w_seg2, u)
        
        # Stitch & interpolate (drop duplicate at the join)
        chord_R_curve = np.concatenate([R1_s, R2_s[1:]])
        chord_curve = np.concatenate([C1_s, C2_s[1:]])
        chord_interp = interp1d(
            chord_R_curve,
            chord_curve,
            kind='cubic',
            bounds_error=False,
            fill_value=(chord_curve[0], chord_curve[-1])
        )
        ChordLength = (lambda x, _f=chord_interp: _f(x))
        
        # # Visualize
        # R_plot = np.linspace(R1, R7, 300)
        # chord_plot = ChordLength(R_plot)
        # plt.figure()
        # plt.plot(R_plot, chord_plot, 'r-', linewidth=2, label='rational Bézier')
        # plt.plot([P1[0], P2[0], P4[0], P5[0], P7[0]], 
        #         [P1[1], P2[1], P4[1], P5[1], P7[1]], 'ko', markerfacecolor='k', label='design pts')
        # plt.plot(R_values, ChordLength_origin(R_values), 'b--', linewidth=1.5, label='original')
        # plt.xlabel('radius (R)')
        # plt.ylabel('normalized chord')
        # plt.legend()
        # plt.grid(True)
        # plt.show()
    
    # Pitch Control
    if para_control == 3 or para_control == 7:  # or your appropriate flag for pitch
        # 1) fixed radials
        pitch_R1 = R_values[0]
        pitch_R7 = R_values[-1]
        max_pitch = 1.4
        
        # 2) designer variables
        pitch_p1y = pitch_con[0]
        pitch_y4 = pitch_p1y + (max_pitch - pitch_p1y) * pitch_con[1]
        pitch_p4x = pitch_con[2]
        pitch_d1 = pitch_con[3]
        pitch_d2 = pitch_con[4]
        pitch_w23 = 0.4
        pitch_w56 = 0.4
        pitch_p7y = pitch_con[5]  # P7.y (tip pitch)
        # w56 = pitch_con[5]
        # p7y = pitch_con[6]

        # 3) anchors (note P2=P3 and P5=P6)
        pitch_p2x = pitch_p4x - (pitch_p4x - pitch_R1) * pitch_d1  # P2.x = P3.x
        pitch_p5x = pitch_p4x + (pitch_R7 - pitch_p4x) * pitch_d2  # P5.x = P6.x
        pitch_P1 = np.array([pitch_R1, pitch_p1y])
        pitch_P2 = np.array([pitch_p2x, pitch_y4])
        pitch_P3 = pitch_P2
        pitch_P4 = np.array([pitch_p4x, pitch_y4])
        pitch_P5 = np.array([pitch_p5x, pitch_y4])
        pitch_P6 = pitch_P5
        pitch_P7 = np.array([pitch_R7, pitch_p7y])
        
        # 4) set weights
        pitch_w1 = np.array([1, pitch_w23, pitch_w23, 1])  # for segment 1
        # pitch_w1 = np.array([1, 1, 1, 1])  # for segment 1 - use unit weights for pitch
        pitch_w2 = np.array([1, pitch_w56, pitch_w56, 1])  # for segment 2
        
        pitch_con_points = np.array([pitch_p1y, pitch_con[1], pitch_p4x, pitch_d1, pitch_d2, pitch_p7y])
        # 5) sample the two rational Béziers
        u = np.linspace(0, 1, 600)
        R1_s, Y1_s = eval_rational_bezier(np.column_stack([pitch_P1, pitch_P2, pitch_P3, pitch_P4]), pitch_w1, u)
        R2_s, Y2_s = eval_rational_bezier(np.column_stack([pitch_P4, pitch_P5, pitch_P6, pitch_P7]), pitch_w2, u)
        
        # 6) stitch & build interpolant
        pitch_R_curve = np.concatenate([R1_s, R2_s[1:]])  # drop duplicate join
        pitch_curve = np.concatenate([Y1_s, Y2_s[1:]])
        
        # Fixed: should modify Pitch, not create PitchLength
        pitch_interp = interp1d(
            pitch_R_curve,
            pitch_curve,
            kind='cubic',
            bounds_error=False,
            fill_value=(pitch_curve[0], pitch_curve[-1])
        )
        Pitch = (lambda x, _f=pitch_interp: _f(x))
        
        # # 7) visualize
        # R_plot = np.linspace(R1, R7, 300)
        # pitch_plot = Pitch(R_plot)  # Fixed: use Pitch instead of PitchLength
        # plt.figure()
        # plt.plot(R_plot, pitch_plot, 'g-', linewidth=2, label='rational Bézier')
        # plt.plot([P1[0], P2[0], P4[0], P5[0], P7[0]], 
        #         [P1[1], P2[1], P4[1], P5[1], P7[1]], 'ks', markerfacecolor='g', label='design pts')
        # plt.plot(R_values, Pitch_origin(R_values), 'b--', linewidth=1.5, label='original')
        # plt.xlabel('radius (R)')
        # plt.ylabel('normalized pitch')
        # plt.legend()
        # plt.grid(True)
        # plt.show()
    
    return Pitch, ChordLength, chord_con_points, pitch_con_points


def eval_rational_bezier(CP, w, u):
    """
    Helper function for one 4-point rational cubic Bézier
    
    Parameters:
    CP: 2×4 array of control points
    w: 1×4 weights
    u: 1×N parameter values
    
    Returns:
    X, Y: 1×N curves X(u), Y(u)
    """
    # Bernstein basis functions for cubic curves
    B0 = (1 - u) ** 3
    B1 = 3 * (1 - u) ** 2 * u
    B2 = 3 * (1 - u) * u ** 2
    B3 = u ** 3
    
    # Numerators (weighted sum of control points times basis functions)
    num = (CP[:, 0:1] * (w[0] * B0) + 
           CP[:, 1:2] * (w[1] * B1) + 
           CP[:, 2:3] * (w[2] * B2) + 
           CP[:, 3:4] * (w[3] * B3))
    
    # Denominator (sum of weights times basis functions)
    den = w[0] * B0 + w[1] * B1 + w[2] * B2 + w[3] * B3
    
    # Rational Bézier curve
    X = num[0, :] / den
    Y = num[1, :] / den
    
    return X, Y