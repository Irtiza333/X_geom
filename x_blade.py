import numpy as np
from scipy.spatial import cKDTree

from para import para
from para_control_bez_updated import para_control_bez_updated, APPLY_COUPLED_CONSTRAINTS
from rot_axis import rot_axis


DEFAULT_BEZIER_CONSTRAINT_MODE = "project"
CLEARANCE_THRESHOLD = 0.025



# Define the anonymous functions for each polynomial
def X_blade(
    pitch_con,
    chord_con,
    x1,
    *,
    return_bezier_info=False,
    bezier_constraint_mode=DEFAULT_BEZIER_CONSTRAINT_MODE,
    apply_coupled_constraints=None,
    write_dat=True,
):
    
    MaxCamber = lambda x: 1*(-4448.8369*x**12 + 30393.6831*x**11 + -92977.6043*x**10 + 168066.8833*x**9 + -199490.6759*x**8 + 163416.9800*x**7 + -94493.8152*x**6 + 38765.7447*x**5 + -11176.4246*x**4 + 2207.9559*x**3 + -285.0846*x**2 + 21.9443*x + -0.7490)

    Pitch = lambda x: 1*(19344.5071*x**12 + -114044.8587*x**11 + 280789.2801*x**10 + -357377.6146*x**9 + 207947.2705*x**8 + 43330.8173*x**7 + -173099.4797*x**6 + 143116.5772*x**5 + -65570.9523*x**4 + 18410.2055*x**3 + -3128.7530*x**2 + 294.0671*x + -10.4419)

    ChordLength = lambda x: 1*(-143202.4761*x**12 + 978274.9902*x**11 + -2992184.0323*x**10 + 5408923.9625*x**9 + -6424276.8851*x**8 + 5271614.6993*x**7 + -3058632.5267*x**6 + 1261908.7720*x**5 + -366745.1423*x**4 + 73096.4455*x**3 + -9470.1908*x**2 + 716.1619*x + -23.7157)

    MaxThickness = lambda x: 1*(-9688.7237*x**12 + 59807.8900*x**11 + -164159.4387*x**10 + 265158.4257*x**9 + -281726.9910*x**8 + 209033.5305*x**7 + -112447.0353*x**6 + 44837.0567*x**5 + -13266.8679*x**4 + 2814.8064*x**3 + -391.1763*x**2 + 29.0131*x + -0.4854)  # by chord

    SkewAngle = lambda x: 1*(-719361.0309*x**12 + 5176951.0162*x**11 + -16746858.1600*x**10 + 32161071.3897*x**9 + -40784306.1013*x**8 + 35930276.7571*x**7 + -22515881.4272*x**6 + 10096627.1844*x**5 + -3209956.6167*x**4 + 704272.6595*x**3 + -100947.8092*x**2 + 8462.8187*x + -312.8100)

    Rake = lambda x: 1*(-334.1390*x**12 + 1599.6222*x**11 + -2939.2891*x**10 + 2255.6140*x**9 + 0.0000*x**8 + -952.8241*x**7 + 0.0000*x**6 + 902.2268*x**5 + -804.8019*x**4 + 345.8731*x**3 + -81.8282*x**2 + 10.2137*x + -0.5245)  # Back calculated to line up with original CAD

    # Define the R values where we want to evaluate the polynomials
    R_values = np.concatenate([np.arange(0.17, 0.985, 0.018), [0.99, 0.999]])  # 100 points for smoother curves

    para_control_flag = 7  # control which parameter is being modified, does not have combined parameter effect

    (
        Pitch,
        ChordLength,
        chord_con_points,
        pitch_con_points,
        bezier_info,
    ) = para_control_bez_updated(
        Pitch,
        ChordLength,
        R_values,
        para_control_flag,
        pitch_con,
        chord_con,
        case_id=x1,
        constraint_mode=bezier_constraint_mode,
        return_reject_info=True,
        apply_coupled_constraints=apply_coupled_constraints,
    )

    use_coupled = (
        APPLY_COUPLED_CONSTRAINTS
        if apply_coupled_constraints is None
        else bool(apply_coupled_constraints)
    )
    if use_coupled and bezier_info["rejected"]:
        points = np.empty((0, 3))
        min_dis = np.inf
        constraint_violation = 1
        if return_bezier_info:
            return (
                points,
                min_dis,
                constraint_violation,
                chord_con_points,
                pitch_con_points,
                Pitch,
                ChordLength,
                bezier_info,
            )
        return points, min_dis, constraint_violation, chord_con_points, pitch_con_points, Pitch, ChordLength

    points = para(
        MaxCamber,
        Pitch,
        ChordLength,
        MaxThickness,
        SkewAngle,
        Rake,
        R_values,
        x1,
        write_dat=write_dat,
    )

    ax_rot = np.array([1, 0, 0])  # rotate about X axis

    R72  = rot_axis(ax_rot, np.deg2rad(72))


    # Rotate each blade from the original, not cumulatively
    points2 = points @ R72

    # Measure nearest distances between all points across blades using KD-tree
    tree2 = cKDTree(points2)
    dists12, idx12 = tree2.query(points, k=1)
    tree1 = cKDTree(points)
    dists21, idx21 = tree1.query(points2, k=1)

    # Global minimum distance (symmetric)
    min_dis = min(float(np.min(dists12)), float(np.min(dists21)))
   
    

    # Keep geometric-clearance failure separate from Bezier projection handling.
    constraint_violation = int(min_dis < CLEARANCE_THRESHOLD)

    if return_bezier_info:
        return (
            points,
            min_dis,
            constraint_violation,
            chord_con_points,
            pitch_con_points,
            Pitch,
            ChordLength,
            bezier_info,
        )

    return points, min_dis, constraint_violation, chord_con_points, pitch_con_points, Pitch, ChordLength



# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'k-', label='Blade 1')
# ax.plot3D(points2[:, 0], points2[:, 1], points2[:, 2], 'b-', label='Blade 2 (72°)')
# ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
# try:
#     ax.set_box_aspect((1, 1, 1))
# except Exception:
#     pass
# ax.legend()
# plt.show()
