import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.interpolate import interp1d
from scipy.optimize import brentq

from Cross_sections import cross_section_points
from Untruncate_profile import untruncated_profile
from Bezier import bezier
from Thickness_curves import thickness_curves

H_max = 80
H_min = 1
L_max = 4*H_max
L_min = 1

surf_length = 350

N_sections = 10
sample_t = 5000

rng = np.random.default_rng()

values = pd.read_csv("corner_points.dat", sep = r"\s+", header = 6)
# Changing this to the full rudder - not to be limited to the small headbox region
# Better for changing parameters, visualising the whole rudder and what it will look like in 3D

# ========= Optimisation Space Definitions ==========

# LE
X_1 = np.array(values.iloc[:, 0])
Y_1 = np.array(values.iloc[:, 1])
# TE
X_2 = np.array(values.iloc[:, 3])
Y_2 = np.array(values.iloc[:, 4])
# Untuncated TE
X_3 = X_2
Y_3 = Y_2
# Rudder
X_r = np.array([X_1[0], X_1[len(X_1)-1], X_2[len(X_2)-1], X_2[0], X_1[0]])
Y_r = np.array([Y_1[0], Y_1[len(Y_1)-1], Y_2[len(Y_2)-1], Y_2[0], Y_1[0]])
# Wind Tunnel Surface
X_surf = np.array([X_1[0], -surf_length])
Y_surf = np.array([Y_1[0], 0])


# Defining Height, Length
H = rng.uniform(H_min, H_max)
LE_r = interp1d(Y_1, X_1, kind='linear')
Hx = LE_r(H)
Hy = H

PHx = np.interp(H, Y_3, X_3)
Para_height_x = np.array([X_3[0], PHx])
Para_height_y = np.array([Y_3[0], Y_3[0] + H])

L = rng.uniform(L_min, L_max)
Lx = X_1[0] - L
Ly = 0
Length_x = np.array([X_1[0], Lx])
Length_y = np.array([Y_1[0], Ly])

# Defining Leading Edge Curve
CX1 = rng.uniform(Lx, Hx)
CX2 = rng.uniform(Lx, Hx)
CY1 = rng.uniform(Y_1[0], Hy)
CY2 = rng.uniform(Y_1[0], Hy)

CX = np.array([Hx, CX1, CX2, Lx])
CY = np.array([Hy, CY1, CY2, Ly])

LEX, LEY = bezier(CX, CY, sample_t)

# Defining Cross-Sections, Chord Lengths
# First, defining rudder leading edge, parallel height, and leading edge chord as lines that can be interpolated on

# 1. Divide height by N sections
H_N = H/N_sections
Section_start = []
Section_end = []
Section_chord = []


# Interpolating Lines
curve = interp1d(LEX, LEY, kind='cubic')
line2 = interp1d(X_3, Y_3, kind='linear')

for i in range(N_sections + 1):
    # 2. Compute local height, find point on both height lines
    H_i = H_N * i
    H_end_x = LE_r(H_i)
    H_end_y = Y_1[0] + H_i
    # H_end_x, H_end_y = point_along_line((X_r[0], Y_r[0]), (X_r[1], Y_r[1]), H_i)
    # PH_end_x, PH_end_y = point_along_line((Para_height_x[0], Para_height_y[0]), (Para_height_x[1], Para_height_y[1]), H_i)
    PH_end_x = np.interp(H_i, Y_3, X_3)
    PH_end_y = Y_3[0] + H_i
    m = (PH_end_y - H_end_y) / (PH_end_x - H_end_x)
    b = H_end_y - m * H_end_x

    def line1(x):
        return m * x + b


    # 3. Compute intersection point between cross-section and rudder trailing edge / transition geometry leading edge
    # Cross-Section - Trailing Edge intersection: 

    def line2_difference(x):
        return line1(x) - line2(x)

    x_intersection_line2 = brentq(line2_difference, np.min(X_3), np.max(X_3))

    y_intersection_line2 = line1(x_intersection_line2)

    # Cross-Section - Leading Edge Curve intersection: 
    

    def curve_difference(x):
        return line1(x) - curve(x)

    x_intersection_curve = brentq(curve_difference, np.min(LEX), np.max(LEX))

    y_intersection_curve = line1(x_intersection_curve)

    # 4. Store intersection points as cross-section start, end points, and section chord
    Section_end.append([x_intersection_line2, y_intersection_line2])
    Section_start.append([x_intersection_curve, y_intersection_curve])
    chord = math.hypot((x_intersection_curve - x_intersection_line2), (y_intersection_curve - y_intersection_line2))
    Section_chord.append(chord)

Section_start = np.array(Section_start)
Section_end = np.array(Section_end)
Section_chord = np.array(Section_chord)
# ================= Thickness Control Curves =========================
T1_Y3 = 0.148
T2_Y3 = 0.1
X1_Y3 = 0.3
X2_Y3 = 0.57468
BxT1, ByT1, T1_X, T1_Y, BxX1, ByX1, X1_X, X1_Y, BxT2, ByT2, T2_X, T2_Y, BxX2, ByX2, X2_X, X2_Y = thickness_curves(T1_Y3, T2_Y3, X1_Y3, X2_Y3, sample_t)

# ================= Generating 3D Sections ===========================

# Following arrays placeholders for points that will form the 3D 
x_coords = []
y_coords = []
z_coords = []

for i in range(len(Section_chord)):
    
    # Determining cross-section geometry
    ci = Section_chord[i]
    Hcsi = Section_start[i][1] - Section_start[0][1]
    theta_i = math.atan2(Section_end[i][1]-Section_start[i][1], Section_end[i][0]-Section_start[i][0])

    # Determining control points from control curves
    x1_control = np.interp(Hcsi/H, BxX1, ByX1)
    z1_control = np.interp(Hcsi/H, BxT1, ByT1)
    x2_control = np.interp(Hcsi/H, BxX2, ByX2)
    z2_control = np.interp(Hcsi/H, BxT2, ByT2)

    Bxi, Bzi = cross_section_points(ci, z1_control, x1_control, x2_control, z2_control, bezier_sample=100)

    # Transforming 'x' coords into x, y coords for 3D geometry
    for j in range(len(Bxi)):
        x_coords.append(Bxi[j] * math.cos(theta_i) + Section_start[i][0])
        y_coords.append(Bxi[j] * math.sin(theta_i) + Section_start[i][1])
        z_coords.append(Bzi[j])
        if j != 0 or j != len(Bxi):
            x_coords.append(Bxi[j] * math.cos(theta_i) + Section_start[i][0])
            y_coords.append(Bxi[j] * math.sin(theta_i) + Section_start[i][1])
            z_coords.append(-Bzi[j])

x_coords = np.array(x_coords)
y_coords = np.array(y_coords)
z_coords = np.array(z_coords)

#=================== VISUALS =====================

# 2D Profile Plot
fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=100)

ax1.plot(X_r, Y_r, label = "Headbox Region", c = "blue")
ax1.plot(X_surf, Y_surf, label = "Surface", c="green")
ax1.plot(X_3, Y_3, label = "Unruncated Trailing Edge", c = "cyan")

ax1.plot([0, 0], [0, H], label = "Height", c="orange")
ax1.plot(Length_x, Length_y, label = "Length", c = "red")
# ax1.plot(Para_height_x, Para_height_y, label = "Parallel Height", linestyle = ":", c = "orange")

ax1.plot(LEX, LEY, label = "Leading Edge Curve", c = "purple")
ax1.plot(CX, CY, label = "Leading Edge Control Points", c = "purple", linestyle = ":", marker = "o")
for i in range(len(Section_start)):
    ax1.plot([Section_start[i][0], Section_end[i][0]], [Section_start[i][1], Section_end[i][1]], linestyle=":", color="grey", marker="o", label="Cross-Sections" if i == 0 else None)

ax1.set_aspect("equal")
ax1.legend()

# Thickness Control Curves
fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=100)

ax2.plot(BxT1, ByT1, label = "Thickness Curve 1", c = "red")
ax2.plot(T1_X, T1_Y, label = "TC 1 Control Points", c = "red", linestyle = ":", marker = "o")

ax2.plot(BxT2, ByT2, label = "Thickness Curve 2", c = "blue")
ax2.plot(T2_X, T2_Y, label = "TC 2 Control Points", c = "blue", linestyle = ":", marker = "o")

ax2.plot(BxX1, ByX1, label = "X Pos. Curve 1", c = "green")
ax2.plot(X1_X, X1_Y, label = "XP 1 Control Points", c = "green", linestyle = ":", marker = "o")

ax2.plot(BxX2, ByX2, label = "X Pos. Curve 2", c = "pink")
ax2.plot(X2_X, X2_Y, label = "XP 2 Control Points", c = "pink", linestyle = ":", marker = "o")

ax2.legend()
ax2.set_xlim(-0.1, 1.5)
ax2.set_ylim(-0.1, 1.1)
ax2.set_aspect("equal")

# 3D Geometry

fig3 = plt.figure(figsize=(8, 6), dpi = 100)
ax3 = fig3.add_subplot(111, projection = '3d')

ax3.scatter(x_coords, y_coords, z_coords, label = "Transition Geometry")
ax3.scatter(LEX, LEY, 0, label = "Leading Edge Profile", c ="purple")
ax3.scatter(X_1, Y_1, 0, label = "Rudder Leading Edge", c = "blue")
ax3.scatter(X_3, Y_3, 0, label = "Rudder Untruncated Trailing Edge", c = "cyan")

ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_aspect("equal")
ax3.set_xlim(x_coords[0] - 10, 200)
ax3.set_ylim(-1, 350)
ax3.set_zlim(-200, 200)

plt.show()



# # =========== CROSS SECTIONS =============

# c = 1
# y1 = 0.148
# x2 = 0.3
# x3 = 0.57468
# y3 = 0.1
# r = 0.01
# bezier_sample = 1000

# Bx, By, Bx_rounded, By_rounded = cross_section_points(c, y1, x2, x3, y3, r, bezier_sample)

# Bx_untrunc, By_untrunc, new_c = untruncated_profile(Bx_rounded, By_rounded, slope_tol=0.001)

# points = np.array([[0, 0, x2*c, x3*c, c], [0, y1*c, y1*c, y3*c, 0]])

# fig, ax = plt.subplots(figsize=(10, 6), dpi = 100)
# ax.plot(Bx, By, c = "blue", label = "Original Profile")
# ax.plot(points[0], points[1], label = "Original Control Points", c = "orange", linestyle = ":", marker = "o")
# ax.plot(Bx_rounded, By_rounded, label = "New Profile", c = "red")
# ax.plot(Bx_untrunc, By_untrunc, label = "Untruncated Profile", c = "green")
# ax.legend()
# ax.set_ylim(0, 0.6)
# ax.set_xlim(-0.01, 1.1)
# ax.set_aspect("equal")
# plt.show()