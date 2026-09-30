# initial sampling for surrogate model. 
# we have 5 variables for chord and 6 variables for pitch, total 11 variables.
# Using optimized latin hypercube sampling. that means each variable will be divided into 55 equal parts for
#  55 samples for the initial training set.

# Step 1: bring in the bounds for each variable.
# Step 2: normalize each variable
# Step 3: sample the initial training set using optimized latin hypercube sampling.
# Step 4: de-normalize the initial training set.
# Step 5: check if any combinations causes the constraint violation: blades intersect.
# Step 6: if there is no constraint violation, plot all the chord and pitch curves (using para_control_bez)
# Step 7: Store the input values (control points) to be used as input for the surrogate model.
# Step 8: if there is constraint violation, remove the combination from the training set.
# Step 9: call X_blade.py to generate the 3d blade coordinates.
# Step 10: call X_cad.py to generate the cad file. Store each file with a naming convention: sample_blade1.# iges

import numpy as np
from X_CAD import X_CAD
from x_blade import X_blade
from scipy.stats import qmc
import matplotlib.pyplot as plt

pitch_bounds = np.array([[0.8, 1.25], #p1y 
                    [0.05, 1.0], #y4
                    [0.35, 0.75], #p4x
                    [0.15, 0.5], #d1
                    [0.15, 0.5], #d2
                    [0.4, 0.7]])  #p7y

chord_bounds =  np.array([
                [0.2, 0.3],   # p1y
                [0.45, 0.85],   # p4x
                [0.15, 0.5],   # d1
                [0.4, 0.7],   # y4
                [0.1, 0.3],    # w56
            ])

N_SAMPLES = 5
SEED = 100

d_pitch = pitch_bounds.shape[0]   # 6
d_chord = chord_bounds.shape[0]   # 5
d_total = d_pitch + d_chord       # 11


bounds_all = np.vstack([pitch_bounds, chord_bounds])

# Build an optimized LHC in [0,1]^11
sampler = qmc.LatinHypercube(d=d_total, seed=SEED, optimization="random-cd")
U = sampler.random(N_SAMPLES)  # shape (55, 11), each column Latin-stratified

# Scale to physical bounds
X_phys = qmc.scale(U, bounds_all[:, 0], bounds_all[:, 1])  # shape (55, 11)

# Split back to your two groups
pitch_phys = X_phys[:, :d_pitch]   # (55, 6)
chord_phys = X_phys[:, d_pitch:]   # (55, 5)
R_fine = np.linspace(0.17, 0.998, 100)
R_pitch_probe = np.linspace(0.17, 0.998, 9)
R_chord_probe = np.linspace(0.17, 0.96, 9)
print(R_pitch_probe)
print(R_chord_probe)

# Prepare matched-color subplots for all samples
colors = plt.cm.tab20(np.linspace(0, 1, N_SAMPLES))
fig, (ax_pitch, ax_chord) = plt.subplots(1, 2, figsize=(12, 5))
ax_pitch.set_title('Pitch (all samples)')
ax_chord.set_title('Chord (all samples)')
ax_pitch.set_xlabel('radius (R)'); ax_pitch.set_ylabel('pitch')
ax_chord.set_xlabel('radius (R)'); ax_chord.set_ylabel('chord')
ax_pitch.grid(True); ax_chord.grid(True)

for i in range(N_SAMPLES):
    pitch_con = pitch_phys[i, :]
    chord_con = chord_phys[i, :]
    points, min_dis, constraint_violation, chord_con_points, pitch_con_points, Pitch, ChordLength = X_blade(pitch_con, chord_con, i)

    Pitch_curve_probe = Pitch(R_pitch_probe)
    ChordLength_curve_probe = ChordLength(R_chord_probe)
    actual_curves=np.concatenate([Pitch_curve_probe, ChordLength_curve_probe])
    
    # Store input values as space-separated .dat (append one line per sample)
    with open('input_curve_values.dat', 'a') as file:
        file.write(' '.join(f"{v:.6f}" for v in actual_curves) + "\n")
        
    # Matching color for pitch/chord across subplots for this sample
    
    c = colors[i % len(colors)]
    ax_pitch.plot(R_fine, Pitch(R_fine), color=c, linewidth=1.3, label=f's{i+1}')
    ax_chord.plot(R_fine, ChordLength(R_fine), color=c, linewidth=1.3, label=f's{i+1}')

    if constraint_violation == 1:
        print(f"Constraint violation for sample {i}")
        print(min_dis)
        continue
    else:
        print(f"No constraint violation for sample {i}")
        input_values = np.concatenate([pitch_con_points, chord_con_points])
        
    X_CAD(points, i)

    # Store input values as space-separated .dat (append one line per sample)
    with open('input_con_values.dat', 'a') as file:
        file.write(' '.join(f"{v:.6f}" for v in input_values) + "\n")

# Finalize plots once after loop
handles1, labels1 = ax_pitch.get_legend_handles_labels()
if len(labels1) <= 15:
    ax_pitch.legend(ncol=2, fontsize=8)
handles2, labels2 = ax_chord.get_legend_handles_labels()
if len(labels2) <= 15:
    ax_chord.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig('Initial_samples.png', dpi=200, bbox_inches='tight')
plt.close(fig)






      