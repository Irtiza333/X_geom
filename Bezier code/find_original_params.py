import numpy as np
from para_control_bez import para_control_bez

# Original functions
Pitch_original = lambda x: 1*(19344.5071*x**12 + -114044.8587*x**11 + 280789.2801*x**10 + -357377.6146*x**9 + 207947.2705*x**8 + 43330.8173*x**7 + -173099.4797*x**6 + 143116.5772*x**5 + -65570.9523*x**4 + 18410.2055*x**3 + -3128.7530*x**2 + 294.0671*x + -10.4419)
ChordLength_original = lambda x: 1*(-143202.4761*x**12 + 978274.9902*x**11 + -2992184.0323*x**10 + 5408923.9625*x**9 + -6424276.8851*x**8 + 5271614.6993*x**7 + -3058632.5267*x**6 + 1261908.7720*x**5 + -366745.1423*x**4 + 73096.4455*x**3 + -9470.1908*x**2 + 716.1619*x + -23.7157)

R_values = np.concatenate([np.arange(0.17, 0.985, 0.018), [0.99, 0.999]])

# Get values at key points
R1 = R_values[0]
R7 = R_values[-1]

# For chord control - values that should reproduce the original
chord_p1y = ChordLength_original(R1)  # Chord at root
chord_y4 = 0.5  # A reasonable middle value
chord_p4x = 0.55  # Middle of the blade
chord_d1 = 0.2  # Distance parameters
chord_w56 = 1.0  # Weight

print("Suggested chord_con values for original curve:")
print(f"[{chord_p1y:.3f}, {chord_p4x:.3f}, {chord_d1:.3f}, {chord_y4:.3f}, {chord_w56:.3f}]")

# For pitch control - values that should reproduce the original  
pitch_p1y = Pitch_original(R1)  # Pitch at root
pitch_y4 = 1.0  # A reasonable middle value
pitch_p4x = 0.4  # Position of control point
pitch_d1 = 0.05  # Distance parameters
pitch_d2 = 0.3
pitch_w23 = 1.0  # Weights
pitch_w56 = 1.0
pitch_p7y = Pitch_original(R7)  # Pitch at tip

print("\nSuggested pitch_con values for original curve:")
print(f"[{pitch_p1y:.3f}, {pitch_y4:.3f}, {pitch_p4x:.3f}, {pitch_d1:.3f}, {pitch_d2:.3f}, {pitch_w23:.3f}, {pitch_w56:.3f}, {pitch_p7y:.3f}]")

# Test with these values
chord_con_original = [chord_p1y, chord_p4x, chord_d1, chord_y4, chord_w56]
pitch_con_original = [pitch_p1y, pitch_y4, pitch_p4x, pitch_d1, pitch_d2, pitch_w23, pitch_w56, pitch_p7y]

print("\nTesting flag 7 with original-matching parameters...")
Pitch7, Chord7, *_ = para_control_bez(Pitch_original, ChordLength_original, R_values, 7, pitch_con_original, chord_con_original)

# Check how close we are
R_test = np.linspace(R1, R7, 50)
chord_diff = np.max(np.abs(Chord7(R_test) - ChordLength_original(R_test)))
pitch_diff = np.max(np.abs(Pitch7(R_test) - Pitch_original(R_test)))

print(f"Max chord difference: {chord_diff:.6f}")
print(f"Max pitch difference: {pitch_diff:.6f}")

if chord_diff < 0.1 and pitch_diff < 0.1:
    print("\n✓ These parameters should work well for flag 7!")
else:
    print("\n⚠ Large differences detected. The parameters may need adjustment.")
