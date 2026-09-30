import numpy as np
from para_control_bez import para_control_bez

# Original functions
Pitch_original = lambda x: 1*(19344.5071*x**12 + -114044.8587*x**11 + 280789.2801*x**10 + -357377.6146*x**9 + 207947.2705*x**8 + 43330.8173*x**7 + -173099.4797*x**6 + 143116.5772*x**5 + -65570.9523*x**4 + 18410.2055*x**3 + -3128.7530*x**2 + 294.0671*x + -10.4419)
ChordLength_original = lambda x: 1*(-143202.4761*x**12 + 978274.9902*x**11 + -2992184.0323*x**10 + 5408923.9625*x**9 + -6424276.8851*x**8 + 5271614.6993*x**7 + -3058632.5267*x**6 + 1261908.7720*x**5 + -366745.1423*x**4 + 73096.4455*x**3 + -9470.1908*x**2 + 716.1619*x + -23.7157)

R_values = np.concatenate([np.arange(0.17, 0.985, 0.018), [0.99, 0.999]])

# Your original control parameters
pitch_con = [1.23, 1.29, 0.38, 0.05, 0.30, 0.91, 0.37, 0.62]
chord_con = [0.23, 0.70, 0.24, 0.46, 0.19]

print("Testing each flag independently...")
print("="*50)

# Test flag 2 (chord only)
print("\nFlag 2 (Chord only):")
Pitch2, Chord2, *_ = para_control_bez(Pitch_original, ChordLength_original, R_values, 2, pitch_con, chord_con)
test_r = [0.17, 0.5, 0.999]
for r in test_r:
    print(f"  R={r:.3f}: Chord orig={ChordLength_original(r):.4f}, modified={Chord2(r):.4f}")
    print(f"         : Pitch orig={Pitch_original(r):.4f}, modified={Pitch2(r):.4f}")

# Test flag 3 (pitch only)  
print("\nFlag 3 (Pitch only):")
Pitch3, Chord3, *_ = para_control_bez(Pitch_original, ChordLength_original, R_values, 3, pitch_con, chord_con)
for r in test_r:
    print(f"  R={r:.3f}: Chord orig={ChordLength_original(r):.4f}, modified={Chord3(r):.4f}")
    print(f"         : Pitch orig={Pitch_original(r):.4f}, modified={Pitch3(r):.4f}")

# Test flag 7 (both)
print("\nFlag 7 (Both):")
Pitch7, Chord7, *_ = para_control_bez(Pitch_original, ChordLength_original, R_values, 7, pitch_con, chord_con)
for r in test_r:
    print(f"  R={r:.3f}: Chord orig={ChordLength_original(r):.4f}, modified={Chord7(r):.4f}")
    print(f"         : Pitch orig={Pitch_original(r):.4f}, modified={Pitch7(r):.4f}")

print("\n" + "="*50)
print("COMPARISON:")
print("Flag 2 should modify chord, leave pitch unchanged")
print("Flag 3 should modify pitch, leave chord unchanged")  
print("Flag 7 should apply BOTH modifications")
print("\nIf flag 7 results differ from applying both flag 2 and 3 changes,")
print("there's a bug in the implementation.")
