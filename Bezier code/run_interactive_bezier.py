"""
Interactive Bezier Control with Real Original Functions
This script uses your actual original polynomial functions from updated_bookcode.py
"""

import numpy as np
import matplotlib.pyplot as plt
from interactive_bezier_control import create_interactive_bezier_control

# Define your REAL original functions (from updated_bookcode.py)
MaxCamber = lambda x: 1*(-4448.8369*x**12 + 30393.6831*x**11 + -92977.6043*x**10 + 168066.8833*x**9 + -199490.6759*x**8 + 163416.9800*x**7 + -94493.8152*x**6 + 38765.7447*x**5 + -11176.4246*x**4 + 2207.9559*x**3 + -285.0846*x**2 + 21.9443*x + -0.7490)

Pitch = lambda x: 1*(19344.5071*x**12 + -114044.8587*x**11 + 280789.2801*x**10 + -357377.6146*x**9 + 207947.2705*x**8 + 43330.8173*x**7 + -173099.4797*x**6 + 143116.5772*x**5 + -65570.9523*x**4 + 18410.2055*x**3 + -3128.7530*x**2 + 294.0671*x + -10.4419)

ChordLength = lambda x: 1*(-143202.4761*x**12 + 978274.9902*x**11 + -2992184.0323*x**10 + 5408923.9625*x**9 + -6424276.8851*x**8 + 5271614.6993*x**7 + -3058632.5267*x**6 + 1261908.7720*x**5 + -366745.1423*x**4 + 73096.4455*x**3 + -9470.1908*x**2 + 716.1619*x + -23.7157)

MaxThickness = lambda x: -9688.7237*x**12 + 59807.8900*x**11 + -164159.4387*x**10 + 265158.4257*x**9 + -281726.9910*x**8 + 209033.5305*x**7 + -112447.0353*x**6 + 44837.0567*x**5 + -13266.8679*x**4 + 2814.8064*x**3 + -391.1763*x**2 + 29.0131*x + -0.4854

SkewAngle = lambda x: 1*(-719361.0309*x**12 + 5176951.0162*x**11 + -16746858.1600*x**10 + 32161071.3897*x**9 + -40784306.1013*x**8 + 35930276.7571*x**7 + -22515881.4272*x**6 + 10096627.1844*x**5 + -3209956.6167*x**4 + 704272.6595*x**3 + -100947.8092*x**2 + 8462.8187*x + -312.8100)

Rake = lambda x: 1*(-334.1390*x**12 + 1599.6222*x**11 + -2939.2891*x**10 + 2255.6140*x**9 + 0.0000*x**8 + -952.8241*x**7 + 0.0000*x**6 + 902.2268*x**5 + -804.8019*x**4 + 345.8731*x**3 + -81.8282*x**2 + 10.2137*x + -0.5245)

# Define R_values (same as in your code)
R_values = np.linspace(0.15, 1, 100)

print("Interactive Bezier Control with REAL Original Functions")
print("=" * 60)
print("Now using your actual polynomial functions from updated_bookcode.py")
print()
print("Blue dashed lines = Your REAL original chord and pitch curves")
print("Red/Green lines = Bezier approximations (adjustable with sliders)")
print()
print("Instructions:")
print("1. The blue dashed lines show your actual original functions")
print("2. Use the sliders to adjust the Bezier curves to match the blue lines")
print("3. Try to get the red (chord) and green (pitch) curves as close as possible")
print("   to the blue dashed originals")
print()

# Create the interactive interface with your REAL functions
interactive_control = create_interactive_bezier_control(
    ChordLength,  # Your REAL chord function
    Pitch,        # Your REAL pitch function  
    R_values      # Your radial positions
)

plt.show()

print("\nInterface created! You should now see:")
print("- Chord plot (left): Blue dashed = real chord, Red = Bezier approximation")
print("- Pitch plot (right): Blue dashed = real pitch, Green = Bezier approximation")
print("- Sliders below: Adjust to match the Bezier curves to the blue originals")