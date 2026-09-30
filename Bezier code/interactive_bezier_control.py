import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.interpolate import interp1d

# Try to set a GUI backend
try:
    matplotlib.use('TkAgg')
    print("Using TkAgg backend")
except:
    try:
        matplotlib.use('Qt5Agg')
        print("Using Qt5Agg backend")
    except:
        print("Using default backend:", matplotlib.get_backend())

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

class InteractiveBezierControl:
    def __init__(self, ChordLength_origin, Pitch_origin, R_values):
        """
        Interactive Bezier curve control with sliders
        
        Parameters:
        ChordLength_origin: original chord function
        Pitch_origin: original pitch function
        R_values: radial positions
        """
        self.ChordLength_origin = ChordLength_origin
        self.Pitch_origin = Pitch_origin
        self.R_values = R_values
        self.R1 = R_values[0]
        self.R7 = R_values[-1]
        
        # Create the figure and subplots
        self.fig = plt.figure(figsize=(16, 10))
        
        # Chord plot (left)
        self.ax_chord = plt.subplot(2, 2, 1)
        self.ax_chord.set_title('Chord Length Control via Rational Bézier')
        self.ax_chord.set_xlabel('radius (R)')
        self.ax_chord.set_ylabel('normalized chord')
        self.ax_chord.grid(True)
        
        # Pitch plot (right)
        self.ax_pitch = plt.subplot(2, 2, 2)
        self.ax_pitch.set_title('Pitch Control via Rational Bézier')
        self.ax_pitch.set_xlabel('radius (R)')
        self.ax_pitch.set_ylabel('normalized pitch')
        self.ax_pitch.grid(True)
        
        # Create slider axes
        self.create_sliders()
        
        # Initial parameter values
        self.update_plots()
        
        # Connect sliders to update function
        self.connect_sliders()
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.35)  # Make room for sliders
        
    def create_sliders(self):
        """Create all the sliders for chord and pitch control"""
        
        # Chord control sliders
        slider_height = 0.02
        slider_spacing = 0.025
        left_start = 0.1
        right_start = 0.55
        bottom_start = 0.25
        
        # Chord sliders (left side)
        self.ax_chord_p1y = plt.axes([left_start, bottom_start - 0*slider_spacing, 0.35, slider_height])
        self.slider_chord_p1y = Slider(self.ax_chord_p1y, '', 0.0, 2.0, valinit=0.5)
        
        self.ax_chord_p4x = plt.axes([left_start, bottom_start - 1*slider_spacing, 0.35, slider_height])
        self.slider_chord_p4x = Slider(self.ax_chord_p4x, '', 0.0, 2.0, valinit=0.55)
        
        self.ax_chord_d1 = plt.axes([left_start, bottom_start - 2*slider_spacing, 0.35, slider_height])
        self.slider_chord_d1 = Slider(self.ax_chord_d1, '', 0.0, 2.0, valinit=0.2)
        
        self.ax_chord_y4 = plt.axes([left_start, bottom_start - 3*slider_spacing, 0.35, slider_height])
        self.slider_chord_y4 = Slider(self.ax_chord_y4, '', 0.0, 2.0, valinit=1.0)
        
        self.ax_chord_w56 = plt.axes([left_start, bottom_start - 5*slider_spacing, 0.35, slider_height])
        self.slider_chord_w56 = Slider(self.ax_chord_w56, '', 0.0, 2.0, valinit=1.0)

        # Display computed chord w23 (not a slider) using para_control_bez relationship
        init_w23 = self.compute_chord_w23(self.slider_chord_p4x.val, self.slider_chord_d1.val)
        self.text_chord_w23 = self.fig.text(
            left_start, bottom_start - 4*slider_spacing,
            f'Chord w23 (auto): {init_w23:.3f}'
        )
        
        # Pitch sliders (right side)
        self.ax_pitch_p1y = plt.axes([right_start, bottom_start - 0*slider_spacing, 0.35, slider_height])
        self.slider_pitch_p1y = Slider(self.ax_pitch_p1y, '', 0.0, 2.0, valinit=0.5)
        
        self.ax_pitch_y4 = plt.axes([right_start, bottom_start - 1*slider_spacing, 0.35, slider_height])
        self.slider_pitch_y4 = Slider(self.ax_pitch_y4, '', 0.0, 2.0, valinit=1.0)
        
        self.ax_pitch_p4x = plt.axes([right_start, bottom_start - 2*slider_spacing, 0.35, slider_height])
        self.slider_pitch_p4x = Slider(self.ax_pitch_p4x, '', 0.0, 2.0, valinit=0.55)
        
        self.ax_pitch_d1 = plt.axes([right_start, bottom_start - 3*slider_spacing, 0.35, slider_height])
        self.slider_pitch_d1 = Slider(self.ax_pitch_d1, '', 0.0, 2.0, valinit=0.15)
        
        self.ax_pitch_d2 = plt.axes([right_start, bottom_start - 4*slider_spacing, 0.35, slider_height])
        self.slider_pitch_d2 = Slider(self.ax_pitch_d2, '', 0.0, 2.0, valinit=0.15)
        
        self.ax_pitch_w23 = plt.axes([right_start, bottom_start - 5*slider_spacing, 0.35, slider_height])
        self.slider_pitch_w23 = Slider(self.ax_pitch_w23, '', 0.0, 2.0, valinit=1.0)
        
        self.ax_pitch_w56 = plt.axes([right_start, bottom_start - 6*slider_spacing, 0.35, slider_height])
        self.slider_pitch_w56 = Slider(self.ax_pitch_w56, '', 0.0, 2.0, valinit=1.0)
        
        self.ax_pitch_p7y = plt.axes([right_start, bottom_start - 7*slider_spacing, 0.35, slider_height])
        self.slider_pitch_p7y = Slider(self.ax_pitch_p7y, '', 0.0, 2.0, valinit=0.3)
        
        # Set initial labels
        self.set_initial_labels()
        
    def set_initial_labels(self):
        """Set the initial labels for all sliders and hide value text"""
        # Set labels and hide value display
        self.slider_chord_p1y.label.set_text(f'Chord P1y: {self.slider_chord_p1y.val:.2f}')
        self.slider_chord_p1y.valtext.set_visible(False)
        
        self.slider_chord_p4x.label.set_text(f'Chord P4x: {self.slider_chord_p4x.val:.2f}')
        self.slider_chord_p4x.valtext.set_visible(False)
        
        self.slider_chord_d1.label.set_text(f'Chord d1: {self.slider_chord_d1.val:.2f}')
        self.slider_chord_d1.valtext.set_visible(False)
        
        self.slider_chord_y4.label.set_text(f'Chord y4: {self.slider_chord_y4.val:.2f}')
        self.slider_chord_y4.valtext.set_visible(False)
        
        self.slider_chord_w56.label.set_text(f'Chord w56: {self.slider_chord_w56.val:.2f}')
        self.slider_chord_w56.valtext.set_visible(False)
        
        self.slider_pitch_p1y.label.set_text(f'Pitch P1y: {self.slider_pitch_p1y.val:.2f}')
        self.slider_pitch_p1y.valtext.set_visible(False)
        
        self.slider_pitch_y4.label.set_text(f'Pitch y4: {self.slider_pitch_y4.val:.2f}')
        self.slider_pitch_y4.valtext.set_visible(False)
        
        self.slider_pitch_p4x.label.set_text(f'Pitch P4x: {self.slider_pitch_p4x.val:.2f}')
        self.slider_pitch_p4x.valtext.set_visible(False)
        
        self.slider_pitch_d1.label.set_text(f'Pitch d1: {self.slider_pitch_d1.val:.2f}')
        self.slider_pitch_d1.valtext.set_visible(False)
        
        self.slider_pitch_d2.label.set_text(f'Pitch d2: {self.slider_pitch_d2.val:.2f}')
        self.slider_pitch_d2.valtext.set_visible(False)
        
        self.slider_pitch_w23.label.set_text(f'Pitch w23: {self.slider_pitch_w23.val:.2f}')
        self.slider_pitch_w23.valtext.set_visible(False)
        
        self.slider_pitch_w56.label.set_text(f'Pitch w56: {self.slider_pitch_w56.val:.2f}')
        self.slider_pitch_w56.valtext.set_visible(False)
        
        self.slider_pitch_p7y.label.set_text(f'Pitch P7y: {self.slider_pitch_p7y.val:.2f}')
        self.slider_pitch_p7y.valtext.set_visible(False)
        
    def connect_sliders(self):
        """Connect all sliders to the update function"""
        self.slider_chord_p1y.on_changed(self.update_plots)
        self.slider_chord_p4x.on_changed(self.update_plots)
        self.slider_chord_d1.on_changed(self.update_plots)
        self.slider_chord_y4.on_changed(self.update_plots)
        self.slider_chord_w56.on_changed(self.update_plots)
        
        self.slider_pitch_p1y.on_changed(self.update_plots)
        self.slider_pitch_y4.on_changed(self.update_plots)
        self.slider_pitch_p4x.on_changed(self.update_plots)
        self.slider_pitch_d1.on_changed(self.update_plots)
        self.slider_pitch_d2.on_changed(self.update_plots)
        self.slider_pitch_w23.on_changed(self.update_plots)
        self.slider_pitch_w56.on_changed(self.update_plots)
        self.slider_pitch_p7y.on_changed(self.update_plots)
        
    def update_plots(self, val=None):
        """Update both chord and pitch plots based on slider values"""
        
        # Update slider labels with current values
        self.slider_chord_p1y.label.set_text(f'Chord P1y: {self.slider_chord_p1y.val:.2f}')
        self.slider_chord_p4x.label.set_text(f'Chord P4x: {self.slider_chord_p4x.val:.2f}')
        self.slider_chord_d1.label.set_text(f'Chord d1: {self.slider_chord_d1.val:.2f}')
        self.slider_chord_y4.label.set_text(f'Chord y4: {self.slider_chord_y4.val:.2f}')
        self.slider_chord_w56.label.set_text(f'Chord w56: {self.slider_chord_w56.val:.2f}')
        
        self.slider_pitch_p1y.label.set_text(f'Pitch P1y: {self.slider_pitch_p1y.val:.2f}')
        self.slider_pitch_y4.label.set_text(f'Pitch y4: {self.slider_pitch_y4.val:.2f}')
        self.slider_pitch_p4x.label.set_text(f'Pitch P4x: {self.slider_pitch_p4x.val:.2f}')
        self.slider_pitch_d1.label.set_text(f'Pitch d1: {self.slider_pitch_d1.val:.2f}')
        self.slider_pitch_d2.label.set_text(f'Pitch d2: {self.slider_pitch_d2.val:.2f}')
        self.slider_pitch_w23.label.set_text(f'Pitch w23: {self.slider_pitch_w23.val:.2f}')
        self.slider_pitch_w56.label.set_text(f'Pitch w56: {self.slider_pitch_w56.val:.2f}')
        self.slider_pitch_p7y.label.set_text(f'Pitch P7y: {self.slider_pitch_p7y.val:.2f}')
        
        # Clear the plots
        self.ax_chord.clear()
        self.ax_pitch.clear()
        
        # Update chord plot
        self.update_chord_plot()
        
        # Update pitch plot
        self.update_pitch_plot()
        
        # Redraw
        self.fig.canvas.draw()

    def compute_chord_w23(self, p4x, d1):
        """Compute chord w23 from p4x and d1 using para_control_bez relationship."""
        R1 = self.R1
        a = 0.995
        x2 = p4x - d1
        r_exact = x2 + 0.78 * (p4x - x2)
        denom = (1 - a) * d1
        if abs(denom) < 1e-9:
            return 1.0
        s3 = (r_exact - a * (p4x - d1) - (1 - a) * R1) / denom
        s = np.cbrt(s3)
        denom2 = 3.0 * s * (1.0 + s)
        if abs(denom2) < 1e-9 or not np.isfinite(denom2):
            return 1.0
        W = (a / (1 - a) - s3) / denom2
        if not np.isfinite(W) or W <= 0:
            return 1.0
        return W
        
    def update_chord_plot(self):
        """Update the chord control plot"""
        # Get slider values
        p1y = self.slider_chord_p1y.val
        p4x = self.slider_chord_p4x.val
        d1 = self.slider_chord_d1.val
        y4 = self.slider_chord_y4.val
        w56 = self.slider_chord_w56.val
        
        # Ensure p4x > p1x + d1 for valid geometry
        p1x = self.R1
        if p4x <= p1x + d1:
            d1 = max(0.01, p4x - p1x - 0.01)
        
        # Compute w23 from p4x and d1 (auto)
        w23 = self.compute_chord_w23(p4x, d1)
        if hasattr(self, 'text_chord_w23'):
            try:
                self.text_chord_w23.set_text(f'Chord w23 (auto): {w23:.3f}')
            except Exception:
                pass
        
        # Build the seven Bézier anchors
        p2x = p4x - d1
        P1 = np.array([p1x, p1y])
        P2 = np.array([p2x, y4])
        P3 = P2  # coincide
        P4 = np.array([p4x, y4])
        P5 = np.array([self.R7, y4])
        P6 = P5  # coincide
        P7 = np.array([self.R7, self.ChordLength_origin(self.R7)])
        
        # Weights for each segment
        w_seg1 = np.array([1, w23, w23, 1])
        w_seg2 = np.array([1, w56, w56, 1])
        
        # Sample the rational Bézier curves
        u = np.linspace(0, 1, 600)
        R1_s, C1_s = eval_rational_bezier(np.column_stack([P1, P2, P3, P4]), w_seg1, u)
        R2_s, C2_s = eval_rational_bezier(np.column_stack([P4, P5, P6, P7]), w_seg2, u)
        
        # Stitch the curves
        R_curve = np.concatenate([R1_s, R2_s[1:]])
        chord_curve = np.concatenate([C1_s, C2_s[1:]])
        
        # Create interpolation function
        try:
            ChordLength_interp = interp1d(
                R_curve, chord_curve, kind='cubic',
                bounds_error=False, fill_value=(chord_curve[0], chord_curve[-1])
            )
            
            # Plot comparison
            R_plot = np.linspace(self.R1, self.R7, 300)
            chord_plot = ChordLength_interp(R_plot)
            
            self.ax_chord.plot(R_plot, chord_plot, 'r-', linewidth=2, label='Rational Bézier')
            self.ax_chord.plot([P1[0], P2[0], P4[0], P5[0], P7[0]], 
                             [P1[1], P2[1], P4[1], P5[1], P7[1]], 'ko', 
                             markerfacecolor='k', markersize=6, label='Control points')
            
        except Exception as e:
            print(f"Chord interpolation error: {e}")
        
        # Plot original curve
        self.ax_chord.plot(self.R_values, self.ChordLength_origin(self.R_values), 
                          'b--', linewidth=2, label='Original chord')
        
        self.ax_chord.set_title('Chord Length Control via Rational Bézier')
        self.ax_chord.set_xlabel('radius (R)')
        self.ax_chord.set_ylabel('normalized chord')
        self.ax_chord.legend()
        self.ax_chord.grid(True)
        
    def update_pitch_plot(self):
        """Update the pitch control plot"""
        # Get slider values
        p1y = self.slider_pitch_p1y.val
        y4 = self.slider_pitch_y4.val
        p4x = self.slider_pitch_p4x.val
        d1 = self.slider_pitch_d1.val
        d2 = self.slider_pitch_d2.val
        p7y = self.slider_pitch_p7y.val
        w23 = self.slider_pitch_w23.val
        w56 = self.slider_pitch_w56.val
        
        # Ensure valid geometry
        if p4x <= self.R1 + d1:
            d1 = max(0.01, p4x - self.R1 - 0.01)
        if p4x + d2 >= self.R7:
            d2 = max(0.01, self.R7 - p4x - 0.01)
        
        # Build control points
        p2x = p4x - d1
        p5x = p4x + d2
        P1 = np.array([self.R1, p1y])
        P2 = np.array([p2x, y4])
        P3 = P2
        P4 = np.array([p4x, y4])
        P5 = np.array([p5x, y4])
        P6 = P5
        P7 = np.array([self.R7, p7y])
        
        # Rational weights for pitch
        w1 = np.array([1, w23, w23, 1])
        w2 = np.array([1, w56, w56, 1])
        
        # Sample the rational Bézier curves
        u = np.linspace(0, 1, 600)
        R1_s, Y1_s = eval_rational_bezier(np.column_stack([P1, P2, P3, P4]), w1, u)
        R2_s, Y2_s = eval_rational_bezier(np.column_stack([P4, P5, P6, P7]), w2, u)
        
        # Stitch the curves
        R_curve = np.concatenate([R1_s, R2_s[1:]])
        pitch_curve = np.concatenate([Y1_s, Y2_s[1:]])
        
        # Create interpolation function
        try:
            Pitch_interp = interp1d(
                R_curve, pitch_curve, kind='cubic',
                bounds_error=False, fill_value=(pitch_curve[0], pitch_curve[-1])
            )
            
            # Plot comparison
            R_plot = np.linspace(self.R1, self.R7, 300)
            pitch_plot = Pitch_interp(R_plot)
            
            self.ax_pitch.plot(R_plot, pitch_plot, 'g-', linewidth=2, label='Rational Bézier')
            self.ax_pitch.plot([P1[0], P2[0], P4[0], P5[0], P7[0]], 
                             [P1[1], P2[1], P4[1], P5[1], P7[1]], 'ks', 
                             markerfacecolor='g', markersize=6, label='Control points')
            
        except Exception as e:
            print(f"Pitch interpolation error: {e}")
        
        # Plot original curve
        self.ax_pitch.plot(self.R_values, self.Pitch_origin(self.R_values), 
                          'b--', linewidth=2, label='Original pitch')
        
        self.ax_pitch.set_title('Pitch Control via Rational Bézier')
        self.ax_pitch.set_xlabel('radius (R)')
        self.ax_pitch.set_ylabel('normalized pitch')
        self.ax_pitch.legend()
        self.ax_pitch.grid(True)

def create_interactive_bezier_control(ChordLength_origin, Pitch_origin, R_values):
    """
    Create an interactive Bezier control interface
    
    Parameters:
    ChordLength_origin: original chord length function
    Pitch_origin: original pitch function  
    R_values: radial positions array
    
    Returns:
    InteractiveBezierControl object
    """
    return InteractiveBezierControl(ChordLength_origin, Pitch_origin, R_values)

# Example usage:
if __name__ == "__main__":
    
    print("Interactive Bezier Control - Creating Test Interface...")
    
    # Create test functions for demonstration
    def test_chord_origin(r):
        """Test chord function - linear taper"""
        return 0.6 - 0.3 * r
    
    def test_pitch_origin(r):
        """Test pitch function - slight curve"""
        return 0.4 + 0.2 * np.sin(np.pi * r / 2)
    
    # Test radial positions
    R_values = np.linspace(0.2, 1.0, 20)
    
    print("Creating interactive control with test functions...")
    print("- Red curve: Bezier chord (adjustable with sliders)")
    print("- Green curve: Bezier pitch (adjustable with sliders)")
    print("- Blue dashed: Original curves for comparison")
    print()
    print("Move the sliders to adjust the Bezier curves!")
    
    # Create and show the interactive interface
    interactive_control = create_interactive_bezier_control(
        test_chord_origin, 
        test_pitch_origin, 
        R_values
    )
    
    plt.show()