import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.interpolate import interp1d
from spmak1 import spmak1
from fnval1 import fnval1

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

class InteractiveCubicSplineControl:
    def __init__(self, ChordLength_origin, Pitch_origin, R_values, con_points_chord=4, con_points_pitch=5):
        """
        Interactive Cubic B-spline control with sliders
        
        Parameters:
        ChordLength_origin: original chord function
        Pitch_origin: original pitch function
        R_values: radial positions
        con_points_chord: number of control points for chord (default: 4)
        con_points_pitch: number of control points for pitch (default: 5)
        """
        self.ChordLength_origin = ChordLength_origin
        self.Pitch_origin = Pitch_origin
        self.R_values = R_values
        self.R1 = R_values[0]
        self.R_end = R_values[-1]
        self.con_points_chord = con_points_chord
        self.con_points_pitch = con_points_pitch
        
        # Create the figure and subplots
        self.fig = plt.figure(figsize=(16, 10))
        
        # Chord plot (left)
        self.ax_chord = plt.subplot(2, 2, 1)
        self.ax_chord.set_title('Chord Length Control via Cubic B-Spline')
        self.ax_chord.set_xlabel('radius (R)')
        self.ax_chord.set_ylabel('normalized chord')
        self.ax_chord.grid(True)
        
        # Pitch plot (right)
        self.ax_pitch = plt.subplot(2, 2, 2)
        self.ax_pitch.set_title('Pitch Control via Cubic B-Spline')
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
        plt.subplots_adjust(bottom=0.4)  # Make room for sliders
        
    def create_sliders(self):
        """Create all the sliders for chord and pitch control"""
        
        slider_height = 0.02
        slider_spacing = 0.025
        left_start = 0.1
        right_start = 0.55
        bottom_start = 0.35
        
        # Chord sliders (left side) - Variables: r2, frac23, chord0, d1, d2
        self.ax_chord_r2 = plt.axes([left_start, bottom_start - 0*slider_spacing, 0.35, slider_height])
        self.slider_chord_r2 = Slider(self.ax_chord_r2, '', self.R1+0.01, self.R_end-0.01, valinit=self.R1 + (self.R_end-self.R1)*0.4)
        
        self.ax_chord_frac23 = plt.axes([left_start, bottom_start - 1*slider_spacing, 0.35, slider_height])
        self.slider_chord_frac23 = Slider(self.ax_chord_frac23, '', 0.0, 1.0, valinit=0.5)
        
        self.ax_chord_chord0 = plt.axes([left_start, bottom_start - 2*slider_spacing, 0.35, slider_height])
        self.slider_chord_chord0 = Slider(self.ax_chord_chord0, '', 0.0, 1.0, valinit=0.25)
        
        self.ax_chord_d1 = plt.axes([left_start, bottom_start - 3*slider_spacing, 0.35, slider_height])
        self.slider_chord_d1 = Slider(self.ax_chord_d1, '', 0.0, 0.7, valinit=0.2)
        
        self.ax_chord_d2 = plt.axes([left_start, bottom_start - 4*slider_spacing, 0.35, slider_height])
        self.slider_chord_d2 = Slider(self.ax_chord_d2, '', 0.0, 0.7, valinit=0.3)
        
        # Pitch sliders (right side) - Variables: r2, frac23, pitch0, d1, d2, tip_frac
        self.ax_pitch_r2 = plt.axes([right_start, bottom_start - 0*slider_spacing, 0.35, slider_height])
        self.slider_pitch_r2 = Slider(self.ax_pitch_r2, '', self.R1+0.01, self.R_end-0.01, valinit=self.R1 + (self.R_end-self.R1)*0.35)
        
        self.ax_pitch_frac23 = plt.axes([right_start, bottom_start - 1*slider_spacing, 0.35, slider_height])
        self.slider_pitch_frac23 = Slider(self.ax_pitch_frac23, '', 0.0, 1.0, valinit=0.5)
        
        self.ax_pitch_pitch0 = plt.axes([right_start, bottom_start - 2*slider_spacing, 0.35, slider_height])
        self.slider_pitch_pitch0 = Slider(self.ax_pitch_pitch0, '', 0.5, 1.5, valinit=1.0)
        
        self.ax_pitch_d1 = plt.axes([right_start, bottom_start - 3*slider_spacing, 0.35, slider_height])
        self.slider_pitch_d1 = Slider(self.ax_pitch_d1, '', 0.0, 0.7, valinit=0.15)
        
        self.ax_pitch_d2 = plt.axes([right_start, bottom_start - 4*slider_spacing, 0.35, slider_height])
        self.slider_pitch_d2 = Slider(self.ax_pitch_d2, '', 0.0, 0.8, valinit=0.2)
        
        self.ax_pitch_tipfrac = plt.axes([right_start, bottom_start - 5*slider_spacing, 0.35, slider_height])
        self.slider_pitch_tipfrac = Slider(self.ax_pitch_tipfrac, '', 0.0, 0.95, valinit=0.5)
        
        # Set initial labels
        self.set_initial_labels()
        
    def set_initial_labels(self):
        """Set the initial labels for all sliders and hide value text"""
        # Chord sliders
        self.slider_chord_r2.label.set_text(f'Chord r2: {self.slider_chord_r2.val:.3f}')
        self.slider_chord_r2.valtext.set_visible(False)
        self.slider_chord_frac23.label.set_text(f'Chord frac23: {self.slider_chord_frac23.val:.2f}')
        self.slider_chord_frac23.valtext.set_visible(False)
        self.slider_chord_chord0.label.set_text(f'Chord chord0: {self.slider_chord_chord0.val:.2f}')
        self.slider_chord_chord0.valtext.set_visible(False)
        self.slider_chord_d1.label.set_text(f'Chord d1: {self.slider_chord_d1.val:.2f}')
        self.slider_chord_d1.valtext.set_visible(False)
        self.slider_chord_d2.label.set_text(f'Chord d2: {self.slider_chord_d2.val:.2f}')
        self.slider_chord_d2.valtext.set_visible(False)
        
        # Pitch sliders
        self.slider_pitch_r2.label.set_text(f'Pitch r2: {self.slider_pitch_r2.val:.3f}')
        self.slider_pitch_r2.valtext.set_visible(False)
        self.slider_pitch_frac23.label.set_text(f'Pitch frac23: {self.slider_pitch_frac23.val:.2f}')
        self.slider_pitch_frac23.valtext.set_visible(False)
        self.slider_pitch_pitch0.label.set_text(f'Pitch pitch0: {self.slider_pitch_pitch0.val:.2f}')
        self.slider_pitch_pitch0.valtext.set_visible(False)
        self.slider_pitch_d1.label.set_text(f'Pitch d1: {self.slider_pitch_d1.val:.2f}')
        self.slider_pitch_d1.valtext.set_visible(False)
        self.slider_pitch_d2.label.set_text(f'Pitch d2: {self.slider_pitch_d2.val:.2f}')
        self.slider_pitch_d2.valtext.set_visible(False)
        self.slider_pitch_tipfrac.label.set_text(f'Pitch tip_frac: {self.slider_pitch_tipfrac.val:.2f}')
        self.slider_pitch_tipfrac.valtext.set_visible(False)
        
    def connect_sliders(self):
        """Connect all sliders to the update function"""
        self.slider_chord_r2.on_changed(self.update_plots)
        self.slider_chord_frac23.on_changed(self.update_plots)
        self.slider_chord_chord0.on_changed(self.update_plots)
        self.slider_chord_d1.on_changed(self.update_plots)
        self.slider_chord_d2.on_changed(self.update_plots)
        
        self.slider_pitch_r2.on_changed(self.update_plots)
        self.slider_pitch_frac23.on_changed(self.update_plots)
        self.slider_pitch_pitch0.on_changed(self.update_plots)
        self.slider_pitch_d1.on_changed(self.update_plots)
        self.slider_pitch_d2.on_changed(self.update_plots)
        self.slider_pitch_tipfrac.on_changed(self.update_plots)
        
    def update_plots(self, val=None):
        """Update both chord and pitch plots based on slider values"""
        
        # Update slider labels with current values
        self.slider_chord_r2.label.set_text(f'Chord r2: {self.slider_chord_r2.val:.3f}')
        self.slider_chord_frac23.label.set_text(f'Chord frac23: {self.slider_chord_frac23.val:.2f}')
        self.slider_chord_chord0.label.set_text(f'Chord chord0: {self.slider_chord_chord0.val:.2f}')
        self.slider_chord_d1.label.set_text(f'Chord d1: {self.slider_chord_d1.val:.2f}')
        self.slider_chord_d2.label.set_text(f'Chord d2: {self.slider_chord_d2.val:.2f}')
        
        self.slider_pitch_r2.label.set_text(f'Pitch r2: {self.slider_pitch_r2.val:.3f}')
        self.slider_pitch_frac23.label.set_text(f'Pitch frac23: {self.slider_pitch_frac23.val:.2f}')
        self.slider_pitch_pitch0.label.set_text(f'Pitch pitch0: {self.slider_pitch_pitch0.val:.2f}')
        self.slider_pitch_d1.label.set_text(f'Pitch d1: {self.slider_pitch_d1.val:.2f}')
        self.slider_pitch_d2.label.set_text(f'Pitch d2: {self.slider_pitch_d2.val:.2f}')
        self.slider_pitch_tipfrac.label.set_text(f'Pitch tip_frac: {self.slider_pitch_tipfrac.val:.2f}')
        
        # Clear the plots
        self.ax_chord.clear()
        self.ax_pitch.clear()
        
        # Update chord plot
        self.update_chord_plot()
        
        # Update pitch plot
        self.update_pitch_plot()
        
        # Redraw
        self.fig.canvas.draw()
        
    def update_chord_plot(self):
        """Update the chord control plot"""
        # Get slider values - r2, frac23, chord0, d1, d2
        r2 = self.slider_chord_r2.val
        frac23 = self.slider_chord_frac23.val
        chord0 = self.slider_chord_chord0.val
        d1 = self.slider_chord_d1.val
        d2 = self.slider_chord_d2.val
        
        # Build control points per para_control_cub
        R_new_chord = np.zeros(4)
        R_new_chord[0] = self.R1
        R_new_chord[1] = r2
        R_new_chord[2] = R_new_chord[1] + (self.R_end - R_new_chord[1]) * frac23
        R_new_chord[3] = self.R_end
        
        # enforce monotonicity
        if R_new_chord[1] <= R_new_chord[0]:
            R_new_chord[1] = R_new_chord[0] + 0.01
        if R_new_chord[2] <= R_new_chord[1]:
            R_new_chord[2] = R_new_chord[1] + 0.01
        if R_new_chord[2] >= R_new_chord[3]:
            R_new_chord[2] = R_new_chord[3] - 0.01
        
        chord_new = np.zeros(4)
        chord_new[0] = chord0
        chord_new[1] = chord_new[0] + d1
        chord_new[2] = chord_new[0] + d2
        chord_new[3] = self.ChordLength_origin(self.R_end)
        
        # Pack into control points
        ctrl_pts = np.array([R_new_chord, chord_new])
        
        try:
            # Build clamped cubic B-spline
            degree = 3
            order = degree + 1
            N = ctrl_pts.shape[1]
            n_int = N - order
            
            if n_int > 0:
                interior = np.linspace(0, 1, n_int + 2)[1:-1]
            else:
                interior = np.array([])
            
            t = np.concatenate([np.zeros(order), interior, np.ones(order)])
            spline_chord = spmak1(t, ctrl_pts)
            
            # Sample the spline
            u_sample = np.linspace(0, 1, 1000)
            xy_sample = fnval1(spline_chord, u_sample)
            r_curve = xy_sample[0, :]
            chord_curve = xy_sample[1, :]
            
            # Create interpolation function
            chord_interp = interp1d(r_curve, chord_curve, kind='cubic',
                                   bounds_error=False, fill_value=(chord_curve[0], chord_curve[-1]))
            
            # Plot the result
            R_plot = np.linspace(self.R1, self.R_end, 300)
            chord_plot = chord_interp(R_plot)
            
            self.ax_chord.plot(R_plot, chord_plot, 'r-', linewidth=2, label='Cubic B-Spline')
            self.ax_chord.plot(R_new_chord, chord_new, 'ko', markersize=6, label='Control points')
            
        except Exception as e:
            print(f"Chord spline error: {e}")
        
        # Plot original curve
        self.ax_chord.plot(self.R_values, self.ChordLength_origin(self.R_values), 
                          'b--', linewidth=2, label='Original chord')
        
        self.ax_chord.set_title('Chord Length Control via Cubic B-Spline')
        self.ax_chord.set_xlabel('radius (R)')
        self.ax_chord.set_ylabel('normalized chord')
        self.ax_chord.legend()
        self.ax_chord.grid(True)
        
    def update_pitch_plot(self):
        """Update the pitch control plot"""
        # Get slider values - r2, frac23, pitch0, d1, d2, tip_frac
        r2 = self.slider_pitch_r2.val
        frac23 = self.slider_pitch_frac23.val
        pitch0 = self.slider_pitch_pitch0.val
        d1 = self.slider_pitch_d1.val
        d2 = self.slider_pitch_d2.val
        tip_frac = self.slider_pitch_tipfrac.val
        
        # Build control points per para_control_cub
        R_new_pitch = np.zeros(4)
        R_new_pitch[0] = self.R1
        R_new_pitch[1] = r2
        R_new_pitch[2] = R_new_pitch[1] + (self.R_end - R_new_pitch[1]) * frac23
        R_new_pitch[3] = self.R_end
        
        if R_new_pitch[1] <= R_new_pitch[0]:
            R_new_pitch[1] = R_new_pitch[0] + 0.01
        if R_new_pitch[2] <= R_new_pitch[1]:
            R_new_pitch[2] = R_new_pitch[1] + 0.01
        if R_new_pitch[2] >= R_new_pitch[3]:
            R_new_pitch[2] = R_new_pitch[3] - 0.01
        
        pitch_new = np.zeros(4)
        pitch_new[0] = pitch0
        pitch_new[1] = pitch_new[0] + d1
        pitch_new[2] = pitch_new[0] + d2
        pitch_new[3] = pitch_new[0] - pitch_new[0] * tip_frac
        
        # Pack into control points
        ctrl_pts_p = np.array([R_new_pitch, pitch_new])
        
        try:
            # Build clamped cubic B-spline
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
            
            # Sample the spline
            u_sample = np.linspace(0, 1, 1000)
            xy_sample = fnval1(spline_pitch, u_sample)
            r_curve = xy_sample[0, :]
            pitch_curve = xy_sample[1, :]
            
            # Create interpolation function
            pitch_interp = interp1d(r_curve, pitch_curve, kind='cubic',
                                  bounds_error=False, fill_value=(pitch_curve[0], pitch_curve[-1]))
            
            # Plot the result
            R_plot = np.linspace(self.R1, self.R_end, 300)
            pitch_plot = pitch_interp(R_plot)
            
            self.ax_pitch.plot(R_plot, pitch_plot, 'g-', linewidth=2, label='Cubic B-Spline')
            self.ax_pitch.plot(R_new_pitch, pitch_new, 'ks', markersize=6, label='Control points')
            
        except Exception as e:
            print(f"Pitch spline error: {e}")
        
        # Plot original curve
        self.ax_pitch.plot(self.R_values, self.Pitch_origin(self.R_values), 
                          'b--', linewidth=2, label='Original pitch')
        
        self.ax_pitch.set_title('Pitch Control via Cubic B-Spline')
        self.ax_pitch.set_xlabel('radius (R)')
        self.ax_pitch.set_ylabel('normalized pitch')
        self.ax_pitch.legend()
        self.ax_pitch.grid(True)

def create_interactive_cubic_spline_control(ChordLength_origin, Pitch_origin, R_values, 
                                          con_points_chord=4, con_points_pitch=5):
    """
    Create an interactive Cubic B-spline control interface
    
    Parameters:
    ChordLength_origin: original chord length function
    Pitch_origin: original pitch function  
    R_values: radial positions array
    con_points_chord: number of control points for chord (default: 4)
    con_points_pitch: number of control points for pitch (default: 5)
    
    Returns:
    InteractiveCubicSplineControl object
    """
    return InteractiveCubicSplineControl(ChordLength_origin, Pitch_origin, R_values,
                                       con_points_chord, con_points_pitch)

# Example usage:
if __name__ == "__main__":
    print("Interactive Cubic B-Spline Control")
    print("=" * 40)
    print("To use this interface, call it from your main script with:")
    print("interactive_control = create_interactive_cubic_spline_control(ChordLength_origin, Pitch_origin, R_values)")
    print("plt.show()")
    print()
    print("Where:")
    print("- ChordLength_origin: your original chord function")
    print("- Pitch_origin: your original pitch function") 
    print("- R_values: your radial positions array")
    print()
    print("Control Points:")
    print("Chord: P2x, P2y, P3x, P3y (R1 and R_end fixed)")
    print("Pitch: P1y, P2x, P2y, P3x, P3y, P4y (R1 and R_end fixed)")
    print("The interface will show both curves with the original curves as blue dashed lines")
    print("for comparison, and you can adjust the sliders to match them.")