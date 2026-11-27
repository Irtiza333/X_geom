"""
xgeom_tool.py

XGeom design tool: set up a geometry's design space with sliders and see the
shape change live. The build button writes the case and its CAD; OCC viewer
shows the solid.

    python xgeom_tool.py                    the vehicle: hull, propeller, rudder, fin and wing in
                                            one window (the rudder alone where xgeom_hull.py is
                                            not there, as in the Rudder folder)
    python xgeom_tool.py rudder             the rudder alone
    python xgeom_tool.py rudder --file outputs/modified/gui.txt
                                            start from a saved set-up (a component file)
    python xgeom_tool.py blade              the propeller blade of blade_control.PARAMS (msc_blade.txt)
    python xgeom_tool.py blade --file my_blade.txt
                                            the blade of another component file
    python xgeom_tool.py hull               the hull of suboff_hull.txt
    python xgeom_tool.py hull --file my_hull.txt
    python xgeom_tool.py hull --file bwb_glider.txt
                                            the Hull slot's other body, the blended wing body
    python xgeom_tool.py wing               the wing of wing.txt (fin: fin.txt)
    python xgeom_tool.py fin --file my_fin.txt
    python xgeom_tool.py vehicle --file my_hull.txt --file my_blade.txt
                                            the vehicle: --file (a hull's, a blade's, a rudder's,
                                            a wing's or a fin's) as often as needed, each goes
                                            to its component

Vehicle
    Component    the hull, the propeller, the rudder, the fin or the wing: its left
                 panel and its Design and 3D sections tabs, as when it is alone; each
                 keeps its state while another is shown.
    Vehicle tab  the hull with the propeller and the rudder mounted at its stern, the
                 fins and the pair of wings on it (xgeom_vehicle.py), as their designs
                 stand; the component being edited is drawn thicker. Placement
                 (positions along the hull as x/L: 0 the nose, or the front of a
                 body without one, 1 the hull's aft end; since 13 Nov 2026):
                 propeller  hub starts at x/L: the hub's forward end on the axis,
                            the blades at the hub's own offset (1: the hub's base on
                            the hull's aft end; default where the hub sits flush on
                            the hull); D: its diameter in the vehicle, the design and
                            its hub scaled to it (default the design's own when it is
                            no larger than the hull's depth or breadth, else half that)
                 rudder     root TE at x/L: where the root's trailing edge sits on
                            the hull (0 to 1; default the start of the cap, a quarter
                            of the propeller's D clear of the blades); rudders: how
                            many, 0 (none) to 12, evenly spaced
                            around the axis (default 4, a cross); first at: the first one's
                            angle around the axis from the top towards starboard (180
                            under the stern); scale (1: its own size, mm to m). Each
                            rudder sits on the hull and follows its slope: the root's
                            TE on the hull at x, its LE on the hull one root chord
                            ahead, the rudder pitched to match (again whenever it is
                            moved or scaled, or the hull changes); a copy of the root
                            section extruded inwards along the rudder's height (dark
                            orange) closes the gaps the flat root leaves on the curved
                            hull. TE rake ticked: the rudder's angle is set instead,
                            its TE line that many deg from square to the axis (top aft
                            positive; 0 square to it, as BB2's stern planes), meeting
                            the hull at x; the rudder rests on the hull (the root's
                            closest point on it) and its root section is extruded along
                            the TE line until it meets the hull fully (the largest gap
                            under it and 2 % of the root chord). Unticked, the box
                            shows the TE's rake that the hull's slope gives.
                 fins       root LE at x/L: where the root's leading edge sits on the
                            hull (default a tenth of the way along the middle body);
                            fins: how many, 0 to 12, evenly spaced around
                            the axis (default 1); first at (default 0, on top); scale.
                            Each sits on the hull as a rudder does, its TE one root
                            chord aft of its LE along the hull, pitched to the hull's
                            slope (its own pitch on top), its root extruded inwards.
                 wings      root LE at x/L as the fins' (default a quarter of the way
                            along the middle body); wings: 2 (the pair) or 0 (none);
                            at: the starboard wing's angle around the axis from the top
                            (0 to 180 deg; default 90, level); the port wing is its
                            mirror image (y to -y): the same dihedral, camber and pitch
                            on both sides; scale.
                 show       the whole vehicle or its stern (from the start of the tail).
                 Vehicle CAD  every component's own case and CAD (as its own button
                            writes them, the case named in the left panel), then the
                            parts made solid, placed as the tab shows them and fused
                            (vehicle_cad.py): outputs/vehicle/<case>_vehicle.step (one
                            solid, mm, the hull's frame), <case>_vehicle_parts.step (the
                            parts placed, not fused) and <case>_vehicle.json (the
                            placement). Each rudder, fin and wing is its lofted XCAD
                            solid with the root extrusion (standing 0.2 mm off the hull
                            on it, so that the fuse is clean); the propeller its blades,
                            closed at the root on the hub cylinder, and the hub. A
                            component whose count is 0 is not built. A few minutes (the
                            propeller's CAD takes most); needs pythonocc-core.
Left panel (rudder)
    Rudder    the Rudder slot's two options (26 Nov 2026; where xgeom_wing.py is):
              modify an existing rudder (the panel below: the original stays
              above H, is changed below it and is there the lower bound of the
              shape), or design a new rudder (the planar model, the wing's panel:
              a rudder of its own, no original; rudder.txt to start with). Each
              keeps its state while the other is shown. A new rudder saved with
              Save set-up loads in the other option as the original to modify.
    File      the set-up file loaded (or saved with Save set-up) last, and
              Load ...: a set-up that Save set-up or an XCAD run wrote (a
              component file of kind rudder_h) takes the place of the design in
              the tool (the original's files, the section degree, every curve's
              order and its join or bounding laws, every variable's value,
              bounds, free flag and relation, the H bounds, the full height, the
              number of sections, the TE radius, the half body, the strips, the
              original as the lower bound). A file that is not a set-up changes
              nothing. The case name is left as it is, so the next XCAD writes
              to the case typed there.
    H         the height of the modified region, between its low and high
              bounds (h_min, h_max); h_max can go up to the top horizontal section
              of the stack (z_full - d, shown under 'where').
    full height
              the rudder's height from the root to the top of the cap (the
              extracted rudder's to start with): a new one stretches the whole
              rudder in height, H, its bounds and the curves' heights with it;
              saved with the set-up, not a design variable.
    tangent at H
              every curve's C1 on the original's tangent at H (slope matched;
              the default), or, unticked, every C1 free: a design variable
              starting where it is, so the curves meet the original with their
              value only. Each curve's box has its own switch for one curve. A
              design with any curve not tangent gets its CAD as two lofts, below
              and above H, joined at the H section (xcad_loft.loft_split: nine
              patches and the root), so the turn at H stays a crease instead of
              being smoothed over and rippling.
    keep the original inside
              the original is the lower bound of the shape: the tool measures,
              on the built sections, how far the modified rudder goes inside the
              original (shown beside the switch) and warns beyond the tolerance
              (mm). Switch and tolerance are saved with the set-up, for an
              optimiser to keep (rudder_modify.inside_original).
    Section control points
              the number of control points of the section's Bezier half-
              section, P0 .. Pn (P0 = (0, 0) the LE and Pn = (1, 0) the sharp TE
              stay put; adding one keeps the shape and adds that point's x and z
              curves). Pick the LE or the x or z of an inner point to edit the
              spanwise curve it follows; "held" marks a curve with no free
              variable. P1 x is held at 0 by default (a round LE).
    Curve     the picked curve's control points C0 .. Cm, from H (C0, pinned to
              the original) down to the root (Cm); how many there are; whether
              C1 sits on the original's tangent at H (slope matched; see
              tangent at H).
              low law, high law: the curve between two bounding laws, each a
              number or an expression of the curves at the same height (P3z,
              orig.P2z, LE, TE, chord, orig.chord; + - * / ( ) min( ) max( )).
              Its rows are then the fractions of the way from the low law (0)
              to the high one (1), so it lies between them whatever they are,
              and it joins the original at H with its value only. Low and high
              the same set the curve: no rows. Both empty: no laws.
              rudder_laws_setup.py writes the set-up that builds on the original.
Left panel (blade)
    File      the blade's component file (xgeom_files.py, claude/xgeom-file-formats.md:
              D, blades, root r/R, hub height, the section, the settings, each
              curve's form and every variable's value, bounds and free flag, the
              radial table). Load reads another one, and the whole panel follows
              it (a table file, no curve or var lines: the curves are fitted to
              it, a parameter flagged N on its vary line as the file has it, one
              flagged Y with its variables free, within low .. high when given).
              Save set-up (below) writes the design as it stands as a component
              file.
    Blade and hub
              the blade radius R, the hub radius and the hub height (m; the hub
              is centred at x = 0 to start with and must cover the blade root).
              A change of either radius stretches the distributions over the
              new span. hub extends (26 Nov 2026): which end moves when the hub
              height changes, fore (the upstream end), aft (the tip) or both
              (half each); the hub's own to start with (a cylinder both, a cone
              aft). Saved with the set-up, so an optimiser moves the same end.
              hub: a cylinder (the above), or a cone (13 Nov 2026, prop_hub.py):
              three more variables, the base radius (the larger end, upstream,
              at the base's x) and the top radius, where the cone meets its
              rounded nose, the sphere tangent to the cone through the tip, the
              hub height from the base; the hub radius is then the blade root
              radius, where the table starts, and the blade runs on into the
              hub, which trims it in the CAD. All held to start with.
              blades Z: the number of blades, 2 to 7 (the file's to start
              with); the clearance check, the propeller tab, the vehicle and
              the CAD's hub sector follow it. It is saved with the set-up; it
              is not in the design vector. Blades
              closer than the clearance (they interfere) give a warning, not a
              problem: moving the root section (the hub radius) or the root's
              chord and pitch may make room for them.
    Sections  the sections of the XCAD points file (para.py's, 53 points
              each) and the 3D sections tab: how many, root to tip, and how
              many of them lie from the tip band's r/R to the tip, closer
              together towards it (the others evenly spaced from the root).
              The clearance check keeps x_blade_new's stations.
    Distributions
              pick pitch, chord, thickness, camber, skew or rake to edit its
              curve, a Bezier over r/R fitted to the file. Curve: 1 segment (C0
              at the root .. Cm at the tip, as many as set) or 2 segments
              (x_blade's form: rational cubics P1 .. P4 and P4 .. P7, P2 = P3 and
              P5 = P6 at P4's value, so zero slope at P4; the weights w23 and w56
              are variables) or table (the distribution as the file's table has
              it, the cubic spline through its rows: no variables, the picker
              notes "(table)"). Switching, or a new number of control points,
              refits the curve: to the file's table while none of its variables
              has changed, else to its present shape.
Left panel (hull)
    File      the hull's component file (the nose, middle-body, tail and cap
              lengths, each curve's order, every variable, the station table: the
              half-height r and with elliptical sections the half-width r'). Load
              reads another one (a table file, the stations alone: the curves
              fitted to it); Save set-up writes the design as one.
    Body      the four lengths, the largest radius r* and the tail end's re (m);
              elliptical sections: the half-width r' (XY view) gets radii and
              curves of its own, r is then the half-height (XZ view).
    Curves    nose r and tail r (and nose r', tail r'), each a CST curve: its
              exponent (N1 the nose's bluntness, N2 the shape of the tail end)
              and its control values; the two control points at the middle body
              are pinned (r* with zero slope there). A new number of control
              points refits the curve as for the blade.
Left panel (hull): Body
              the Hull slot holds one of two bodies (20 Nov 2026): the cylindrical
              hull (above) or the blended wing body (xgeom_wing.BwbAdapter, kind
              bwb, bwb_glider.txt to start with: the wing's panel below over the
              half-span from the centre line, every curve flat there, the views
              and the 3D sections with the other half mirrored; CAD BWB writes the
              whole body, the half and its mirror image fused, or with the half
              switch the half alone). Each keeps its state while the other is
              shown; Load ... (or --file) of either body's file puts the slot on
              it. The Vehicle tab places its parts on the cylindrical hull only:
              with the blended wing body in the slot it says so (the blended wing
              body's vehicle comes later).
Left panel (wing, fin)
    File      the wing's (or fin's) component file (the span, the section by name
              or as a coordinate file, the pitch axis, the settings, each curve's
              form, every variable, and per row the pitch, chord, t/c, f/c, sweep
              and rake (dihedral)). Load and Save set-up as for the blade.
    Span and sections
              the span (mm; the curves stretch over it); the sections of the XCAD
              file (evenly from the root to the flat tip); change points: how many
              Bezier points along the chord change the thickness and the camber
              line (0 to 8; their curves dt and dc; more of them keep the shape).
    Distributions
              pick chord, sweep, rake, pitch, thickness or camber, or a change dt1
              .. dc8, to edit its curve over the span, 1 or 2 segments as the
              blade's, or as the table has it. The root's sweep and rake are
              pinned at 0: the root's LE is the origin.
Each variable:  free | name | low | slider | high | value | where | dep
    free      ticked: a design variable within low .. high; unticked: held
    name      what a relation calls the variable: its key in its curve (v1, d2,
              u4, ...) or its own name (span, H, radius); a variable of another
              curve goes by its full name (chord.v1), as the set-up files have it
    d         the position of a control point, as the fraction d of the span
              left (rudder: below the point above it; blade: up to the tip;
              blade, 2 segments: d1 from P4 to the root, d2 from P4 to the tip)
    dx        rudder LE control point: x offset from the original LE, mm (negative forward)
    value     rudder: a section coordinate, fraction of the sharp chord;
              blade: the distribution's value (P/D, c/D, t/c, f/c, deg, rake/D);
              hull: a lengths or a radius (m), an exponent or a CST coefficient;
              wing, fin: mm, deg, t/c, f/c, a change in fractions of the chord
    where     rudder: the height; blade: r/R; hull: the x it sits at (m); wing,
              fin: y along the span (mm), the planform area for the span
    low, high the bounds: each a number, or (25 Nov 2026) an expression of other
              variables, such as v1 or 0.9*chord.v0 (numbers, names, + - * /,
              brackets): a dependent bound, as P4's pitch between P1's and the
              cap in the MSc propeller optimisation. The variable then lies
              between its bounds wherever the others are: when one of them moves
              it keeps its place between its bounds (the fraction of the way
              from low to high), and that fraction, 0 to 1, is what an optimiser
              varies (the set-up's design vector), so every sample lies inside
              its bounds. An expression longer than its box is shown in full on
              a line under the row. Bounds may cross (low above high) where a
              number faces an expression; the tool then warns.
    dep       unticked: an independent variable, a number within low .. high.
              Ticked: a dependent one (24 Nov 2026, xgeom_relations.py), through
              a dependent bound (above) or a relation: the line that opens under
              the row takes its relation to other variables, an operator
              (< <= > >= =) and an expression of them (0.5v4 is 0.5*v4), such
              as > 0.5*v4 or <= 0.5*v2 + 2*chord.v3; several separated by commas
              (> 0.5*v0, < v1). Return sets it. A relation is a condition on top
              of the bounds: the variable stays where it is until the relation
              would break and then follows (an optimiser checks it, or projects
              onto it). With = the variable takes the expression's value and is
              no design variable. The line shows the limits the variable has
              now (its bounds, narrowed by its relations), and the slider runs
              over them. A variable may depend on dependent ones, but not in a
              loop. Relations that cannot hold inside low .. high leave the
              variable on a bound, with a warning (the build still runs). dep
              unticked: independent again (the relation
              dropped, the dependent bounds numbers where they are). Saved with
              the set-up: on the var line of a component file (the low and high
              columns, the relation after the flag), in the rudder's set-up as
              "lo_expr", "hi_expr" and "relation"; an expression whose variable
              is gone (a curve's new form) is dropped, which the status bar
              says.
Right panel
    Design       rudder: planform, the picked curve against the original, and in
                 3D five sections from the root to H with their control polygons
                 and each control point's track along the span. Blade: the
                 expanded outline, the picked curve against the parameter file,
                 and in 3D the blade as the CAD builds it. Hull: the half-profiles
                 of the whole hull, the picked curve's part against the parameter
                 file, and the hull in 3D. Wing, fin, blended wing body: the
                 picked curve against the parameter file and the sections over
                 the chord (dotted: without the changes), nothing else. The
                 blade's and the hull's 2D plots have r/R and x across; the
                 rudder's, the wing's and the fin's are drawn as the rudder stands
                 (x, or the curve's value, across and the height or span up). Drag
                 the 3D view to turn it.
    3D sections  the whole rudder as its sections: modified (blue), the cut at H
                 (red), unchanged (grey); all blades of the propeller; the hull's
                 sections by part; the wing's or fin's XCAD sections.
Save set-up ... writes the component as it stands as a component file (every
variable with its value, bounds, free flag and, a dependent one, its relation;
each curve's form and points, the settings, the table; xgeom_files.py): a dialog
opens on the file it came from.
Load ... and --file read it back exactly; the
optimiser reads it (blade_modify.load_component, wing_modify.load_component,
hull_modify.load_component, rudder_modify.load_component, xgeom_files.load). The
rudder modified below H is a component as the others (26 Nov 2026; kind
rudder_h: the original's files, the curves with their join at H or their laws,
the variables); XGeom no longer reads the JSON set-up it had. Load ... (or
--file) of a planar rudder's component file (kind rudder:
bb2/outputs/bb2_sternplane.txt, outputs/wind_tunnel_rudder.txt) makes that
rudder the original of the model modified below H (19 Nov 2026, the user: the
panel must stay the same, H and tangent at H with it): xgeom_from_cad.
rudder_from_planar writes its sections, tip and cap as an extraction would and
fits them, beside the file (<name>_modified_below_H/, about half a minute the
first time, read back while the file is unchanged), and the design starts as
that rudder unchanged.
The build button builds the component of its panel (in the vehicle its text
names it): XCAD (rudder) writes the case to the output folder (XCAD file, section
table, design record and the set-up as a component file, <case>.txt), lofts the
XCAD file into a solid and writes it as STEP. CAD (blade) writes the case (its component file <case>.txt,
XCAD points, design JSON) and the CAD the Output box's "CAD writes" names: one
blade on its 1/Z of the hub (with a cylinder hub the DRDC five-surface blade
and the hub sector of the MSc pipeline, untrimmed surfaces; with a cone the
periodic sector as a solid), half the propeller (an even Z) or the whole, as
solids, IGES and STEP (blade_modify.write_cad). CAD (hull) writes the case (component file, design JSON,
stations, hydrostatics, section points, check plot) and the STEP. CAD (wing,
fin) writes the case (component file, XCAD file, sections table, design JSON,
check plot) to outputs/wing or outputs/fin and the loft of the
XCAD file as STEP. OCC viewer opens the last STEP (needs pythonocc-core). Save
CAD ... (and the Vehicle tab's Save vehicle CAD ...) writes the CAD of the
design as it stands to a file picked in a dialog, STEP (.step, .stp) or IGES
(.igs, .iges): the last build's when nothing has changed since, else the build
runs first (as its button does, the case written to the output folder too). The
IGES of the blade is its own DRDC IGES; the others are converted from their STEP
(faces as trimmed surfaces, mm). Every build's outcome (and the error of a
failed one) is added to outputs/xgeom_tool_log.txt.
Half body: the rudder, the fin, the wing and the hull have a 'half body'
switch under Output: their build and Save CAD then write the CAD cut at the
plane of symmetry (the rudder, fin and wing at their section plane, the hull
at its vertical centre plane), for CFD on half the domain; the switch is
saved with the set-up. The sections must be symmetric (a cambered wing or
fin is refused with a message), and the Vehicle CAD always uses full bodies.
The bar at the bottom: green ok, amber a build running (the build button counts
the seconds), orange a warning (e.g. blades closer than the clearance; the
build still runs), red a problem or an error; a design with a problem (e.g. a
hub too short for the blade root) is not built until it is solved.

Other geometries plug in as adapters with the methods of
xgeom_rudder.RudderAdapter (registered in ADAPTERS) and a panel function here
(PANELS) for what is particular to them.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import subprocess
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import tkinter as tk                                                 # noqa: E402
from tkinter import filedialog, ttk                                  # noqa: E402

import matplotlib                                                    # noqa: E402
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure                                 # noqa: E402

from xgeom_common import Row                                         # noqa: E402

ADAPTERS = {"rudder": ("xgeom_rudder", "RudderAdapter"), "blade": ("xgeom_blade", "BladeAdapter"),
            "hull": ("xgeom_hull", "HullAdapter"), "wing": ("xgeom_wing", "WingAdapter"),
            "fin": ("xgeom_wing", "FinAdapter")}
PLANAR_RUDDER = ("xgeom_wing", "PlanarRudderAdapter")   # a planar rudder's model (13 Nov 2026; BB2's stern plane):
                                                         # the tool loads its files as the original of the model
                                                         # modified below H since 19 Nov (rudder_from_planar)
BODIES = {"hull": ("xgeom_hull", "HullAdapter"), "bwb": ("xgeom_wing", "BwbAdapter")}   # the Hull slot's two
BODY_NAMES = {"hull": "cylindrical hull", "bwb": "blended wing body"}                   # bodies (20 Nov 2026)
RUDDERS = {"modify": ADAPTERS["rudder"], "new": PLANAR_RUDDER}   # the Rudder slot's two options (26 Nov 2026):
RUDDER_NAMES = {"modify": "modify an existing rudder",           # the model modified below H (an original, the
                "new": "design a new rudder"}                    # lower bound of the shape), the planar model (none)
NEW_RUDDER = os.path.join(HERE, "rudder.txt")                    # what a new rudder starts from
VEHICLE = ("hull", "blade", "rudder", "fin", "wing")   # the vehicle's components, in the switcher's order
NAMES = {"hull": "Hull", "blade": "Propeller", "rudder": "Rudder", "fin": "Fin", "wing": "Wing"}
COUNTS = ("rudders", "fins", "wings")              # the Vehicle tab's whole numbers
POSITIONS = ("prop_x", "rudder_x", "fin_x", "wing_x")   # the Vehicle tab's positions along the hull: x/L (0 the
                                                         # nose, 1 the aft end), 5 decimals
INK2, RED = "#52514e", "#e34948"
STATUS = {"ok": ("#dfeedd", "black"), "busy": ("#f6d58e", "black"), "warn": ("#f0a35e", "black"),
          "error": ("#c62f2e", "white")}                                                     # bar colours
VEHICLE_TAB = 2                                    # the notebook's tabs: Design, 3D sections, Vehicle
VEHICLE_DIR = os.path.join("outputs", "vehicle")   # the Vehicle CAD's files (relative to this folder)
LOG = os.path.join(HERE, "outputs", "xgeom_tool_log.txt")   # every build's outcome
CAD_TYPES = [("STEP", "*.step *.stp"), ("IGES", "*.igs *.iges")]   # Save CAD's file types
FROM_CAD_KINDS = ("blade", "hull", "rudder")   # From CAD ...: the components with an extraction (xgeom_from_cad)
COLS = 9                                       # a variable row's columns: free | name | (label) | low | slider |
                                               # high | value | where | dep


def planar_rudder_file(path):
    """Is path a planar rudder's component file (kind rudder: the Rudder slot makes it the original of the
    model modified below H, xgeom_from_cad.rudder_from_planar), rather than a set-up of that model (kind
    rudder_h)?"""
    if not path or not os.path.isfile(path):
        return False
    if importlib.util.find_spec(PLANAR_RUDDER[0]) is None:          # the Rudder folder: no planar model
        return False
    import xgeom_files as XF
    return XF.component_kind(path) == "rudder"


def modified_rudder_file(path):
    """Is path a set-up of the rudder modified below H (a component file of kind rudder_h)?"""
    import xgeom_files as XF
    return bool(path) and os.path.isfile(path) and XF.component_kind(path) == XF.MODIFIED


def is_planar(ad):
    """An adapter of a planar component (xgeom_wing: the wing, the fin, the planar rudder)?"""
    return getattr(ad, "role", None) is not None


def load_adapter(name, params=None, progress=None):
    """The adapter of a component, from its component file when given; the rudder's is the model modified
    below H (its set-up a file of kind rudder_h), a planar rudder's component file (planar_rudder_file) its
    original (xgeom_from_cad.rudder_from_planar; progress(text) is told what runs)."""
    if name == "rudder" and planar_rudder_file(params):
        import xgeom_from_cad as FC
        return FC.rudder_from_planar(params, progress=progress or (lambda text: None))["adapter"]
    if name == "hull" and body_of_file(params) == "bwb":           # the Hull slot's other body
        name = "bwb"
    module, cls = BODIES[name] if name == "bwb" else ADAPTERS[name]
    return getattr(importlib.import_module(module), cls)(**({"params": params} if params else {}))


def body_of_file(path):
    """The Hull slot's body a component file is for: 'bwb' (kind bwb, a blended wing body), else 'hull'."""
    if not path or not os.path.isfile(path):
        return "hull"
    import xgeom_files as XF
    try:
        return "bwb" if XF.component_kind(path) == "bwb" else "hull"
    except (OSError, ValueError):
        return "hull"


def body_of(ad):
    """Which body an adapter in the Hull slot is: 'bwb' (the blended wing body) or 'hull' (the cylindrical one)."""
    return "bwb" if getattr(ad, "role", None) == "bwb" else "hull"


def rudder_of(ad):
    """Which option an adapter in the Rudder slot is: 'new' (the planar model: a rudder of its own) or 'modify'
    (the model modified below H: an existing rudder)."""
    return "new" if is_planar(ad) else "modify"


def available(name):
    """Is the geometry's adapter in this folder (the Rudder folder has the rudder's only)?"""
    return importlib.util.find_spec(ADAPTERS[name][0]) is not None


def same(a, b):
    """Two build keys (App._design_key) alike: tuples item by item, numbers and strings by value, anything
    else (a design, a parameter file) the same object."""
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, (str, int, float, bool, type(None))) or isinstance(b, (str, int, float, bool, type(None))):
        return a == b
    return a is b


def save_built(kind, step, path):
    """A build's STEP written to path as Save CAD asks (xcad_loft.save_cad): the blade's IGES is its own
    DRDC IGES, written next to its STEP. Returns how."""
    import xcad_loft as XL
    return XL.save_cad(step, path, os.path.splitext(step)[0] + ".iges" if kind == "blade" else None)


def then_save(res, kind, path):
    """A build's result with its CAD then written to path (Save CAD), when the build made one."""
    if not path:
        return res
    if not res.get("view"):
        res["message"] = f"NOT saved to {path} (no CAD); " + res.get("message", "")
        return res
    try:
        how = save_built(kind, res["view"], path)
    except Exception as exc:
        res.update(message=f"NOT saved to {path}: {type(exc).__name__}: {exc}; " + res["message"], error=True)
        return res
    res["message"] = f"saved {path} ({how}); " + res["message"]
    res.setdefault("report", []).append(f"  saved: {path} ({how})")
    return res


def file_kind(path):
    """The component a component file (its kind: a blended wing body's is the Hull slot's, a rudder modified
    below H's the Rudder slot's) or an old parameter file (its first line; its component refuses it) is
    for."""
    import xgeom_files as XF
    kind = XF.file_kind(path)
    return {"bwb": "hull", XF.MODIFIED: "rudder"}.get(kind, kind)


def grid_row(r):
    """The grid row of line r of a box of variables (0 the header): each line takes three, a variable's own
    and, under it, for a dependent one its relation and (a bound's expression too long for its box) its bounds
    in full; both hidden for an independent one."""
    return 3 * r


BOX = 8                                        # the characters a low or high box shows


DEPENDENT_HINT = ("low, high: a number, or an expression of other variables (v1, 0.9*chord.v0): a dependent bound.  "
                  "dep: a relation to other variables (> 0.5*v4).")
LAW_BOX = 62                                   # the characters a law's box shows
LAW_HINT = ("low law, high law: a number, or an expression of the curves at the same height (P3z, orig.P2z, LE, TE, "
            "chord, orig.chord; + - * / ( ) min( ) max( )). The curve then lies between them, and its variables "
            "are the fraction of the way: 0 on the low law, 1 on the high one. Both empty: no laws.")


class VarRow:
    """free | name | (label) | low | slider | high | value | where | dep, for one variable. low and high each
    take a number or an expression of other variables ('v1', '0.9*chord.v0': a dependent bound; the variable
    then keeps its place between its bounds when the others move). dep is ticked for a variable that depends
    on others, through a bound or a relation; the line under it takes its relation to them ('> 0.5*v4', '<=
    0.5*v2 + 2*chord.v3', '= v0'; xgeom_relations.py) and shows the limits it has now, over which the slider
    runs; a line under that has the bounds in full when a box cuts one. Unticked: independent, numbers for low
    and high and no relation."""

    def __init__(self, app, parent, r, spec):
        self.app, self.name, self.digits = app, spec.name, spec.digits
        ad = app.ad
        self._busy = False
        self._open = False                               # dep ticked, no relation entered yet
        g = grid_row(r)
        self.free = tk.BooleanVar(value=ad.is_free(self.name))
        self.free_box = ttk.Checkbutton(parent, variable=self.free, command=self.on_free)
        self.free_box.grid(row=g, column=0)
        ttk.Label(parent, text=self.name.split(".")[-1]).grid(row=g, column=1, sticky="w", padx=(0, 6))
        self.label = ttk.Label(parent, text=spec.label, width=12)
        self.label.grid(row=g, column=2, sticky="w")
        self.lo = ttk.Entry(parent, width=BOX, justify="right")
        self.hi = ttk.Entry(parent, width=BOX, justify="right")
        lo, hi = ad.bounds(self.name)
        self.scale = ttk.Scale(parent, from_=lo, to=hi, orient="horizontal", length=120, command=self.on_scale)
        self.val = ttk.Entry(parent, width=9, justify="right")
        self.info = ttk.Label(parent, text="", foreground=INK2, width=9)
        for c, w in enumerate((self.lo, self.scale, self.hi, self.val, self.info), start=3):
            w.grid(row=g, column=c, sticky="ew", padx=1, pady=1)
        self.dep = tk.BooleanVar(value=ad.is_dependent(self.name))
        ttk.Checkbutton(parent, variable=self.dep, command=self.on_dep).grid(row=g, column=8)
        self.rel = ttk.Entry(parent)                     # the relation line, shown for a dependent variable
        self.rel_info = ttk.Label(parent, text="", foreground=INK2)
        self._rel_line = ((ttk.Label(parent, text="relation", foreground=INK2), dict(column=2, sticky="e", padx=4)),
                          (self.rel, dict(column=3, columnspan=3, sticky="ew", padx=1, pady=(0, 3))),
                          (self.rel_info, dict(column=6, columnspan=3, sticky="w", padx=3)))
        for w, opts in self._rel_line:
            opts["row"] = g + 1
        self.full = ttk.Label(parent, text="", foreground=INK2)      # the bounds in full, when a box cuts one
        self._full_at = dict(row=g + 2, column=2, columnspan=COLS - 2, sticky="w", padx=4, pady=(0, 3))
        for e, fn in ((self.lo, self.on_bounds), (self.hi, self.on_bounds), (self.val, self.on_value),
                      (self.rel, self.on_relation)):
            e.bind("<Return>", fn)
            e.bind("<FocusOut>", fn)
        self.refresh()

    def refresh(self, slider=True):
        ad = self.app.ad
        v = ad.get(self.name)
        lo, hi = ad.bounds(self.name)
        low, high = ad.limits(self.name)                 # a dependent variable's: its bounds and its relations
        dep, defined = ad.is_dependent(self.name), ad.is_defined(self.name)
        room = high > low and not defined
        focus = self.app.root.focus_get()
        self._busy = True
        try:
            self.scale.configure(from_=low if room else lo, to=high if room else hi)
            self.scale.state(["!disabled" if room else "disabled"])
            if slider:
                self.scale.set(v)
            self.val.state(["!disabled"])
            for e, x in zip((self.lo, self.hi, self.val), self._bounds_shown() + (f"{v:.{self.digits}f}",)):
                if focus is not e:
                    e.delete(0, "end")
                    e.insert(0, x)
            self.val.state(["disabled" if defined else "!disabled"])        # '=': it follows the others
            self.free_box.state(["disabled" if defined else "!disabled"])
            self.info.configure(text=ad.row_info(self.name))
            self.label.configure(foreground="black" if ad.varies(self.name) else INK2)
            self._relation_line(v, low, high, dep, defined, focus)
        finally:
            self._busy = False

    def _bounds_shown(self):
        """What the low and high boxes show: a number, or a dependent bound's expression."""
        ad = self.app.ad
        return tuple(f"{b:.{self.digits}f}" if t is None else t
                     for b, t in zip(ad.bounds(self.name), ad.bound_text(self.name)))

    def _relation_line(self, v, low, high, dep, defined, focus):
        """The line under the row: there for a dependent variable (and once dep is ticked), with its relation
        and what the variable has now: its limits, the value it follows ('='), or no room."""
        ad, d = self.app.ad, self.digits
        show = dep or self._open
        self.dep.set(show)
        for w, opts in self._rel_line:
            if show:
                w.grid(**opts)
            else:
                w.grid_remove()
        long = [f"{w} {t}" for w, t in zip(("low", "high"), ad.bound_text(self.name)) if t and len(t) > BOX]
        if long:                                         # an expression its box cuts: here in full
            self.full.configure(text="bounds:  " + ",  ".join(long))
            self.full.grid(**self._full_at)
        else:
            self.full.grid_remove()
        if not show:
            return
        if focus is not self.rel:
            self.rel.delete(0, "end")
            self.rel.insert(0, ad.relation(self.name))
        if not dep:                                      # nothing entered yet: an example with a variable like it
            curve, key = self.name.split(".")[0], self.name.split(".")[-1]
            names = [n for n in ad.meta if n != self.name]
            near = [n.split(".")[1] for n in names if "." in n and n.split(".")[0] == curve]
            like = [k for k in near if k.rstrip("0123456789") == key.rstrip("0123456789")]
            self.rel_info.configure(text=f"e.g.  > 0.5*{(like or near or names or ['v0'])[0]}", foreground=INK2)
        elif defined:
            self.rel_info.configure(text=f"now {v:.{d}f}", foreground=INK2)
        elif high < low:
            self.rel_info.configure(text="no room now: low above high", foreground=RED)
        else:
            self.rel_info.configure(text=f"now {low:.{d}f} .. {high:.{d}f}", foreground=INK2)

    def on_scale(self, s):
        if not self._busy:
            self.app.set_value(self.name, float(s), self)

    def on_value(self, _event=None):
        try:
            v = float(self.val.get())
        except ValueError:
            return self.refresh()
        if abs(v - self.app.ad.get(self.name)) > 0.5 * 10.0 ** (-self.digits):
            self.app.set_value(self.name, v, None)

    def on_bounds(self, _event=None):
        """The low or high box changed (Return, or the box left): a number, or an expression of other variables
        ('v1', '0.9*chord.v0': a dependent bound). A text that is neither is named in the status bar and
        changes nothing. A box left as it is shown keeps its bound exactly (not the rounded number)."""
        ad = self.app.ad
        try:
            typed = (self.lo.get().strip(), self.hi.get().strip())
        except tk.TclError:                              # the row is gone (panel rebuilt)
            return
        given, changed = [], False
        for t, s, b, e in zip(typed, self._bounds_shown(), ad.bounds(self.name), ad.bound_text(self.name)):
            same = t == s
            if not same and e is None:                   # the same number, written another way ('0.5', '.50')
                try:
                    same = float(t) == round(b, self.digits)
                except ValueError:
                    pass
            given.append((b if e is None else e) if same else t)
            changed = changed or not same
        if not changed:
            return
        try:
            ad.set_bounds(self.name, *given)
        except ValueError as exc:                        # no bound, or one refused: nothing has changed
            self.app.say(f"{self.name}: {exc}", error=True)
            return
        except Exception as exc:                         # keep the tool alive
            self.app.say(f"{type(exc).__name__}: {exc}", error=True)
        for e, x in zip((self.lo, self.hi), self._bounds_shown()):       # as the tool writes it ('0.9*v0')
            e.delete(0, "end")
            e.insert(0, x)
        self.app.after_change()

    def on_free(self):
        self.app.ad.set_free(self.name, self.free.get())
        self.app.refresh_labels()
        self.app.schedule_update()

    def on_dep(self):
        """The dep switch. Ticked: the relation line opens, and the variable depends on others once a relation
        is entered there (or an expression in its low or high box). Unticked: independent again, its relation
        dropped and its dependent bounds numbers, where they are now."""
        self._open = bool(self.dep.get())
        if not self._open and self.app.ad.is_dependent(self.name):
            try:
                self.app.ad.set_independent(self.name)
            except Exception as exc:                     # refused: bounds that cross cannot be numbers
                self.app.say(str(exc) if isinstance(exc, ValueError) else f"{type(exc).__name__}: {exc}", error=True)
                self.refresh(slider=False)               # dep stays ticked
                return
            self.app.after_change()
            return
        self.refresh(slider=False)
        if self._open:
            self.rel.focus_set()

    def on_relation(self, _event=None):
        """The relation entered (Return, or the entry left): the variable depends on the others as it says, and
        the design follows at once; an empty one makes it independent. A text that is no relation is named in
        the status bar and changes nothing."""
        ad = self.app.ad
        try:
            text = self.rel.get().strip()
        except tk.TclError:                              # the row is gone (panel rebuilt)
            return
        if text == ad.relation(self.name):
            return
        try:
            shown = ad.set_relation(self.name, text)
        except ValueError as exc:
            self.app.say(f"{self.name}: {exc}", error=True)
            return
        self.rel.delete(0, "end")                        # as the tool writes it ('> 0.5*v4' for '>0.5v4')
        self.rel.insert(0, shown)
        self.app.after_change()


class App:
    """The tool's window for one geometry, or for the vehicle (geometry 'vehicle'): its components (VEHICLE,
    those whose adapter is here) one at a time in the left panel and the Design and 3D sections tabs, and
    all of them in the Vehicle tab. params: a component file's path, or for the vehicle a list of them (each
    goes to the component it is for)."""

    def __init__(self, root, adapter_name="rudder", params=None):
        self.root = root
        self.vehicle = adapter_name == "vehicle"
        self.kinds = [k for k in VEHICLE if available(k)] if self.vehicle else [adapter_name]
        if self.vehicle and "hull" not in self.kinds:
            raise RuntimeError("the vehicle needs xgeom_hull.py (the X_geom folder)")
        self.start = {k: {"params": None} for k in self.kinds}
        for path in ([params] if isinstance(params, str) else list(params or [])):
            kind = file_kind(path)
            if kind is None:                                # no component file (a JSON set-up of old, say)
                import xgeom_files as XF
                raise XF.not_component(path)
            kind = kind if self.vehicle else adapter_name
            if kind not in self.start:
                raise ValueError(f"{path}: not a component file of " + ", ".join(self.kinds))
            self.start[kind]["params"] = path
        self.adapters, self.errors, self.frames, self.picked, self._vcache = {}, {}, {}, {}, {}
        self.bodies = {}                                    # the Hull slot's body not shown: body -> its adapter
        self.rudders = {}                                   # the Rudder slot's option not shown: option -> its adapter
        self.kind, self.ad, self.placement, self._vinfo = None, None, None, None
        self.curve = tk.StringVar()
        self.rows = []
        self._pending = False
        self._job = None
        self._note = None                                   # a message the next view update shows instead of ok
        self.step = None
        self.built = {}                                     # kind or 'vehicle' -> the last build's key and STEP
        self.save_dir = None                                # the folder Save CAD wrote to last
        self._build_layout()
        self.switch(self.kinds[0])

    # ------------------------------------------------------------ components
    def _adapter(self, kind):
        """The component's adapter, made when first needed from its component file; for the vehicle None if
        it cannot be made (the error is kept in self.errors)."""
        if kind in self.adapters:
            return self.adapters[kind]
        if kind in self.errors:
            return None
        st = self.start[kind]
        try:
            if kind == "rudder" and planar_rudder_file(st["params"]):
                self.say(f"the rudder of {os.path.basename(st['params'])} as the original modified below H (its "
                         f"sections fitted the first time, about half a minute) ...", busy=True)
                self.root.update_idletasks()
            ad = load_adapter(kind, st["params"])
        except Exception as exc:
            if not self.vehicle:
                raise
            traceback.print_exc()
            self.errors[kind] = f"{type(exc).__name__}: {exc}"
            return None
        self.adapters[kind] = ad
        return ad

    def replace_adapter(self, kind, ad):
        """Another model in a component's slot (the Rudder: the planar model or the modified-below-H one): the
        new adapter, its views made afresh, the Vehicle tab's part and the last build forgotten; shown at once
        when the slot is."""
        old = self.frames.pop(kind, None)
        if old:
            old["design"].destroy()
            old["sk"].destroy()
        self.adapters[kind] = ad
        self.errors.pop(kind, None)
        self._vcache.pop(kind, None)
        self.built.pop(kind, None)
        if self.kind == kind:
            self.kind = None                                # switch() shows the slot afresh
            self.switch(kind)

    def on_body(self):
        """The Hull slot's Body box: the cylindrical hull or the blended wing body (BODIES); each keeps its state
        while the other is shown; the blended wing body starts from bwb_glider.txt the first time."""
        want = self.vbody.get()
        ad = self.adapters.get("hull")
        have = body_of(ad)
        if want == have:
            return
        try:
            new = self.bodies.get(want)
            if new is None:
                self.say(f"loading the {BODY_NAMES[want]} ...", busy=True)
                self.root.update_idletasks()
                module, cls = BODIES[want]
                new = getattr(importlib.import_module(module), cls)()
        except Exception as exc:
            traceback.print_exc()
            self.vbody.set(have)
            self.say(f"{BODY_NAMES[want]}: not loaded: {type(exc).__name__}: {exc}", error=True)
            return
        self.bodies.pop(want, None)
        self.bodies[have] = ad
        self.replace_adapter("hull", new)
        self._note = (f"the Hull slot holds the {BODY_NAMES[want]}" +
                      ("; the Vehicle tab places the parts on the cylindrical hull (the blended wing body's vehicle "
                       "comes later)" if want == "bwb" and self.vehicle else ""))
        self.say(self._note)

    def on_rudder(self):
        """The Rudder slot's box: modify an existing rudder (the model modified below H: the original kept
        above H and, below it, the lower bound of the shape) or design a new one (the planar model: no
        original; it starts from rudder.txt the first time). Each keeps its state while the other is shown."""
        want = self.vrudder.get()
        ad = self.adapters.get("rudder")
        have = rudder_of(ad)
        if want == have:
            return
        try:
            new = self.rudders.get(want)
            if new is None:
                self.say(f"loading the rudder to {RUDDER_NAMES[want].split(' ', 1)[0]} ...", busy=True)
                self.root.update_idletasks()
                module, cls = RUDDERS[want]
                new = getattr(importlib.import_module(module), cls)(**({"params": NEW_RUDDER} if want == "new" else {}))
        except Exception as exc:
            traceback.print_exc()
            self.vrudder.set(have)
            self.say(f"{RUDDER_NAMES[want]}: not loaded: {type(exc).__name__}: {exc}", error=True)
            return
        self.rudders.pop(want, None)
        self.rudders[have] = ad
        self.replace_adapter("rudder", new)
        self._note = ("the Rudder slot: a new rudder, no original (Save set-up writes it as a component file, "
                      "which the other option loads as the original to modify)" if want == "new" else
                      "the Rudder slot: an existing rudder, modified below H")
        self.say(self._note)

    def on_laws(self, _event=None):
        """The picked curve's low and high law as typed (Return, or a box left): both given, the curve lies
        between them from now on and its variables are the fractions of the way; both empty, its laws are taken
        off. One alone waits for the other. A text that is no law is named in the status bar and changes
        nothing."""
        ad, c = self.ad, self.curve.get()
        try:
            low, high = (self.law[k].get().strip() for k in ("low", "high"))
        except tk.TclError:                              # the boxes are gone (panel rebuilt)
            return
        had = ad.laws_of(c)
        if (low, high) == (had or ("", "")) or bool(low) != bool(high):
            return
        try:
            ad.set_laws(c, low or None, high or None)
        except ValueError as exc:                        # no law: nothing has changed
            self.say(f"{c}: {exc}", error=True)
            return
        except Exception as exc:                         # keep the tool alive
            self.say(f"{type(exc).__name__}: {exc}", error=True)
            return
        now = ad.laws_of(c)
        if now == had:                                   # the same laws, written another way: as the tool writes them
            for k, text in zip(("low", "high"), now or ("", "")):
                self.law[k].delete(0, "end")
                self.law[k].insert(0, text)
            return
        self.after_change(structure=True)
        self._note = (f"{c}: its laws taken off, its control values again, where it was" if now is None else
                      f"{c}: set by its laws at every height, no variables" if ad.law_defined(c) else
                      f"{c}: between its laws; its variables are the fractions of the way (0 on the low law, 1 on "
                      f"the high one), free, and it joins the original at H with its value only")

    def on_keep_inside(self, _event=None):
        """The rudder's 'keep the original inside' switch and its tolerance (mm): the preview measures how far
        the modified rudder goes inside the original and warns beyond the tolerance; saved with the set-up."""
        try:
            on, text = bool(self.keep_inside.get()), self.inside_tol.get().strip()
        except tk.TclError:                              # the box is gone (panel rebuilt)
            return
        s = self.ad.settings
        try:
            tol = float(text)
        except ValueError:
            tol = None
        if tol is not None and (on, tol) == (bool(s.get("keep_inside")), s.get("inside_tol_mm")):
            return
        try:
            if tol is None:
                raise ValueError(f"the tolerance is a number of mm, not '{text}'")
            self.ad.set_keep_inside(on, tol)
        except ValueError as exc:
            self.say(str(exc), error=True)
        self.inside_tol.delete(0, "end")
        self.inside_tol.insert(0, f"{s['inside_tol_mm']:g}")
        self.keep_inside.set(bool(s.get("keep_inside")))
        self.schedule_update()

    def _noun(self, kind):
        """What the slot holds, for the labels: the component's name (the blended wing body: BWB)."""
        ad = self.adapters.get(kind)
        return getattr(ad, "short", None) or NAMES.get(kind, kind).lower()

    def _frames(self, kind):
        """A component's Design and 3D sections figures, made when it is first shown."""
        if kind not in self.frames:
            ad = self.adapters[kind]
            f = {"design": ttk.Frame(self.tab_design), "sk": ttk.Frame(self.tab_3d)}
            f["fig"] = Figure(figsize=(10, 7.5), dpi=100)
            f["canvas"] = FigureCanvasTkAgg(f["fig"], master=f["design"])
            NavigationToolbar2Tk(f["canvas"], f["design"]).update()
            f["canvas"].get_tk_widget().pack(fill="both", expand=True)
            f["views"] = ad.views(f["fig"])
            f["fig3"] = Figure(figsize=(10, 7.5), dpi=100)
            f["canvas3"] = FigureCanvasTkAgg(f["fig3"], master=f["sk"])
            f["canvas3"].get_tk_widget().pack(fill="both", expand=True)
            f["view3"] = ad.skeleton_view(f["fig3"])
            self.frames[kind] = f
        return self.frames[kind]

    def switch(self, kind):
        """Show a component: its left panel and its Design and 3D sections tabs."""
        if kind == self.kind:
            return
        if kind not in self.adapters:
            self.say(f"loading the {NAMES.get(kind, kind).lower()} ...", busy=True)
            self.root.update_idletasks()
        ad = self._adapter(kind)
        if ad is None:
            self.say(f"{NAMES[kind]}: not loaded: {self.errors[kind]}", error=True)
            if self.kind is not None:
                self.comp.set(self.kind)
            return
        if self.kind is not None:
            self.picked[self.kind] = self.curve.get()
            self.frames[self.kind]["design"].pack_forget()
            self.frames[self.kind]["sk"].pack_forget()
        self.kind, self.ad = kind, ad
        f = self._frames(kind)
        f["design"].pack(fill="both", expand=True)
        f["sk"].pack(fill="both", expand=True)
        self.fig, self.canvas, self.views = f["fig"], f["canvas"], f["views"]
        self.fig3, self.canvas3, self.view3 = f["fig3"], f["canvas3"], f["view3"]
        self.curve.set(self.picked.get(kind, ad.names[0]))
        if self.vehicle:
            self.comp.set(kind)
        self._title()
        self.rebuild_panel()
        self.update_views()

    def _title(self):
        self.root.title(f"XGeom design tool: {'vehicle, ' if self.vehicle else ''}{self.ad.title}")

    # ------------------------------------------------------------ layout
    def _build_layout(self):
        self.status = tk.Label(self.root, text="", anchor="w", justify="left", font=("TkDefaultFont", 12, "bold"),
                               padx=10, pady=6)
        self.status.pack(fill="x", side="bottom")                  # packed first: always in view
        self.status.bind("<Configure>", lambda e: self.status.configure(wraplength=max(e.width - 30, 200)))
        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True)
        left, right = ttk.Frame(paned), ttk.Frame(paned)
        paned.add(left, weight=0)
        paned.add(right, weight=1)
        if self.vehicle:                                           # the component switcher
            bar = ttk.Frame(left)
            bar.pack(fill="x", side="top", padx=6, pady=(6, 2))
            ttk.Label(bar, text="Component", font=("TkDefaultFont", 11, "bold")).pack(side="left", padx=(0, 8))
            self.comp = tk.StringVar()
            for k in self.kinds:
                ttk.Radiobutton(bar, text=NAMES[k], value=k, variable=self.comp,
                                command=lambda: self.switch(self.comp.get())).pack(side="left", padx=4)
            ttk.Separator(left, orient="horizontal").pack(fill="x", side="top", pady=(2, 0))
        self.outbar = ttk.Frame(left)                              # the Output box, under the panel: always in view
        self.outbar.pack(fill="x", side="bottom")
        canvas = tk.Canvas(left, width=640, highlightthickness=0)
        sb = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        self.panel = ttk.Frame(canvas)
        self.panel.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.panel, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self.book = ttk.Notebook(right)
        self.book.pack(fill="both", expand=True)
        self.tab_design, self.tab_3d = ttk.Frame(self.book), ttk.Frame(self.book)
        self.book.add(self.tab_design, text="Design")
        self.book.add(self.tab_3d, text="3D sections")
        if self.vehicle:
            self.tab_vehicle = ttk.Frame(self.book)
            self.book.add(self.tab_vehicle, text="Vehicle")
            self._vehicle_layout()
        self.book.bind("<<NotebookTabChanged>>", lambda e: self.schedule_update())

    def _vehicle_layout(self):
        """The Vehicle tab: the placement settings above the 3D view (xgeom_vehicle)."""
        import xgeom_vehicle as XV
        tab = self.tab_vehicle
        bar = ttk.Frame(tab)
        bar.pack(fill="x", side="top", padx=6, pady=(6, 2))
        num = dict(from_=-100.0, to=100.0, increment=0.005, width=8)
        frac = dict(from_=0.0, to=1.0, increment=0.001, width=8)      # positions along the hull: x/L
        angle = dict(from_=-360.0, to=720.0, increment=15.0, width=6)
        scale = dict(from_=0.01, to=100.0, increment=0.05, width=6)
        count = dict(from_=0, to=XV.MAX_RUDDERS, increment=1, width=4)
        rows = [("Propeller", (("prop_x", "hub starts at x/L", dict(frac, to=2.0)),
                               ("prop_d", "D (m)", dict(num, from_=0.001))), "show"),
                ("Rudder", (("rudder_x", "root TE at x/L", frac), ("rudders", "rudders", count),
                            ("rudder_angle", "first at (deg)", angle), ("rudder_scale", "scale", scale)), "rake")]
        if "fin" in self.kinds:
            rows.append(("Fin", (("fin_x", "root LE at x/L", frac), ("fins", "fins", count),
                                 ("fin_angle", "first at (deg)", angle), ("fin_scale", "scale", scale)), None))
        if "wing" in self.kinds:
            rows.append(("Wing", (("wing_x", "root LE at x/L", frac),
                                  ("wings", "wings", dict(count, to=2, increment=2)),
                                  ("wing_angle", "at (deg)", dict(angle, from_=1.0, to=179.0)),
                                  ("wing_scale", "scale", scale)), None))
        self.vset = {}
        for r, (title, items, extra) in enumerate(rows):
            ttk.Label(bar, text=title, font=("TkDefaultFont", 10, "bold")).grid(row=r, column=0, sticky="w")
            c = 1
            for key, text, kw in items:
                ttk.Label(bar, text=text).grid(row=r, column=c, sticky="e", padx=(12, 3))
                sb = ttk.Spinbox(bar, command=self.on_vehicle, **kw)
                sb.grid(row=r, column=c + 1, sticky="w", pady=1)
                sb.bind("<Return>", self.on_vehicle)
                sb.bind("<FocusOut>", self.on_vehicle)
                self.vset[key] = sb
                c += 2
            if extra == "rake":                         # the rudders' own angle, or the hull's slope
                self.vrake = tk.BooleanVar(value=False)
                ttk.Checkbutton(bar, text="TE rake (deg)", variable=self.vrake,
                                command=self.on_vehicle).grid(row=r, column=c, sticky="e", padx=(12, 3))
                sb = ttk.Spinbox(bar, command=self.on_vehicle, from_=-XV.MAX_RAKE, to=XV.MAX_RAKE, increment=1.0,
                                 width=6)
                sb.grid(row=r, column=c + 1, sticky="w", pady=1)
                sb.bind("<Return>", self.on_vehicle)
                sb.bind("<FocusOut>", self.on_vehicle)
                self.vset["rudder_rake"] = sb
            if extra == "show":
                self.vshow = tk.StringVar(value="vehicle")
                ttk.Label(bar, text="show").grid(row=r, column=c, sticky="e", padx=(24, 3))
                for i, (val, text) in enumerate((("vehicle", "the vehicle"), ("stern", "the stern"))):
                    ttk.Radiobutton(bar, text=text, value=val, variable=self.vshow,
                                    command=self.on_vehicle).grid(row=r, column=c + 1 + i, sticky="w", padx=2)
        row = ttk.Frame(bar)
        row.grid(row=len(rows), column=0, columnspan=12, sticky="w", pady=(4, 0))
        ttk.Label(row, text="CAD", font=("TkDefaultFont", 10, "bold")).pack(side="left")
        self.vbuild_btn = ttk.Button(row, text="Vehicle CAD", command=self.on_vehicle_build)
        self.vbuild_btn.pack(side="left", padx=(12, 6))
        self.vsave_btn = ttk.Button(row, text="Save vehicle CAD ...", command=lambda: self.on_save_cad("vehicle"))
        self.vsave_btn.pack(side="left", padx=(0, 6))
        ttk.Button(row, text="OCC viewer", command=self.open_occ_viewer).pack(side="left")
        ttk.Label(row, text=f"each component's CAD (its own folder, the case named in the left panel), placed as "
                            f"shown and fused: {VEHICLE_DIR}/<case>_vehicle.step (mm)", foreground=INK2,
                  wraplength=520, justify="left").pack(side="left", padx=10)
        self.figv = Figure(figsize=(10, 7), dpi=100)
        self.canvasv = FigureCanvasTkAgg(self.figv, master=tab)
        self.canvasv.get_tk_widget().pack(fill="both", expand=True)
        self.viewv = XV.VehicleView(self.figv)

    def rebuild_panel(self):
        """(Re)create the left panel: the geometry's own boxes (PANELS), the
        picked curve and the output."""
        case = self.case.get() if hasattr(self, "case") and self.case.winfo_exists() else "gui"
        for w in self.panel.winfo_children():
            w.destroy()
        self.rows, self.picks, self.pinned = [], {}, []
        ad = self.ad
        ttk.Label(self.panel, text=ad.title, font=("TkDefaultFont", 11, "bold")).pack(anchor="w", padx=6, pady=(6, 0))
        self.free_label = ttk.Label(self.panel, text="", foreground=INK2, wraplength=590, justify="left")
        self.free_label.pack(anchor="w", padx=6)
        (wing_panel if is_planar(ad) else PANELS[self.kind])(self)

        c = self.curve.get()
        if c not in ad.names:
            c = ad.names[0]
            self.curve.set(c)
        box = ttk.LabelFrame(self.panel, text=f"Curve {getattr(ad, 'curve_label', str)(c)}")
        box.pack(fill="x", padx=6, pady=4)
        ttk.Label(box, text=ad.describe(c), foreground=INK2, wraplength=560, justify="left").pack(fill="x", padx=4)
        top = ttk.Frame(box)
        top.pack(fill="x", pady=2)
        choice = getattr(ad, "segment_choice", False)
        nseg = ad.segments(c) if choice else 1                   # 0: as the table has it
        if choice:
            ttk.Label(top, text="segments").pack(side="left", padx=(4, 2))
            self.nseg = tk.IntVar(value=nseg)
            for n in getattr(ad, "segment_options", (1, 2)):
                text = ad.segment_text(n) if hasattr(ad, "segment_text") else str(n) if n else "table"
                ttk.Radiobutton(top, text=text, value=n, variable=self.nseg,
                                command=self.on_segments).pack(side="left")
        if nseg == 2:
            ttk.Label(top, text="P1 .. P7: P2 = P3 and P5 = P6 at P4's value (zero slope at P4)",
                      foreground=INK2).pack(side="left", padx=8)
        elif nseg == 3:
            ttk.Label(top, text="P0 .. P6: two cubics joined smoothly at P3, flat at the centre line",
                      foreground=INK2).pack(side="left", padx=8)
        elif nseg == 1:
            lo, hi = getattr(ad, "order_range", (1, 12))
            ttk.Label(top, text="control points").pack(side="left",
                                                       padx=(10 if getattr(ad, "segment_choice", False) else 4, 2))
            self.nctrl = ttk.Spinbox(top, from_=lo + 1, to=hi + 1, width=4, command=self.on_order)
            self.nctrl.set(ad.order(c) + 1)
            self.nctrl.bind("<Return>", self.on_order)
            self.nctrl.pack(side="left")
        laws = ad.laws_of(c) if hasattr(ad, "set_laws") else None    # the rudder modified below H: bounding laws
        if laws and ad.law_defined(c):
            self.nctrl.state(["disabled"])                           # the law is the curve: no control points
        if ad.joins:
            self.tangent = tk.BooleanVar(value=ad.design.curves[c].join == "G1")
            tan = ttk.Checkbutton(top, text="C1 on the original's tangent at H (slope matched)",
                                  variable=self.tangent, command=self.on_tangent)
            tan.pack(side="left", padx=10)
            if laws:
                tan.state(["disabled"])                              # between its laws: it joins with its value
        if hasattr(ad, "set_laws"):
            self.law = {}
            for k, side in enumerate(("low", "high")):
                row = ttk.Frame(box)
                row.pack(fill="x", pady=(0, 1))
                ttk.Label(row, text=f"{side} law", width=8).pack(side="left", padx=(4, 2))
                e = self.law[side] = ttk.Entry(row, width=LAW_BOX)
                e.insert(0, laws[k] if laws else "")
                e.bind("<Return>", self.on_laws)
                e.bind("<FocusOut>", self.on_laws)
                e.pack(side="left", fill="x", expand=True, padx=(0, 4))
            ttk.Label(box, text=LAW_HINT, foreground=INK2, wraplength=590, justify="left").pack(fill="x", padx=4)
        grid = ttk.Frame(box)
        grid.pack(fill="x", pady=2)
        self._header(grid)
        for r, spec in enumerate(ad.curve_rows(c), start=1):
            if spec.kind == "var":
                self.rows.append(VarRow(self, grid, r, spec))
            else:
                ttk.Label(grid, text=spec.label, foreground=RED).grid(row=grid_row(r), column=2, sticky="w")
                lab = ttk.Label(grid, text=spec.text, foreground=INK2)
                lab.grid(row=grid_row(r), column=3, columnspan=COLS - 3, sticky="w")
                self.pinned.append(lab)
        ttk.Label(box, text=DEPENDENT_HINT, foreground=INK2, wraplength=590, justify="left").pack(fill="x", padx=4)

        for w in self.outbar.winfo_children():
            w.destroy()
        box = ttk.LabelFrame(self.outbar, text="Output")
        box.pack(fill="x", padx=6, pady=4)
        row = ttk.Frame(box)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="case").pack(side="left", padx=(4, 2))
        self.case = ttk.Entry(row, width=16)
        self.case.insert(0, case)
        self.case.pack(side="left")
        ttk.Button(row, text="Save set-up ..." if hasattr(ad, "save_file") else "Save set-up",
                   command=self.on_save).pack(side="left", padx=(6, 0))
        self.build_btn = ttk.Button(row, text=self.build_label(self.kind),
                                    command=lambda k=self.kind: self.on_build(k))
        self.build_btn.pack(side="left", padx=6)
        self.save_btn = ttk.Button(row, text="Save CAD ...", command=lambda k=self.kind: self.on_save_cad(k))
        self.save_btn.pack(side="left", padx=(0, 6))
        self._buttons()
        ttk.Button(row, text="OCC viewer", command=self.open_occ_viewer).pack(side="left")
        if "half" in getattr(ad, "settings", {}):
            self.vhalf = tk.BooleanVar(value=bool(ad.settings.get("half", False)))
            ttk.Checkbutton(box, text=getattr(ad, "half_text", None) or
                            "half body: the CAD cut at the plane of symmetry (for CFD on half the domain; the "
                            "Vehicle CAD always uses full bodies)",
                            variable=self.vhalf, command=self.on_half).pack(anchor="w", padx=4, pady=(0, 3))
        if getattr(ad, "parts", None):                               # the blade: what the CAD writes
            row = ttk.Frame(box)
            row.pack(fill="x", padx=4, pady=(0, 3))
            ttk.Label(row, text="CAD writes").pack(side="left", padx=(0, 4))
            self.vpart = tk.StringVar(value=ad.settings.get("part", ad.parts[0]))
            for part in ad.parts:
                ttk.Radiobutton(row, text=ad.part_label(part, short=True), value=part, variable=self.vpart,
                                command=self.on_part).pack(side="left", padx=(0, 6))
            ttk.Label(row, text="(1/Z with a cylinder hub: the DRDC surfaces, else solids; the Vehicle CAD takes "
                                "this part, as a solid)", foreground=INK2,
                      wraplength=200, justify="left").pack(side="left")
        self.refresh_labels()

    def _pick(self, parent, r, c, name, text):
        rb = ttk.Radiobutton(parent, text=text, value=name, variable=self.curve, command=self.on_curve)
        rb.grid(row=r, column=c, sticky="w", padx=4)
        self.picks[name] = (rb, text)

    @staticmethod
    def _header(parent):
        for c, t in enumerate(("free", "name", "", "low", "", "high", "value", "where", "dep")):
            ttk.Label(parent, text=t, foreground=INK2).grid(row=0, column=c, sticky="w", padx=(0, 4))

    def refresh_labels(self):
        self.free_label.configure(text=self.ad.summary())
        if getattr(self, "sec_info", None) is not None and self.sec_info.winfo_exists():
            self.sec_info.configure(text=self.ad.sections_info())     # blade: the steps follow the root
        note = getattr(self.ad, "pick_note", None)                  # blade: "(file)", the file's own curve
        for name, (rb, text) in self.picks.items():                 # "held": no free variable on that curve
            rb.configure(text=text + (note(name) if note else "  (held)" if self.ad.n_free(name)[0] == 0 else ""))
        texts = [s.text for s in self.ad.curve_rows(self.curve.get()) if s.kind == "pinned"]
        for lab, t in zip(self.pinned, texts):
            lab.configure(text=t)

    # ------------------------------------------------------------ changes
    def guard(self, fn):
        try:
            return fn()
        except Exception as exc:                                  # keep the tool alive
            self.say(f"{type(exc).__name__}: {exc}", error=True)

    def set_value(self, name, value, source):
        self.guard(lambda: self.ad.set(name, value))
        for row in self.rows:
            row.refresh(slider=row is not source)
        self.refresh_labels()
        self.schedule_update()

    def after_change(self, structure=False, views=True):
        if structure:
            self.rebuild_panel()
        else:
            for row in self.rows:
                row.refresh()
            self.refresh_labels()
        if views:
            self.schedule_update()

    def on_half(self):
        """The half-body switch: the next build or Save CAD writes the half (the component alone, not the
        vehicle)."""
        self.ad.settings["half"] = bool(self.vhalf.get())
        self.say("half body: the next build writes the CAD cut at the plane of symmetry"
                 if self.ad.settings["half"] else "full body again")

    def on_curve(self):
        self.rebuild_panel()
        self.schedule_update()

    def on_order(self, _event=None):
        c = self.curve.get()
        try:
            n = int(self.nctrl.get()) - 1
        except ValueError:
            return
        if n != self.ad.order(c):
            msg = self.guard(lambda: self.ad.set_order(c, n))
            self.after_change(structure=True)
            if isinstance(msg, str):
                self._note = f"{c}: {n + 1} control points, {msg}"

    def on_segments(self):
        """Blade, wing, fin: the picked curve as 1 or 2 segments, fitted again, or as the table has it (0)."""
        c, n = self.curve.get(), self.nseg.get()
        if n == self.ad.segments(c):
            return
        msg = self.guard(lambda: self.ad.set_segments(c, n))
        self.after_change(structure=True)
        if msg is not None:
            what = {2: "2 segments", 3: "2 smooth segments"}.get(n, f"{n} segment")
            self._note = f"{c}: " + (msg if n == 0 else f"{what}, {msg}")

    def on_tangent(self):
        c = self.curve.get()
        self.guard(lambda: self.ad.set_tangent(c, self.tangent.get()))
        self.after_change(structure=True)

    def on_tangent_all(self):
        """The rudder's 'tangent at H': every curve's C1 on the original's tangent, or every one free (each
        curve's own switch sets one); a design with any curve free there gets its CAD as two lofts."""
        on = bool(self.tangent_all.get())
        self.guard(lambda: self.ad.set_tangent_all(on))
        self.after_change(structure=True)
        self._note = ("every curve tangent at H again (C1 on the original's tangent): one loft" if on else
                      "no curve tangent at H: C1 free on each, where it was; the CAD lofts below and above H on its "
                      "own and joins the two there")

    def on_points(self, _event=None):
        try:
            n = int(self.npts.get()) - 1
        except ValueError:
            return
        if n != self.ad.degree:
            err = self.guard(lambda: self.ad.set_degree(n))
            self.after_change(structure=True)
            if err is not None:
                self._note = (f"sections now of degree {self.ad.degree} ({self.ad.degree + 1} control points); "
                              f"design carried over, largest change {err:.1e} (fraction of the chord)")

    def on_sections(self, _event=None):
        """Blade: the number of sections and the tip band from the Sections box."""
        s = self.ad.settings
        try:
            new = (int(self.sec["sections"].get()), float(self.sec["tip_band"].get()),
                   int(self.sec["tip_sections"].get()))
        except ValueError:
            new = None
        except tk.TclError:                                          # the box is gone (panel rebuilt)
            return
        if new and new != (s["sections"], s["tip_band"], s["tip_sections"]):
            if self.guard(lambda: self.ad.set_sections(*new) or True):
                self.schedule_update()
        for k, sb in self.sec.items():                              # what is used
            sb.set(f"{s[k]:.3f}" if k == "tip_band" else s[k])
        self.sec_info.configure(text=self.ad.sections_info())

    def on_hub_shape(self):
        """Blade: the hub as a cylinder or a cone (the adapter's set_hub_shape): new rows in the Blade and hub
        box."""
        shape = self.vhub.get()
        if shape == self.ad.hub_shape:
            return
        msg = self.guard(lambda: self.ad.set_hub_shape(shape))
        self.after_change(structure=True)
        if msg is not None:
            self._note = f"hub: {msg}"

    def on_hub_extends(self):
        """Blade: which end of the hub moves when its height changes (the adapter's set_hub_extends); the hub
        itself stays where it is."""
        end = self.vhub_end.get()
        got = self.guard(lambda: self.ad.set_hub_extends(end))
        if got is None:
            self.vhub_end.set(self.ad.hub_extends)
            return
        self.say("hub: a new height moves " + ("both ends, half each (the middle stays)" if got == "both" else
                                              "its upstream end (the tip stays)" if got == "fore" else
                                              "its tip (the base stays)"))

    def on_part(self):
        """Blade: what the CAD button writes (whole, half, one blade on 1/Z of the hub)."""
        part = self.vpart.get()
        got = self.guard(lambda: self.ad.set_part(part))
        if got is None:                                              # refused (half of an odd Z): as it was
            self.vpart.set(self.ad.settings["part"])
            return
        self.say(f"CAD writes {self.ad.part_label()}" + (" (solids)" if part != "sector" or self.ad.hub_shape == "cone"
                                                          else " (the DRDC surfaces of the MSc pipeline)"))
        self.refresh_labels()

    def on_blades(self, _event=None):
        """Blade: the number of blades from the Blade and hub box."""
        try:
            z = int(self.nblades.get())
        except ValueError:
            z = None
        except tk.TclError:                                          # the box is gone (panel rebuilt)
            return
        if z is not None and z != self.ad.params.blades:
            if self.guard(lambda: self.ad.set_blades(z) or True):
                self.refresh_labels()
                self.schedule_update()
        self.nblades.set(self.ad.params.blades)                      # what is used

    def on_span(self, _event=None):
        """Rudder: the full height from the Region box."""
        try:
            v = float(self.span_box.get())
        except ValueError:
            v = None
        except tk.TclError:                                          # the box is gone (panel rebuilt)
            return
        if v is not None and abs(v - self.ad.span) > 5e-3:
            r = self.guard(lambda: self.ad.set_span(v))
            if r is not None:
                self.after_change()
                self._note = f"full height {self.ad.span:.2f} mm: the rudder stretched by {r:.4g} in height"
        self.span_box.set(f"{self.ad.span:.2f}")

    def on_wing_sections(self, _event=None):
        """Wing, fin: the number of sections and of change points from the Span and sections box."""
        ad = self.ad
        try:
            n, k = int(self.wsec["sections"].get()), int(self.wsec["changes"].get())
        except ValueError:
            n = k = None
        except tk.TclError:                                          # the box is gone (panel rebuilt)
            return
        if n is not None and n != ad.settings["sections"]:
            if self.guard(lambda: ad.set_sections(n) or True):
                self.refresh_labels()
                self.schedule_update()
        if k is not None and k != ad.design.changes:
            msg = self.guard(lambda: ad.set_changes(k))
            if isinstance(msg, str):                                  # the curves changed: a new panel
                if self.curve.get() not in ad.names:
                    self.curve.set(ad.names[0])
                self.after_change(structure=True)
                self._note = f"{k} change points: {msg}"
                return
        self.wsec["sections"].set(ad.settings["sections"])           # what is used
        self.wsec["changes"].set(ad.design.changes)

    def on_elliptic(self):
        """Hull: elliptical sections (r' of its own) or circular ones."""
        self.guard(lambda: self.ad.set_elliptic(self.elliptic.get()))
        self.after_change(structure=True)

    def on_vehicle(self, _event=None):
        """The Vehicle tab's settings: read, checked and drawn."""
        pl = self.placement
        if pl is None:
            return
        try:
            new = {k: int(float(sb.get())) if k in COUNTS else float(sb.get()) for k, sb in self.vset.items()}
        except (ValueError, tk.TclError):
            new = None
        L = getattr(self.adapters["hull"].design, "length", float("inf")) if "hull" in self.adapters else float("inf")
        if new is not None and "rudder_rake" in new and not self.vrake.get():
            new["rudder_rake"] = None                                 # pitched to the hull's slope
        problem = None if new is None else self._xv().placement_problem(new, L)
        if problem:
            self.say(problem, error=True)
            new = None
        if new is not None:
            for k in ("rudder_angle", "fin_angle"):
                if k in new:
                    new[k] %= 360.0
            new["view"] = self.vshow.get()
            if any(getattr(pl, k) != v for k, v in new.items()):
                for k, v in new.items():
                    setattr(pl, k, v)
                self.schedule_update()
        self._fill_vehicle()

    @staticmethod
    def _xv():
        import xgeom_vehicle as XV
        return XV

    def _fill_vehicle(self):
        """The Vehicle tab's boxes show the placement in use."""
        pl = self.placement
        for k, sb in self.vset.items():
            if k == "rudder_rake":
                self._fill_rake()
                continue
            v = getattr(pl, k)
            sb.set(f"{v:d}" if k in COUNTS else f"{v:g}" if k.endswith("_angle") else
                   f"{v:.2f}" if k.endswith("_scale") else f"{v:.5f}" if k in POSITIONS else f"{v:.3f}")
        self.vshow.set(pl.view)

    def _fill_rake(self):
        """The rudders' TE rake: ticked and the value set, or unticked and greyed with the TE's rake that the
        hull's slope gives (the Vehicle tab's last view)."""
        v, sb = self.placement.rudder_rake, self.vset["rudder_rake"]
        self.vrake.set(v is not None)
        if v is None:
            got = (self._vinfo or {}).get("rudder_rake")
            v = round(float(got[0]), 2) if got else 0.0
        sb.configure(state="normal")
        sb.set(f"{v:g}")
        sb.configure(state="normal" if self.vrake.get() else "disabled")

    def say(self, text, error=False, busy=False, warn=False):
        """The status bar: green ok, amber busy (a build running), orange a
        warning (the build still runs), red a problem or an error."""
        bg, fg = STATUS["error" if error else "busy" if busy else "warn" if warn else "ok"]
        self.status.configure(text=text, background=bg, foreground=fg)

    # ------------------------------------------------------------ views
    def schedule_update(self):
        if not self._pending:
            self._pending = True
            self.root.after(15, self.update_views)

    def update_views(self):
        self._pending = False
        tab = self.book.index(self.book.select())
        if self.vehicle and tab == VEHICLE_TAB:
            try:
                self._update_vehicle()
            except Exception as exc:
                self.say(f"vehicle view failed: {type(exc).__name__}: {exc}", error=True)
                traceback.print_exc()
            return
        three_d = tab == 1
        try:
            pv = self.ad.preview(skeleton=three_d)
            if three_d:
                self.view3.update(pv)
                self.canvas3.draw_idle()
            else:
                self.views.update(pv, self.curve.get())
                self.canvas.draw_idle()
            lab = getattr(self, "inside_label", None)    # the rudder: how far it goes inside the original
            if lab is not None and lab.winfo_exists():
                m = pv.get("inside")
                lab.configure(text="" if m is None else "now: the original is enclosed" if m["depth"] <= 0.0 else
                              f"now: inside the original by up to {m['depth']:.3f} mm (at {100.0 * m['x_c']:.0f} % "
                              f"of its chord, height {m['y']:.1f} mm)",
                              foreground=INK2 if m is None or m["depth"] <= self.ad.settings["inside_tol_mm"] else RED)
            notes = self.ad.relation_notes()             # a relation dropped with a variable that is gone
            if notes:
                self._note = "; ".join(([self._note] if self._note else []) + notes)
            if pv["problems"]:
                self.say(f"problem (no {self.ad.build_label} until it is solved): " + "; ".join(pv["problems"]),
                         error=True)
            elif self._job is not None:
                self.say(self._job["message"], busy=True)
            elif pv.get("warnings"):
                self.say(f"warning ({self.ad.build_label} still possible): " + "; ".join(pv["warnings"])
                         + (f". {self._note}" if self._note else ""), warn=True)
            else:
                self.say(self._note or f"ok ({pv['seconds'] * 1e3:.0f} ms)")
            self._note = None
        except Exception as exc:
            self.say(f"preview failed: {type(exc).__name__}: {exc}", error=True)
            traceback.print_exc()

    def _vehicle_part(self, kind):
        """A component's lines for the Vehicle tab (xgeom_vehicle.PARTS), made again only when its design or
        its parameter file in use (the blade's Z) has changed."""
        import xgeom_vehicle as XV
        ad = self.adapters[kind]
        key = (ad.design, getattr(ad, "params", None))
        hit = self._vcache.get(kind)
        if hit is None or any(a is not b for a, b in zip(hit[0], key)):
            self._vcache[kind] = (key, XV.PARTS[kind](ad))
        return self._vcache[kind][1]

    def _update_vehicle(self):
        """The Vehicle tab: every component as its design stands, placed on the hull."""
        import xgeom_vehicle as XV
        t0 = time.time()
        for k in self.kinds:
            if k not in self.adapters and k not in self.errors:
                self.say(f"loading the {NAMES[k].lower()} for the vehicle ...", busy=True)
                self.root.update_idletasks()
                self._adapter(k)
        if "hull" not in self.adapters:
            self.say("the vehicle needs the hull: " + self.errors.get("hull", "not loaded"), error=True)
            return
        if body_of(self.adapters["hull"]) == "bwb":
            self.viewv.note("The Vehicle tab places the propeller, the rudders, the fins and the wings on the "
                            "cylindrical hull:\npick it in the Hull panel's Body box (the blended wing body's "
                            "vehicle comes later).")
            self.canvasv.draw_idle()
            self.say("vehicle: the Hull slot holds the blended wing body; the vehicle is placed on the cylindrical "
                     "hull (the Hull panel's Body box)", warn=True)
            return
        parts = {k: self._vehicle_part(k) for k in self.kinds if k in self.adapters}
        if self.placement is None:
            self.placement = XV.default_placement(parts)
            self._fill_vehicle()
        self._vinfo = self.viewv.update(parts, self.placement, self.kind)
        if self.placement.rudder_rake is None and "rudder_rake" in self.vset:
            self._fill_rake()                                         # the rake the hull's slope gives
        self.canvasv.draw_idle()
        parts = {k: parts[k] for k in XV.active(parts, self.placement)}      # on the vehicle (a count above 0)
        problems = [f"{NAMES[k].lower()}: {q}" for k, part in parts.items() for q in part["problems"]]
        problems += [f"{NAMES[k].lower()} not loaded: {e}" for k, e in self.errors.items()]
        warnings = [f"{NAMES[k].lower()}: {q}" for k, part in parts.items() for q in part.get("warnings", ())]
        if problems:
            self.say("problem: " + "; ".join(problems), error=True)
        elif self._job is not None:
            self.say(self._job["message"], busy=True)
        elif warnings:
            self.say("warning: " + "; ".join(warnings), warn=True)
        else:
            self.say(f"vehicle ok ({(time.time() - t0) * 1e3:.0f} ms)")

    # ------------------------------------------------------------ build
    def build_label(self, kind):
        """The build button's text: the component's (in the vehicle with its name)."""
        ad = self.adapters[kind]
        return f"{ad.build_label} {self._noun(kind)}" if self.vehicle else ad.build_label

    def on_build(self, kind=None, save_to=None):
        """The left panel's build button, made for one component (kind): that component writes its case and
        its CAD (its adapter's build_task, run off the GUI thread); with save_to (Save CAD) the CAD is then
        written to that file too."""
        kind = kind or self.kind
        ad = self.adapters.get(kind)
        if self._job is not None or ad is None:
            return
        label = self.build_label(kind)
        try:
            problems = ad.preview()["problems"]
            if problems:
                self.say(f"{label}: the design cannot be built: " + problems[0], error=True)
                return
            case = self.case.get().strip() or "gui"
            task = ad.build_task(case)                                  # the design as it is now
            keys = {kind: self._design_key(kind)}
        except Exception as exc:
            self._log(label, "-", 0.0, traceback.format_exc())
            self.say(f"{label} failed: {type(exc).__name__}: {exc}", error=True)
            return
        self._start(kind, label, case, lambda job: then_save(task(), kind, save_to),
                    f"{label} running: writing {case} and building the CAD"
                    + (f", then saving {os.path.basename(save_to)}" if save_to else ""), keys)

    def on_vehicle_build(self, save_to=None):
        """The Vehicle tab's Vehicle CAD: each component's own case and CAD, then the parts placed as the tab
        shows them and fused into one solid (vehicle_cad.run_vehicle); with save_to (Save vehicle CAD) the
        vehicle's CAD is then written to that file too."""
        import dataclasses
        import xgeom_vehicle as XV
        if self._job is not None:
            return
        label = "Vehicle CAD"
        try:
            for k in self.kinds:
                if k not in self.adapters and k not in self.errors:
                    self._adapter(k)
            if "hull" not in self.adapters:
                raise RuntimeError("the vehicle needs the hull: " + self.errors.get("hull", "not loaded"))
            if body_of(self.adapters["hull"]) == "bwb":
                self.say(f"{label}: the vehicle is placed on the cylindrical hull; the Hull slot holds the blended "
                         f"wing body (pick the cylindrical hull in the Hull panel's Body box)", error=True)
                return
            parts = {k: self._vehicle_part(k) for k in self.kinds if k in self.adapters}
            if self.placement is None:
                self.placement = XV.default_placement(parts)
            placement = dataclasses.replace(self.placement)
            parts = {k: parts[k] for k in XV.active(parts, placement)}       # a component whose count is 0: none
            problems = [f"{NAMES[k].lower()}: {q}" for k, part in parts.items() for q in part["problems"]]
            if problems:
                self.say(f"{label}: the vehicle cannot be built: " + problems[0], error=True)
                return
            info = XV.assemble(parts, placement)[1]
            case = self.case.get().strip() or "gui"
            # full bodies (fused); the blade: the part its panel's 'CAD writes' names, as a solid
            tasks = {k: self.adapters[k].build_task(case, half=False) for k in parts}
            keys = {k: self._design_key(k) for k in parts}
            keys["vehicle"] = self._design_key("vehicle")
        except Exception as exc:
            self._log(label, "-", 0.0, traceback.format_exc())
            self.say(f"{label} failed: {type(exc).__name__}: {exc}", error=True)
            return

        def run(job):
            import vehicle_cad as VC

            def progress(text):
                job["message"] = f"{label} running: {text} ..."
            return then_save(VC.run_vehicle(VEHICLE_DIR, case, tasks, parts, info, placement, progress), "vehicle",
                             save_to)
        self._start("vehicle", label, case, run, f"{label} running: the components' CAD, then the vehicle"
                    + (f", then saving {os.path.basename(save_to)}" if save_to else ""), keys)

    def _design_key(self, kind):
        """What a build depends on, to tell whether the last build is of the design as it stands: the
        component's design and parameter file (its adapter makes new ones on every change) and its settings;
        for 'vehicle' its components' keys and the placement (not the view)."""
        if kind != "vehicle":
            ad = self.adapters[kind]
            return ad.design, getattr(ad, "params", None), repr(getattr(ad, "settings", None))
        import dataclasses
        import xgeom_vehicle as XV
        parts = {k: self._vehicle_part(k) for k in self.kinds if k in self.adapters}
        pl = self.placement
        if pl is None:
            return ()
        place = tuple((f.name, getattr(pl, f.name)) for f in dataclasses.fields(pl) if f.name != "view")
        return tuple((k, self._design_key(k)) for k in XV.active(parts, pl)) + (place,)

    def on_save_cad(self, kind=None):
        """Save CAD ...: the CAD of a component (kind 'vehicle': of the vehicle) as its design stands, to a
        STEP or IGES file picked here (CAD_TYPES): the last build's when it is of the design as it stands,
        else the build runs first (on_build, on_vehicle_build) and then writes the file."""
        kind = kind or self.kind
        if self._job is not None or (kind != "vehicle" and kind not in self.adapters):
            return
        if kind == "vehicle" and body_of(self.adapters.get("hull")) == "bwb":
            self.say("Save vehicle CAD: the vehicle is placed on the cylindrical hull; the Hull slot holds the "
                     "blended wing body (pick the cylindrical hull in the Hull panel's Body box)", error=True)
            return
        import xcad_loft as XL
        case = self.case.get().strip() or "gui"
        vehicle = kind == "vehicle"
        start = self.save_dir or os.path.abspath(VEHICLE_DIR if vehicle else self.adapters[kind].out_dir)
        name = "the vehicle" if vehicle else f"the {(getattr(self.adapters[kind], 'name', None) or NAMES.get(kind, kind)).lower()}"
        path = filedialog.asksaveasfilename(title=f"Save the CAD of {name}", defaultextension=".step",
                                            initialdir=start if os.path.isdir(start) else HERE,
                                            initialfile=f"{case}_vehicle.step" if vehicle else f"{case}.step",
                                            filetypes=CAD_TYPES)
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".step"
        if os.path.splitext(path)[1].lower() not in XL.CAD_EXTENSIONS:
            self.say(f"Save CAD: {os.path.basename(path)} is not a .step, .stp, .igs or .iges file", error=True)
            return
        self.save_dir = os.path.dirname(path)
        built = self.built.get(kind)
        if built and os.path.exists(built["step"]) and same(built["key"], self._design_key(kind)):
            def run(job):                                           # the last build is of this design
                how = save_built(kind, built["step"], path)
                return {"view": built["step"], "error": False, "report": [f"saved: {path} ({how})"],
                        "message": f"{path} ({how})"}
            self._start("save", "Save CAD", case, run, f"Save CAD: writing {os.path.basename(path)}")
        elif vehicle:
            self.on_vehicle_build(save_to=path)
        else:
            self.on_build(kind, save_to=path)

    def _start(self, kind, label, case, run, message, keys=None, then=None):
        """Run a build off the GUI thread: run(job) returns the result dict (report, message, error, view).
        keys: the build keys (_design_key) of what it builds, by kind, kept with its STEP for Save CAD. then:
        called with the result on the GUI thread when the job succeeds (From CAD loads its set-up)."""
        job = self._job = {"kind": kind, "label": label, "case": case, "done": False, "result": None,
                           "error": None, "t0": time.time(), "message": message + " ...", "keys": keys or {},
                           "then": then}
        self._buttons()
        self.say(job["message"], busy=True)

        def work():
            try:
                job["result"] = run(job)
            except Exception:
                job["error"] = traceback.format_exc()
            job["done"] = True
        threading.Thread(target=work, daemon=True).start()
        self.root.after(200, self._poll)

    def _buttons(self):
        """The build buttons: disabled while a build runs, the running one counting its seconds."""
        job = self._job
        btns = [(getattr(self, "build_btn", None), self.kind), (getattr(self, "vbuild_btn", None), "vehicle")]
        for btn, kind in btns:
            if btn is None or not btn.winfo_exists():
                continue
            if job is None:
                btn.configure(state="normal", text="Vehicle CAD" if kind == "vehicle" else self.build_label(kind))
            else:
                btn.configure(state="disabled")
                if job["kind"] == kind:
                    btn.configure(text=f"{time.time() - job['t0']:.0f} s")     # the build's time so far
        for btn in (getattr(self, "save_btn", None), getattr(self, "vsave_btn", None)):
            if btn is not None and btn.winfo_exists():
                btn.configure(state="normal" if job is None else "disabled")

    def _poll(self):
        job = self._job
        if not job["done"]:
            self._buttons()
            if self.status.cget("text") != job["message"]:
                self.say(job["message"], busy=True)
            self.root.after(200, self._poll)
            return
        self._job = None
        self._buttons()
        label, secs = job["label"], time.time() - job["t0"]
        if job["error"]:
            print(job["error"])
            self._log(label, job["case"], secs, job["error"])
            self.say(f"{label} failed: " + job["error"].strip().splitlines()[-1], error=True)
            return
        res = job["result"]
        try:
            print("\n".join(res["report"]))
        except Exception:                                           # a console that cannot show the report
            pass
        if res.get("view"):
            self.step = res["view"]
            steps = dict(res.get("record", {}).get("steps", {}), **{job["kind"]: res["view"]})
            for k, key in job["keys"].items():                     # the builds Save CAD can use as they are
                if steps.get(k):
                    self.built[k] = {"key": key, "step": steps[k]}
        self._log(label, job["case"], secs, res["message"])
        self.say(f"{label} done in {secs:.1f} s: {res['message']}", error=res["error"], warn=res.get("warn", False))
        if job.get("then") and not res["error"]:
            try:
                job["then"](res)
            except Exception as exc:
                traceback.print_exc()
                self.say(f"{label}: {type(exc).__name__}: {exc}", error=True)

    @staticmethod
    def _log(label, case, seconds, text):
        """One line per build (and the traceback of a failure) in LOG, to look at afterwards."""
        try:
            os.makedirs(os.path.dirname(LOG), exist_ok=True)
            with open(LOG, "a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {label}  case {case}  {seconds:.1f} s\n"
                         + "".join(f"    {line}\n" for line in str(text).strip().splitlines()))
        except OSError:
            pass

    def save_target(self):
        """Where Save set-up's dialog opens: (folder, file name) of the file the component came from, else the
        output folder and the case."""
        cur = getattr(self.ad, "file", None)
        if cur:
            return os.path.dirname(cur), os.path.basename(cur)
        return os.path.join(HERE, self.ad.out_dir), f"{self.case.get().strip() or 'gui'}.txt"

    def on_save(self):
        """Save set-up: the component as a component file (the dialog opens on the file it came from; with
        none, on the output folder and the case)."""
        folder, name = self.save_target()
        what = (getattr(self.ad, "name", None) or NAMES.get(self.kind, self.kind)).lower()
        path = filedialog.asksaveasfilename(title=f"Save the {what}'s set-up as a component file",
                                            initialdir=folder, initialfile=name,
                                            defaultextension=".txt",
                                            filetypes=[("component files", "*.txt"), ("all files", "*")])
        if path and self.guard(lambda: self.ad.save_file(path)):
            self._title()
            self.rebuild_panel()
            self.say(f"set-up saved: {path}")

    def on_load_params(self):
        """Load ...: a component file takes the place of the design in the tool, and the whole panel follows
        it. The Rudder slot: a planar rudder's file (kind rudder) becomes the original of the model modified
        below H, or, with a new rudder in the slot, loads into it; a set-up modified below H (kind rudder_h)
        loads into that model, the slot put on it."""
        cur = getattr(self.ad, "file", None)
        rudder = self.kind == "rudder"
        start = os.path.dirname(cur) if cur else os.path.join(HERE, self.ad.out_dir) if rudder else HERE
        path = filedialog.askopenfilename(title=f"{NAMES.get(self.kind, self.kind)}: a component file",
                                          initialdir=start if os.path.isdir(start) else HERE,
                                          filetypes=[("component files", "*.txt"), ("all files", "*")])
        if path and rudder and planar_rudder_file(path) and not is_planar(self.ad):
            self._load_planar_rudder(path)               # a planar rudder: the original modified below H
            return
        if path and self.kind == "hull" and body_of_file(path) != body_of(self.ad):    # the Hull slot's other body
            want, have = body_of_file(path), body_of(self.ad)

            def other():
                module, cls = BODIES[want]
                new = getattr(importlib.import_module(module), cls)(params=path)
                self.bodies.pop(want, None)
                self.bodies[have] = self.ad
                self.replace_adapter("hull", new)
                return True
            self.say(f"reading {os.path.basename(path)} ...")
            self.root.update_idletasks()
            if self.guard(other):
                self._note = f"loaded {path}: the {BODY_NAMES[want]} in the Hull slot"
                self.say(self._note)
            return
        if path and rudder and is_planar(self.ad) and modified_rudder_file(path):
            def back():                                  # the model modified below H from its set-up
                import xgeom_rudder as XR
                ad = XR.RudderAdapter(params=path)
                self.rudders.pop("modify", None)         # the new rudder keeps its state for the Rudder box
                self.rudders["new"] = self.ad
                self.replace_adapter("rudder", ad)
                return True
            self.say(f"reading {os.path.basename(path)} ...")
            self.root.update_idletasks()
            if self.guard(back):
                self._note = f"set-up loaded: {path} (the rudder modified below H)"
                self.say(self._note)
            return
        if path:
            self.say(f"reading {os.path.basename(path)} ...")
            self.root.update_idletasks()
            if self.guard(lambda: self.ad.load_params(path) or True):
                self._title()
                self.after_change(structure=True)
                self._note = f"loaded {path}"            # stays once the views are drawn
                self.say(self._note)

    def on_from_cad(self):
        """From CAD ...: a STEP or IGES file of the component; off the GUI thread its parameters are extracted
        (xgeom_from_cad), fitted in the form with the least error and saved as a set-up (a component file),
        which is then loaded, so the panel shows where the starting geometry's parameters lie."""
        if self._job is not None:
            return
        import xgeom_from_cad as FC
        kind = self.kind
        if kind not in FROM_CAD_KINDS:
            self.say(f"From CAD: no extraction for the {NAMES.get(kind, kind).lower()} yet", error=True)
            return
        start = self.save_dir or os.path.abspath(os.path.join(HERE, self.ad.out_dir))
        path = filedialog.askopenfilename(title=f"{NAMES.get(kind, kind)}: a CAD file to extract",
                                          initialdir=start if os.path.isdir(start) else HERE,
                                          filetypes=CAD_TYPES + [("all files", "*")])
        if not path:
            return
        case = FC.case_name(path)
        out_dir = os.path.abspath(os.path.join(HERE, self.ad.out_dir))
        label = f"From CAD ({NAMES.get(kind, kind).lower()})"

        def run(job):
            def progress(text):
                job["message"] = f"{label}: {text} ..."
            res = FC.from_cad(kind, path, out_dir, case, progress)
            res["report"] = [f"{label}: {os.path.basename(path)} -> {res['setup']}"] + res["report"]
            res.update(view=None, error=False, message=f"{os.path.basename(res['setup'])} from "
                                                       f"{os.path.basename(path)} ({res['seconds']:.0f} s)")
            return res
        self._start(kind, label, case, run, f"{label}: reading {os.path.basename(path)}", then=self._from_cad_done)

    def _from_cad_done(self, res):
        """After From CAD: the extracted file loaded (the blade's and hull's parameter tables, the rudder's
        set-up), each curve put in the form that fits best (the blade, the hull: the adapter refits it to the
        file, as a change in the panel would), the panel rebuilt."""
        ad, kind = self.ad, self.kind
        if kind == "rudder" and is_planar(ad):          # the extraction is the modified-below-H model's
            import xgeom_rudder as XR
            ad = XR.RudderAdapter(params=res["setup"])
            self.replace_adapter("rudder", ad)
        elif kind == "rudder":
            ad.load_params(res["setup"])
        elif kind == "blade":
            ad.load_params(res["file"])
            ad.set_forms({c: form for c, (form, _) in res["forms"].items()})
        else:
            ad.load_params(res["file"])
            for c, (form, _) in res["forms"].items():
                ad.set_order(c, form)
        self._title()
        self.after_change(structure=True)
        forms = "; ".join(res["report"][1:1 + len(res["forms"])])
        self._note = f"from {os.path.basename(res['file'])}: {forms}; set-up saved as {os.path.basename(res['setup'])}"
        self.say(self._note)

    def _load_planar_rudder(self, path):
        """A planar rudder's component file in the Rudder slot: off the GUI thread it becomes the original of
        the model modified below H (xgeom_from_cad.rudder_from_planar: its sections written and fitted beside
        the file the first time, read back after), which then takes the slot; the panel stays the one of that
        model."""
        if self._job is not None:
            return
        import xgeom_from_cad as FC
        label = "Planar rudder"

        def run(job):
            def progress(text):
                job["message"] = f"{label}: {text} ..."
            res = FC.rudder_from_planar(path, progress=progress)
            res.update(view=None, error=False,
                       message=f"{os.path.basename(path)} as the original modified below H"
                               + (" (made before)" if res["cached"] else f" ({res['seconds']:.0f} s)"))
            return res
        self._start("rudder", label, "load", run, f"{label}: {os.path.basename(path)} as the original modified "
                                                   f"below H", then=self._planar_rudder_done)

    def _planar_rudder_done(self, res):
        """After _load_planar_rudder: the rudder modified below H on that original in the slot."""
        self.replace_adapter("rudder", res["adapter"])
        self._note = f"rudder loaded: {res['name']} as the original modified below H, its set-up {res['setup']}"
        self.say(self._note)

    def open_occ_viewer(self):
        if not self.step:
            self.say(f"press {self.build_label(self.kind)} first: the viewer shows the CAD it writes", error=True)
            return
        subprocess.Popen([sys.executable, os.path.join(HERE, "xcad_loft.py"), "--view-step", self.step], cwd=HERE)
        self.say(f"OCC viewer: {os.path.basename(self.step)} (its own window, a few seconds)")


def rudder_box(app):
    """The Rudder slot's first box (26 Nov 2026): modify an existing rudder (the model modified below H,
    xgeom_rudder) or design a new one (the planar model, xgeom_wing; a rudder of its own, no original), each
    keeping its state while the other is shown. Only where the planar model is (the X_geom folder)."""
    if importlib.util.find_spec(PLANAR_RUDDER[0]) is None:
        return
    box = ttk.LabelFrame(app.panel, text="Rudder")
    box.pack(fill="x", padx=6, pady=4)
    row = ttk.Frame(box)
    row.pack(fill="x", pady=2)
    option = rudder_of(app.ad)
    app.vrudder = tk.StringVar(value=option)
    for o in RUDDERS:
        ttk.Radiobutton(row, text=RUDDER_NAMES[o], value=o, variable=app.vrudder,
                        command=app.on_rudder).pack(side="left", padx=4)
    ttk.Label(box, text=("the original stays above H and is changed below it; there it is the lower bound of the "
                         "shape (keep the original inside, in the Region box)" if option == "modify" else
                         "a rudder of its own, no original: its span and its distributions along it. Saved (Save "
                         "set-up), it loads in the other option as the original to modify"),
              foreground=INK2, wraplength=590, justify="left").pack(anchor="w", padx=4)


def rudder_panel(app):
    """The rudder's boxes: the two options, the file, the region H, the full height, the original as the
    lower bound, and the section control points."""
    import xgeom_rudder as XR
    ad = app.ad
    rudder_box(app)
    table = ad.files["section_table"]
    file_box(app, ad.file or "(no set-up file: the design started from modify_control.py)",
             f"the original: {os.path.relpath(table, HERE) if os.path.abspath(table).startswith(HERE) else table} "
             f"(its section table; the tip, the cap and the extracted sections with it), sections of degree "
             f"{ad.degree}, full height {ad.span:.6g} mm")

    box = ttk.LabelFrame(app.panel, text="Region")
    box.pack(fill="x", padx=6, pady=4)
    app._header(box)
    app.rows.append(VarRow(app, box, 1, Row("var", "H (mm)", "H", 3)))
    row = ttk.Frame(box)
    row.grid(row=grid_row(2), column=0, columnspan=COLS, sticky="w", padx=2, pady=(4, 2))
    ttk.Label(row, text="full height").pack(side="left", padx=(2, 4))
    app.span_box = ttk.Spinbox(row, from_=round(XR.SPAN_RANGE[0] * ad.span0, 1),
                               to=round(XR.SPAN_RANGE[1] * ad.span0, 1), increment=1.0, width=8, command=app.on_span)
    app.span_box.set(f"{ad.span:.2f}")
    app.span_box.bind("<Return>", app.on_span)
    app.span_box.bind("<FocusOut>", app.on_span)
    app.span_box.pack(side="left")
    ttk.Label(row, text=f"mm, the root to the top of the cap (the extracted rudder's: {ad.span0:.1f} mm); the "
                        f"whole rudder stretches in height, H and its bounds with it", foreground=INK2,
              wraplength=430, justify="left").pack(side="left", padx=6)
    state = ad.tangent_state()
    row = ttk.Frame(box)
    row.grid(row=grid_row(3), column=0, columnspan=COLS, sticky="w", padx=2, pady=(2, 2))
    app.tangent_all = tk.BooleanVar(value=state == "all")
    ttk.Checkbutton(row, text="tangent at H", variable=app.tangent_all,
                    command=app.on_tangent_all).pack(side="left", padx=(2, 4))
    n_g1 = sum(ad.design.curves[c].join == "G1" for c in ad.names)
    ttk.Label(row, text=("every curve leaves the original with its slope (C1 on its tangent); off: with its value "
                         "only, C1 free, and the CAD lofts the rudder below and above H on its own and joins the "
                         "two there (nine patches)" if state == "all" else
                         "no curve tangent: C1 free on each; the CAD lofts the rudder below and above H on its own "
                         "and joins the two there (nine patches)" if state == "none" else
                         f"{n_g1} of {len(ad.names)} curves tangent (each curve's own switch below); the CAD lofts "
                         f"the rudder below and above H on its own and joins the two there (nine patches); tick: "
                         f"every curve tangent"),
              foreground=INK2, wraplength=430, justify="left").pack(side="left", padx=6)
    row = ttk.Frame(box)
    row.grid(row=grid_row(4), column=0, columnspan=COLS, sticky="w", padx=2, pady=(2, 2))
    app.keep_inside = tk.BooleanVar(value=bool(ad.settings.get("keep_inside")))
    ttk.Checkbutton(row, text="keep the original inside, within", variable=app.keep_inside,
                    command=app.on_keep_inside).pack(side="left", padx=(2, 2))
    app.inside_tol = ttk.Entry(row, width=6, justify="right")
    app.inside_tol.insert(0, f"{ad.settings.get('inside_tol_mm', XR.INSIDE_TOL):g}")
    app.inside_tol.bind("<Return>", app.on_keep_inside)
    app.inside_tol.bind("<FocusOut>", app.on_keep_inside)
    app.inside_tol.pack(side="left")
    ttk.Label(row, text="mm").pack(side="left", padx=(2, 6))
    app.inside_label = ttk.Label(row, text="", foreground=INK2, wraplength=330, justify="left")
    app.inside_label.pack(side="left")
    ttk.Label(box, text="the original is the lower bound of the shape: measured on the built sections, a warning "
                        "beyond the tolerance (saved with the set-up, for an optimiser to keep)",
              foreground=INK2, wraplength=590, justify="left").grid(row=grid_row(5), column=0, columnspan=COLS,
                                                                    sticky="w", padx=4)

    box = ttk.LabelFrame(app.panel, text="Section control points (pick the curve to edit)")
    box.pack(fill="x", padx=6, pady=4)
    top = ttk.Frame(box)
    top.pack(fill="x", pady=2)
    ttk.Label(top, text="control points").pack(side="left", padx=(4, 2))
    app.npts = ttk.Spinbox(top, from_=5, to=13, width=4, command=app.on_points)
    app.npts.set(ad.degree + 1)
    app.npts.bind("<Return>", app.on_points)
    app.npts.pack(side="left")
    ttk.Label(top, text=f"P0 .. P{ad.degree}, Bezier of degree {ad.degree}", foreground=INK2).pack(side="left", padx=6)
    grid = ttk.Frame(box)
    grid.pack(fill="x", padx=4, pady=2)
    app._pick(grid, 0, 1, "LE", "LE x (planform)")
    ttk.Label(grid, text="P0   (0, 0), the LE", foreground=INK2).grid(row=1, column=0, columnspan=3, sticky="w")
    for i in range(1, ad.degree):
        ttk.Label(grid, text=f"P{i}", width=4).grid(row=i + 1, column=0, sticky="w")
        app._pick(grid, i + 1, 1, f"P{i}x", "x")
        app._pick(grid, i + 1, 2, f"P{i}z", "z")
    ttk.Label(grid, text=f"P{ad.degree}   (1, 0), the sharp TE", foreground=INK2).grid(
        row=ad.degree + 1, column=0, columnspan=3, sticky="w")


def file_box(app, path, info):
    """The panel's first box: the component's file (relative to this folder when inside it), what it holds,
    and Load ... (Save set-up below writes it)."""
    box = ttk.LabelFrame(app.panel, text="File")
    box.pack(fill="x", padx=6, pady=4)
    ttk.Label(box, text=os.path.relpath(path, HERE) if path.startswith(HERE) else path,
              wraplength=560).pack(anchor="w", padx=4)
    ttk.Label(box, text=info, foreground=INK2, wraplength=560).pack(anchor="w", padx=4)
    row = ttk.Frame(box)
    row.pack(fill="x", pady=3)
    ttk.Button(row, text="Load ...", command=app.on_load_params).pack(side="left", padx=4)
    from_cad = app.kind in FROM_CAD_KINDS and not is_planar(app.ad)    # no planar extraction yet
    if from_cad:
        ttk.Button(row, text="From CAD ...", command=app.on_from_cad).pack(side="left", padx=(0, 4))
    planar = app.kind == "rudder" and is_planar(app.ad)
    ttk.Label(row, text="a component file (.txt); Save set-up (below) writes the design as one"
                        + ("; From CAD ...: a STEP or IGES file, its parameters extracted and fitted (the form "
                           "with the least error), the set-up saved" if from_cad else "")
                        + ("; a set-up of the rudder modified below H (kind rudder_h) puts the slot on that "
                           "option" if planar else "")
                        + ("; a planar rudder's file (kind rudder) becomes the original modified below H"
                           if app.kind == "rudder" and not planar
                           and importlib.util.find_spec(PLANAR_RUDDER[0]) is not None else "")
                        + ("; a cylindrical hull's or a blended wing body's file (kind hull, bwb) puts the slot on "
                           "that body" if app.kind == "hull" else ""),
              foreground=INK2, wraplength=470 if not from_cad else 380).pack(side="left", padx=6)


def blade_panel(app):
    """The blade's boxes: the file, the blade and hub, the sections and the
    distributions."""
    import blade_modify as BM
    ad = app.ad
    p = ad.params
    hub_h = f"not given ({BM.HUB_HEIGHT:g} m)" if p.hub_height is None else f"{p.hub_height:g} m"
    if p.airfoil:
        import airfoils
        sections = "airfoil " + airfoils.describe(p.airfoil)
    else:
        sections = f"sections: {p.section_table}"
    z_file = getattr(ad, "file_blades", p.blades)
    file_box(app, ad.file, f"D {p.diameter:g} m, {z_file} blades, root r/R {p.root_r:g}, hub height {hub_h}, "
                           f"{len(p.r)} rows; {sections}")

    box = ttk.LabelFrame(app.panel, text="Blade and hub (the curves stretch over the span between them)")
    box.pack(fill="x", padx=6, pady=4)
    app._header(box)
    specs = ad.global_rows()
    for r, spec in enumerate(specs, start=1):
        app.rows.append(VarRow(app, box, r, spec))
    row = ttk.Frame(box)
    row.grid(row=grid_row(len(specs) + 1), column=0, columnspan=COLS, sticky="w", padx=2, pady=(4, 2))
    ttk.Label(row, text="hub").pack(side="left", padx=(2, 4))
    app.vhub = tk.StringVar(value=ad.hub_shape)
    for shape in ad.hub_shapes:
        ttk.Radiobutton(row, text=shape, value=shape, variable=app.vhub, command=app.on_hub_shape).pack(side="left")
    ttk.Label(row, text=("a cone from the base radius (the larger end, upstream, at hub base x) to the top "
                         "radius, closed by the sphere tangent to it through the tip, hub height from the base; "
                         "root r is where the table starts: the blade runs on into the hub, which trims it"
                         if ad.hub_shape == "cone" else
                         "a cylinder of the hub radius (the blade root ring lies on it), centred on x = 0 until its "
                         "height changes at one end; a cone: base and top radii, the base's x and a rounded nose"),
              foreground=INK2, wraplength=440, justify="left").pack(side="left", padx=6)
    row = ttk.Frame(box)
    row.grid(row=grid_row(len(specs) + 2), column=0, columnspan=COLS, sticky="w", padx=2, pady=(2, 2))
    ttk.Label(row, text="hub extends").pack(side="left", padx=(2, 4))
    app.vhub_end = tk.StringVar(value=ad.hub_extends)
    for end in ad.hub_ends:
        ttk.Radiobutton(row, text=end, value=end, variable=app.vhub_end, command=app.on_hub_extends).pack(side="left")
    ttk.Label(row, text="the end that moves when the hub height changes: fore (upstream; the tip stays), aft (the "
                        "tip; the base stays) or both (half each; the middle stays)",
              foreground=INK2, wraplength=440, justify="left").pack(side="left", padx=6)
    row = ttk.Frame(box)
    row.grid(row=grid_row(len(specs) + 3), column=0, columnspan=COLS, sticky="w", padx=2, pady=(2, 2))
    ttk.Label(row, text="blades Z").pack(side="left", padx=(2, 4))
    app.nblades = ttk.Spinbox(row, from_=BM.BLADES[0], to=BM.BLADES[1], width=4, command=app.on_blades)
    app.nblades.set(p.blades)
    app.nblades.bind("<Return>", app.on_blades)
    app.nblades.bind("<FocusOut>", app.on_blades)
    app.nblades.pack(side="left")
    ttk.Label(row, text=f"the file has {z_file}; the clearance check, the propeller tab, the vehicle and the "
                        f"CAD's hub sector follow it; blades that interfere give a warning", foreground=INK2,
              wraplength=470, justify="left").pack(side="left", padx=6)

    box = ttk.LabelFrame(app.panel, text="Sections (the XCAD points file and the 3D sections tab; 53 points each)")
    box.pack(fill="x", padx=6, pady=4)
    row = ttk.Frame(box)
    row.pack(fill="x", pady=2)
    app.sec = {}
    for key, text, kw in (("sections", "sections", dict(from_=3, to=999, width=5)),
                          ("tip_sections", "the last", dict(from_=2, to=998, width=4)),
                          ("tip_band", "from r/R", dict(from_=0.5, to=0.99, increment=0.01, width=6))):
        ttk.Label(row, text=text).pack(side="left", padx=(4, 2))
        sb = ttk.Spinbox(row, command=app.on_sections, **kw)
        sb.pack(side="left")
        sb.bind("<Return>", app.on_sections)
        sb.bind("<FocusOut>", app.on_sections)
        app.sec[key] = sb
    ttk.Label(row, text="to the tip").pack(side="left", padx=4)
    app.sec_info = ttk.Label(box, foreground=INK2, wraplength=560, justify="left")
    app.sec_info.pack(anchor="w", padx=4)
    app.on_sections()                                                # fills the boxes

    box = ttk.LabelFrame(app.panel, text="Distributions (pick the curve to edit)")
    box.pack(fill="x", padx=6, pady=4)
    grid = ttk.Frame(box)
    grid.pack(fill="x", padx=4, pady=2)
    for i, c in enumerate(ad.names):
        app._pick(grid, i // 3, i % 3, c, f"{c}  {BM.UNITS[c]}")


def body_box(app):
    """The Hull slot's first box: its body, the cylindrical hull (xgeom_hull) or the blended wing body (xgeom_wing,
    kind bwb; 20 Nov 2026), each keeping its state while the other is shown."""
    box = ttk.LabelFrame(app.panel, text="Body")
    box.pack(fill="x", padx=6, pady=4)
    row = ttk.Frame(box)
    row.pack(fill="x", pady=2)
    body = body_of(app.ad)
    app.vbody = tk.StringVar(value=body)
    for b, (module, _) in BODIES.items():
        if importlib.util.find_spec(module) is not None:
            ttk.Radiobutton(row, text=BODY_NAMES[b], value=b, variable=app.vbody,
                            command=app.on_body).pack(side="left", padx=4)
    ttk.Label(row, text=("a body of revolution, or with elliptical sections: stations along x, the nose and the tail "
                         "as CST curves" if body == "hull" else
                         "the wing's model over the half-span from the centre line, mirrored; every curve flat at the "
                         "centre line"), foreground=INK2, wraplength=330, justify="left").pack(side="left", padx=8)


def hull_panel(app):
    """The hull's boxes: the body choice, the file, the body (lengths, radii, elliptical sections) and the
    curves."""
    ad = app.ad
    p = ad.params
    body_box(app)
    file_box(app, ad.file or "(SUBOFF's equations: suboff_hull.txt is not here)",
             f"L {p.length:g} m: nose {p.nose:g}, middle {p.middle:g}, tail {p.tail:g}, cap {p.cap:g} m; "
             f"{len(p.x)} stations, " + ("axisymmetric" if p.axisymmetric else "elliptical sections (r and r')"))

    box = ttk.LabelFrame(app.panel, text="Body: the parts' lengths and the radii (m)")
    box.pack(fill="x", padx=6, pady=4)
    app._header(box)
    specs = ad.global_rows()
    for r, spec in enumerate(specs, start=1):
        app.rows.append(VarRow(app, box, r, spec))
    app.elliptic = tk.BooleanVar(value=ad.elliptic)
    ttk.Checkbutton(box, text="elliptical sections: the half-width r' has radii and curves of its own",
                    variable=app.elliptic, command=app.on_elliptic).grid(row=grid_row(len(specs) + 1), column=0,
                                                                         columnspan=COLS, sticky="w", padx=2,
                                                                         pady=(4, 2))

    box = ttk.LabelFrame(app.panel, text="Curves (pick the curve to edit)")
    box.pack(fill="x", padx=6, pady=4)
    grid = ttk.Frame(box)
    grid.pack(fill="x", padx=4, pady=2)
    for i, c in enumerate(ad.names):
        text = (c.replace("_", " ") if ad.design.axisymmetric else
                {"nose_r": "nose r (XZ)", "tail_r": "tail r (XZ)", "nose_rp": "nose r' (XY)",
                 "tail_rp": "tail r' (XY)"}[c])
        app._pick(grid, i // 2, i % 2, c, text)


def wing_panel(app):
    """The wing's (or the fin's, the planar rudder's, the blended wing body's) boxes: the file, the span and the
    sections, the distributions and the section changes; in the Hull slot the body choice first."""
    import airfoils
    import wing_modify as WM
    ad = app.ad
    p = ad.params
    if app.kind == "hull":
        body_box(app)
    if app.kind == "rudder":
        rudder_box(app)
    symmetric = WM.base_shape(p.airfoil, p.camber_xc)[2]["symmetric"]
    file_box(app, ad.file, (f"half-span {p.span:g} mm (from the centre line), " if p.flat else f"span {p.span:g} mm, ")
             + f"{len(p.eta)} rows, pitch axis at {p.ref_xc:g} of the chord; airfoil {airfoils.describe(p.airfoil)}"
                           + (f" (symmetric: f/c scales the NACA 4-digit camber line, its maximum at {p.camber_xc:g} c)"
                              if symmetric else "") + (f"; {p.changes} change points" if p.changes else ""))

    box = ttk.LabelFrame(app.panel, text="Half-span and sections (the curves stretch over the half-span)" if p.flat
                         else "Span and sections (the curves stretch over the span)")
    box.pack(fill="x", padx=6, pady=4)
    app._header(box)
    specs = ad.global_rows()
    for r, spec in enumerate(specs, start=1):
        app.rows.append(VarRow(app, box, r, spec))
    row = ttk.Frame(box)
    row.grid(row=grid_row(len(specs) + 1), column=0, columnspan=COLS, sticky="w", padx=2, pady=(4, 2))
    app.wsec = {}
    for key, text, kw in (("sections", "sections", dict(from_=3, to=401, width=5)),
                          ("changes", "change points", dict(from_=0, to=WM.MAX_CHANGES, width=4))):
        ttk.Label(row, text=text).pack(side="left", padx=(4, 2))
        sb = ttk.Spinbox(row, command=app.on_wing_sections, **kw)
        sb.pack(side="left")
        sb.bind("<Return>", app.on_wing_sections)
        sb.bind("<FocusOut>", app.on_wing_sections)
        app.wsec[key] = sb
    app.wsec["sections"].set(ad.settings["sections"])
    app.wsec["changes"].set(ad.design.changes)
    app.sec_info = ttk.Label(box, foreground=INK2, wraplength=560, justify="left")
    app.sec_info.grid(row=grid_row(len(specs) + 2), column=0, columnspan=COLS, sticky="w", padx=4)

    box = ttk.LabelFrame(app.panel, text="Distributions and section changes (pick the curve to edit)")
    box.pack(fill="x", padx=6, pady=4)
    grid = ttk.Frame(box)
    grid.pack(fill="x", padx=4, pady=2)
    for i, c in enumerate(WM.CURVES):
        app._pick(grid, i // 3, i % 3, c, f"{c}  {WM.UNITS[c]}")
    k = ad.design.changes
    if k:
        ttk.Label(grid, text="changes along the chord:", foreground=INK2).grid(row=2, column=0, columnspan=3,
                                                                              sticky="w", pady=(4, 0))
        for i in range(1, k + 1):
            app._pick(grid, 2 + i, 0, f"dt{i}", f"dt{i}  thickness")
            app._pick(grid, 2 + i, 1, f"dc{i}", f"dc{i}  camber")


PANELS = {"rudder": rudder_panel, "blade": blade_panel, "hull": hull_panel, "wing": wing_panel, "fin": wing_panel}


def main():
    default = "vehicle" if available("hull") else "rudder"
    ap = argparse.ArgumentParser(description="XGeom design tool")
    ap.add_argument("geometry", nargs="?", default=default, choices=sorted(ADAPTERS) + ["vehicle"],
                    help=f"what to design (default here: {default})")
    ap.add_argument("--file", dest="params", action="append", default=[],
                    help="a component file to start from: the blade's, the hull's, the wing's or the fin's, a "
                         "rudder's set-up modified below H or a planar rudder's (the vehicle: any of them)")
    args = ap.parse_args()
    if args.geometry == "vehicle" and not available("hull"):
        ap.error("the vehicle needs xgeom_hull.py (the X_geom folder)")
    if args.geometry != "vehicle":
        if not available(args.geometry):
            ap.error(f"{ADAPTERS[args.geometry][0]}.py is not in this folder")
        if len(args.params) > 1:
            ap.error("one --file for one geometry")
    params = [os.path.abspath(p) for p in args.params]
    if args.geometry != "vehicle":
        params = (params or [None])[0]
    os.chdir(HERE)                                   # the adapters' files are relative to this folder
    root = tk.Tk()
    root.geometry(f"{min(1600, root.winfo_screenwidth() - 40)}x{min(940, root.winfo_screenheight() - 90)}+10+10")
    App(root, args.geometry, params)
    root.mainloop()


if __name__ == "__main__":
    main()
