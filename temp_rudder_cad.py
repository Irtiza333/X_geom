"""
temp_rudder_cad.py

Temporary: the rudder's CAD solid from a file of modified sections, with the
original sections above them filled in.

    python temp_rudder_cad.py modified_m.dat
    python temp_rudder_cad.py modified_m.dat outputs/sections_xcad_with_cap_y0_y200_m.dat

Both files are XCAD point files like sections_xcad_y0_y80_m.dat: '#k' before
each section, then "x y z" per line in metres (z the height), every section a
closed loop of the same number of points starting at the LE. The original file
defaults to outputs/sections_xcad_with_cap_y0_y200_m.dat. Its sections above the
top modified section are added, everything is lofted into one solid, and the
solid is written as <modified file>.step (mm).

Needs numpy, scipy and pythonocc-core.
"""

import os
import sys

import numpy as np
from scipy.interpolate import BSpline

from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeSolid,
                                     BRepBuilderAPI_MakeWire, BRepBuilderAPI_Sewing)
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCC.Core.Geom import Geom_BSplineCurve
from OCC.Core.GProp import GProp_GProps
from OCC.Core.gp import gp_Pnt
from OCC.Core.ShapeFix import ShapeFix_Shell, ShapeFix_Solid
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal
from OCC.Core.TopAbs import TopAbs_SHELL
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods

ORIGINAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "sections_xcad_with_cap_y0_y200_m.dat")
N_POLES, DEGREE = 40, 3


def read_sections(path):
    """The sections of an XCAD point file, each an (n, 3) array in mm."""
    sections = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("#"):
                sections.append([])
            elif line:
                sections[-1].append([float(v) for v in line.split()])
    return [np.array(s) * 1000.0 for s in sections]


def bspline_edge(points):
    """A cubic B-spline least-squares fitted to the points, end points held,
    N_POLES poles on uniform knots: every section's curves share knots, so the
    loft needs no knot merging."""
    knots = np.r_[[0.0] * DEGREE, np.linspace(0.0, 1.0, N_POLES - DEGREE + 1), [1.0] * DEGREE]
    a = BSpline.design_matrix(np.linspace(0.0, 1.0, len(points)), knots, DEGREE).toarray()
    poles = np.empty((N_POLES, 3))
    poles[0], poles[-1] = points[0], points[-1]
    rhs = points - np.outer(a[:, 0], poles[0]) - np.outer(a[:, -1], poles[-1])
    poles[1:-1] = np.linalg.lstsq(a[:, 1:-1], rhs, rcond=None)[0]
    arr = TColgp_Array1OfPnt(1, N_POLES)
    for i, (x, y, z) in enumerate(poles):
        arr.SetValue(i + 1, gp_Pnt(float(x), float(y), float(z)))
    unique = np.linspace(0.0, 1.0, N_POLES - DEGREE + 1)
    k_arr, m_arr = TColStd_Array1OfReal(1, len(unique)), TColStd_Array1OfInteger(1, len(unique))
    for i, kv in enumerate(unique):
        k_arr.SetValue(i + 1, float(kv))
        m_arr.SetValue(i + 1, DEGREE + 1 if i in (0, len(unique) - 1) else 1)
    return BRepBuilderAPI_MakeEdge(Geom_BSplineCurve(arr, k_arr, m_arr, DEGREE)).Edge()


def wire(*edges):
    w = BRepBuilderAPI_MakeWire()
    for e in edges:
        w.Add(e)
    return w.Wire()


def loft(wires, ruled):
    ts = BRepOffsetAPI_ThruSections(False, ruled, 1.0e-6)
    ts.CheckCompatibility(False)
    for w in wires:
        ts.AddWire(w)
    ts.Build()
    return ts.Shape()


def build_solid(sections):
    """Horizontal sections lofted smoothly, one ruled step onto the tilted tip
    section, the tip section and the cap loops lofted smoothly; the root closed
    by a plane, the last loop by a plane (if horizontal) or the ruled face
    across it; sewn and made a solid."""
    i_te = (len(sections[0]) + 1) // 2 - 1                    # the LE is first, the TE at i_te
    wires = [wire(bspline_edge(s[:i_te + 1]), bspline_edge(s[i_te:])) for s in sections]
    flat = [np.ptp(s[:, 2]) < 1e-6 for s in sections]
    k = flat.index(False) if False in flat else len(flat)    # the first section that is not horizontal
    faces = [loft(wires[:k], False)]
    if k < len(wires):
        faces.append(loft(wires[k - 1:k + 1], True))
        if len(wires) - k > 1:
            faces.append(loft(wires[k:], False))
    faces.append(BRepBuilderAPI_MakeFace(wires[0], True).Face())
    top = sections[-1]
    if flat[-1]:
        faces.append(BRepBuilderAPI_MakeFace(wires[-1], True).Face())
    else:
        faces.append(loft([wire(bspline_edge(top[:i_te + 1])), wire(bspline_edge(top[i_te:][::-1]))], True))
    sew = BRepBuilderAPI_Sewing(1.0e-3)
    for f in faces:
        sew.Add(f)
    sew.Perform()
    shell = ShapeFix_Shell(topods.Shell(TopExp_Explorer(sew.SewedShape(), TopAbs_SHELL).Current()))
    shell.Perform()
    solid = ShapeFix_Solid(BRepBuilderAPI_MakeSolid(shell.Shell()).Solid())
    solid.Perform()
    return solid.Solid()


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    modified = read_sections(sys.argv[1])
    original = read_sections(sys.argv[2] if len(sys.argv) > 2 else ORIGINAL)
    top = max(s[:, 2].max() for s in modified)
    above = [s for s in original if s[:, 2].min() > top + 1e-6]
    sections = modified + above
    if len({len(s) for s in sections}) != 1:
        sys.exit("every section needs the same number of points")
    solid = build_solid(sections)
    props = GProp_GProps()
    brepgprop.VolumeProperties(solid, props, 1.0e-7)               # adaptive: the default rule is 0.2 % off here
    print(f"{len(modified)} modified sections up to {top:.4f} mm + {len(above)} original sections above; solid "
          f"{'valid' if BRepCheck_Analyzer(solid).IsValid() else 'NOT valid'}, volume {props.Mass():.0f} mm^3")
    out = os.path.splitext(sys.argv[1])[0] + ".step"
    writer = STEPControl_Writer()
    writer.Transfer(solid, STEPControl_AsIs)
    writer.Write(out)
    print(f"written {out}")


if __name__ == "__main__":
    main()
