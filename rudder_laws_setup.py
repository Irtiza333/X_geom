"""
rudder_laws_setup.py

A set-up of the rudder modified below H for building on the original: the
original keeps its place and mass is only added. It is the rule of the
planar-rudder optimisation on the wind-tunnel rudder (the folder
Planar_Rudder_Optimization), written with bounding laws
(rudder_modify.Design.laws) instead of code of its own:

    python rudder_laws_setup.py                    writes outputs/modified/add_to_original.txt
    python rudder_laws_setup.py --a 2 --b 2        the thickness up to 2 times the original's (default 10, 10)
    python xgeom_tool.py rudder --file outputs/modified/add_to_original.txt

The set-up is a component file of kind rudder_h (xgeom_files.py), as every
other component's: rudder_modify.load_component reads it for an optimiser.

The sections are the quartics P0 .. P4 of the section fits; all lengths below
are mm, the laws turn them into fractions of the section's own sharp chord:

LE    a cubic from the original LE at H to the root, the root up to L_MAX
      forward (LE.dx3); the two inner control points between the original LE
      at their heights and the root's x (dependent bounds), their heights in
      order, 0.05 H from the ends.
P1    on the LE line (P1x held at 0), as high as P2 (P1z's law: P2z).
P2    its height from the original's to `a` times it; its x from the original's
      forward to the LE.
P3    its height from the original's to `b` times it; its x between the two
      lines through the original P3 that open as P3 rises: from the original
      P2 through the original P3 (not behind the original's sharp TE), and
      from that TE through the original P3 (not ahead of the original P2). At
      the original's height the two meet: P3 stays where it is until it is
      raised.

Each law is a cubic over the height, pinned on the original at H: 26 free
variables (H; 5 for the LE; 5 for each of P2x, P2z, P3x, P3z). Every low law is
the original's side (for the two x the low law is the larger one), so every
fraction is 0 on the original and the original itself is all fractions 0: a
thickness fraction 0 is the original's thickness in mm wherever the LE goes.

These laws act on control points: they make most designs enclose the original,
not all (the LE moved forward with nothing raised lies 0.3 mm inside it near
mid chord). The set-up therefore keeps the original inside by measure
(keep_inside, 0.01 mm): the tool warns, and an optimiser takes
rudder_modify.inside_original(design, ...)["depth"] as a constraint.
"""

import argparse
import os

import numpy as np

import xgeom_rudder as XR

L_MAX = 160.0          # mm, how far forward of the original the root LE may go
H_RANGE = (1.0, 80.0)  # mm, the height of the modified region
ORDER = 3              # every curve a cubic: 4 control points
X2, TE = "orig.LE+orig.P2x*orig.chord", "orig.LE+orig.chord"       # x of the original P2 and of its sharp TE, mm


def laws(a=10.0, b=10.0):
    """{curve: (low law, high law)} of the rule; a and b: how many times the
    original's height P2 and P3 may reach."""
    y2, y3 = "orig.P2z*orig.chord", "orig.P3z*orig.chord"                  # their heights, mm
    return {
        "P2x": (f"({X2}-LE)/chord", "0"),
        "P2z": (f"{y2}/chord", f"{a:g}*{y2}/chord"),
        "P1z": ("P2z", "P2z"),
        "P3z": (f"{y3}/chord", f"{b:g}*{y3}/chord"),
        "P3x": (f"min({TE}-LE, {X2}-LE+(orig.P3x-orig.P2x)*orig.chord*(P3z*chord-{y2})/({y3}-{y2}))/chord",
                f"max({X2}-LE, {TE}-LE+(orig.P3x-1)*orig.chord*P3z*chord/({y3}))/chord")}


def set_up(ad, a=10.0, b=10.0, l_max=L_MAX, h_range=H_RANGE):
    """Put the rule on a rudder adapter (xgeom_rudder.RudderAdapter) and return it."""
    for c in ad.names:
        ad.set_order(c, ORDER)
    ad.set_tangent_all(False)
    ad.set_bounds("H", *h_range)
    ad.set("H", h_range[1])
    orig = ad.orig
    y = np.linspace(0.0, orig.z_top, 201)
    le = orig.value("LE", y)
    k = float((le[-1] - le[0]) / orig.z_top)                # the original LE: straight, x = x0 + k y
    if np.abs(le - le[0] - k * y).max() > 0.05:
        raise ValueError("the original LE is not straight: the LE's dependent bounds here take it as a line")
    ad.set_bounds("LE.dx3", -float(l_max), 0.0)
    ad.set_bounds("LE.d1", 0.05, 0.95)
    ad.set_bounds("LE.d2", 0.0, "1-0.05/(1-d1)")
    ad.set_bounds("LE.dx1", f"dx3-{k:.6g}*(1-d1)*H", 0.0)
    ad.set_bounds("LE.dx2", f"dx3-{k:.6g}*(1-d1)*(1-d2)*H", 0.0)
    for c, (low, high) in laws(a, b).items():
        ad.set_laws(c, low, high)
    ad.set_keep_inside(True, XR.INSIDE_TOL)
    return ad


def main():
    ap = argparse.ArgumentParser(description="the build-on-the-original rule as a rudder set-up with bounding laws")
    ap.add_argument("--a", type=float, default=10.0, help="P2's height up to this many times the original's")
    ap.add_argument("--b", type=float, default=10.0, help="P3's height up to this many times the original's")
    ap.add_argument("--out", default=None, help="the set-up file (default: add_to_original.txt in the rudder's "
                                               "output folder)")
    args = ap.parse_args()
    ad = set_up(XR.RudderAdapter(), args.a, args.b)
    path = ad.save_file(args.out or os.path.join(ad.out_dir, "add_to_original.txt"))
    pv = ad.preview()
    print(f"{path}: {ad.summary()}")
    for c in ad.names:
        if ad.laws_of(c):
            print(f"  {c:4s} {ad.laws_of(c)[0]}  ..  {ad.laws_of(c)[1]}")
    print("  problems:", pv["problems"] or "none", "| warnings:", pv["warnings"] or "none",
          f"| inside the original: {pv['inside']['depth']:.4f} mm")


if __name__ == "__main__":
    main()
