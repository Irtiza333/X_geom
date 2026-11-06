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
top modified section are added, everything is lofted into one solid with the
mesh patch topology of xcad_loft (user, 5 Nov 2026): bicubic patches fitted by
least squares, the side up to the collar (10 mm below the top horizontal
section), the tip band, the cap rim to the line where its edge rounding meets
the deck, the crown to the ridge, seams at the LE and the TE, nine faces. The
solid is written as <modified file>.step (mm).

Needs numpy, scipy, pythonocc-core and xcad_loft.py beside it.
"""

import os
import sys

import xcad_loft as XL

ORIGINAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "sections_xcad_with_cap_y0_y200_m.dat")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    modified = XL.read_loops(sys.argv[1])
    original = XL.read_loops(sys.argv[2] if len(sys.argv) > 2 else ORIGINAL)
    top = max(s[:, 2].max() for s in modified)
    above = [s for s in original if s[:, 2].min() > top + 1e-6]
    sections = modified + above
    if len({len(s) for s in sections}) != 1:
        sys.exit("every section needs the same number of points")
    res = XL.loft_loops(sections)
    print(f"{len(modified)} modified sections up to {top:.4f} mm + {len(above)} original sections above; "
          f"{res.faces} faces, solid {'valid' if res.valid else 'NOT valid'}"
          f"{f' ({res.free_edges} free edges)' if res.free_edges else ''}, volume {res.volume:.0f} mm^3, "
          f"fit {res.fit_error:.3f} mm")
    out = os.path.splitext(sys.argv[1])[0] + ".step"
    XL.write_step(res.shape, out)
    print(f"written {out}")


if __name__ == "__main__":
    main()
