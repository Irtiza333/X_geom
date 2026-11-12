"""
temp_rudder_cad.py

Temporary: the rudder's CAD solid from a file of modified sections, with the
original sections above them filled in.

    python temp_rudder_cad.py modified_m.dat
    python temp_rudder_cad.py modified_m.dat outputs/sections_xcad_with_cap_y0_y200_m.dat

Both files are XCAD point files like sections_xcad_y0_y80_m.dat: "x y z" per
line in metres (z the height), every section a loop starting at the LE. A
'#k' line before each section marks it off; a file without '#' lines is split
where the height column changes instead (each such section lies at one
height). The sections need not share a number of points: one whose count
differs from the original file's is resampled by cosine arc into its layout
(the first point, the LE, kept first; the point farthest from it, the TE, at
the middle index; the loop closed back on the LE). The original file defaults
to outputs/sections_xcad_with_cap_y0_y200_m.dat. Its sections above the top
modified section are added, everything is lofted into one solid with
xcad_loft's five patches (user, 11 Nov 2026: the four along the span run up
to the tip's edge, the tip patch covers the tip face alone): bicubic patches
fitted by least squares, the LE and TE strips wrapping their edges (the TE
strip narrowed to the TE's rounding), the central pressure and suction
panels, the tip patch over the cap, and the flat root; six faces. Where the
sections step from one to the next (sections written from the Bezier model
alone under the extracted ones, as before 12 Nov 2026), the loft eases the
step out of the sections below and above it, so the solid crosses it without
a ripple; the step is printed. The solid is written as <modified file>.step
(mm).

Needs numpy, scipy, pythonocc-core and xcad_loft.py beside it.
"""

import os
import sys

import numpy as np

import xcad_loft as XL

ORIGINAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "sections_xcad_with_cap_y0_y200_m.dat")
Z_TOL = 1.0e-9                                             # m: a larger height change starts a new section


def read_sections(path):
    """The sections of an XCAD point file, each an (n, 3) array in mm: split at the '#' lines, or, in a
    file without any, where the height column changes."""
    rows, marks = [], []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("#"):
                marks.append(len(rows))
            elif s:
                rows.append([float(v) for v in s.split()])
    rows = np.array(rows)
    if marks:
        cuts = sorted({0, len(rows)} | set(marks))
    else:
        cuts = [0] + list(np.flatnonzero(np.abs(np.diff(rows[:, 2])) > Z_TOL) + 1) + [len(rows)]
    return [rows[a:b] * 1000.0 for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def cosine(n):
    return 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))


def resample(p, u):
    """Points of a polyline at the fractions u of its arc."""
    t = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    out = np.empty((len(u), p.shape[1]))
    for c in range(p.shape[1]):
        out[:, c] = np.interp(u * t[-1], t, p[:, c])
    return out


def to_layout(s, n_pts):
    """A section resampled into the layout of n_pts points: the first point (the LE) kept first, the point
    farthest from it (the TE) at the middle index, the sides on cosine arcs, closed back on the LE; a
    section already at n_pts points is left as it is."""
    if len(s) == n_pts:
        return s
    p = s[:-1] if np.linalg.norm(s[0] - s[-1]) < 1.0e-6 else s     # drop a closing duplicate (mm)
    i_te = int(np.argmax(np.linalg.norm(p - p[0], axis=1)))
    m = (n_pts + 1) // 2                                   # the TE lands at index m - 1, the loft's seam
    side1 = resample(p[:i_te + 1], cosine(m))
    side2 = resample(np.vstack((p[i_te:], p[:1])), cosine(n_pts - m + 1))
    return np.vstack((side1, side2[1:]))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    modified = read_sections(sys.argv[1])
    original = read_sections(sys.argv[2] if len(sys.argv) > 2 else ORIGINAL)
    counts = [len(s) for s in original]
    n_pts = max(set(counts), key=counts.count)             # the original file's layout
    top = max(s[:, 2].max() for s in modified)
    above = [s for s in original if s[:, 2].min() > top + 1e-6]
    sections = [to_layout(s, n_pts) for s in modified + above]
    resampled = sum(s is not t for s, t in zip(modified + above, sections))
    res = XL.loft_loops(sections)
    print(f"{len(modified)} modified sections up to {top:.4f} mm + {len(above)} original sections above"
          f"{f', {resampled} resampled to {n_pts} points' if resampled else ''}; "
          f"{res.faces} faces, solid {'valid' if res.valid else 'NOT valid'}"
          f"{f' ({res.free_edges} free edges)' if res.free_edges else ''}, volume {res.volume:.0f} mm^3, "
          f"fit {res.fit_error:.3f} mm")
    for note in res.notes:
        print(f"  {note}")
    out = os.path.splitext(sys.argv[1])[0] + ".step"
    XL.write_step(res.shape, out)
    print(f"written {out}")


if __name__ == "__main__":
    main()
