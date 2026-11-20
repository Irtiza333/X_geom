"""
temp_rudder_cad.py

Temporary: the rudder's CAD solid from a file of modified sections, with the
original sections above them filled in, lofted as two parts: the modified one
(the root up to the top modified section, H) and the unmodified one above H.

    python temp_rudder_cad.py modified_m.dat
    python temp_rudder_cad.py modified_m.dat outputs/sections_xcad_with_cap_y0_y200_m.dat

Both files are XCAD point files like sections_xcad_y0_y80_m.dat: "x y z" per
line in metres (z the height), every section a loop starting at the LE. A '#k'
line before each section marks it off; a file without '#' lines is split where
the height column changes instead (each such section lies at one height). The
modified file holds horizontal sections from the root up to H. The original
file defaults to outputs/sections_xcad_with_cap_y0_y200_m.dat; its loops above
H are added: the horizontal sections more than half its section spacing above
H (none then sits a hair's breadth over the section at H), the tilted tip
section and the cap.

Two lofts, always (the user, 20 Nov 2026; xcad_loft.loft_split at the top
modified section). The lower loft runs through the modified sections, the
upper one through the section at H and the original's loops above it. Each is
lofted on its own with xcad_loft's five patches (the blade's layout: the LE
and TE strips wrapping their edges, the central pressure and suction panels,
and on the upper loft the tip patch, its corners on the strip lines under the
tip), both through the section at H exactly and with the same strips (the LE
strip as wide as the LE's rounding, the TE strip as wide as the tip patch's TE
edge needs to be as long as its LE edge) and the same knots across them, so
they meet on the same four curves at H and the seams run on across it. The
surface may turn at H as the modified sections do (a modification whose curves
leave the original there without its slope): nothing of the modified part is
smoothed into the unmodified one, nor the other way. Sewn with the flat root
into one solid: nine patches and the root, ten faces. A step from the section
at H to the original's sections above it (sections from another source) is
eased out of the original's sections over the band above H (xcad_loft's
_ease_flats); the step is printed.

Any number of points. The loft takes loops of one layout (the same number of
points, the LE first, the TE at index (n + 1) // 2 - 1, closed back on the LE)
and fits its walls point index by point index along the span, so its loops
have to correspond point by point. Every modified section, whatever its number
of points, is therefore put onto the original's layout at its height (Layout):
its smooth curve (the cubic spline through its points that the loft samples,
xcad_loft._SmoothLoop) at the arc fractions of each side that the original's
points have there (the original's sections interpolated by height), the LE
kept first and the TE, the point farthest from the LE, at the TE's index. A
section given the other way round (the other side first) is turned round, an
open loop closed (repeated points dropped), and one that is already in the
original's layout at its height comes back unchanged. Until 20 Nov 2026 a
section of another number of points was spaced on cosine arcs, which is not
the original's layout (cosine in x): the loft read that change of layout at H
as a step of 3.6 mm and eased it over 180 mm of the span (the wind-tunnel
rudder's own sections given at 121 points: the solid up to 0.12 mm off them;
now 0.017 mm, as closely as 121 points describe the TE's round).

The solid is written as <modified file>.step (mm).

Needs numpy, scipy, pythonocc-core (or cadquery-ocp) and xcad_loft.py beside it.
"""

import os
import sys

import numpy as np

import xcad_loft as XL

ORIGINAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "sections_xcad_with_cap_y0_y200_m.dat")
Z_TOL = 1.0e-9                                             # m: a larger height change starts a new section
FLAT_TOL = 1.0e-6                                          # a loop within this share of its size (at least 1 mm)
                                                           # of one height is horizontal (as xcad_loft takes it)
SAME_TOL = 1.0e-9                                          # a section whose points sit within this share of a
                                                           # side's arc of the layout's is in it already


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
    return [closed(rows[a:b] * 1000.0) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def closed(s):
    """The loop without repeated points, closed back on its first point, the LE."""
    keep = np.r_[True, np.linalg.norm(np.diff(s, axis=0), axis=1) > 1.0e-9]
    s = s[keep]
    return s if np.linalg.norm(s[0] - s[-1]) <= 1.0e-9 else np.vstack((s, s[:1]))


def is_flat(s):
    return float(np.ptp(s[:, 2])) <= FLAT_TOL * max(1.0, float(np.abs(s).max()))


def signed_area(s):
    """The loop's signed area in the section's plane (x, y): its sense of turning."""
    x, y = s[:, 0], s[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def te_index(p):
    """The index of a closed loop's TE: its point farthest from the LE (the first point)."""
    return int(np.argmax(np.linalg.norm(p[:-1] - p[0], axis=1)))


def side_fractions(s, i_te):
    """The arc fractions of a closed loop's points (LE first, the TE at i_te) along its smooth curve: (side
    A from the LE to the TE, side B from the TE back to the LE), each running from 0 to 1."""
    sl = XL._SmoothLoop(s, i_te)
    a = sl.pt_arc[:i_te + 1] / sl.s_te
    b = (sl.pt_arc[i_te:] - sl.s_te) / (sl.S - sl.s_te)
    return a, b, sl


class Layout:
    """The original's point layout over the height: the arc fractions of each side of its horizontal
    sections of n_pts points, interpolated by height (clamped at the lowest and the highest), and its sense
    of turning."""

    def __init__(self, flats, n_pts):
        self.n_pts, self.i_te = n_pts, (n_pts + 1) // 2 - 1
        self.z = np.array([s[0, 2] for s in flats])
        fr = [side_fractions(s, self.i_te)[:2] for s in flats]
        self.fa, self.fb = np.array([f[0] for f in fr]), np.array([f[1] for f in fr])
        self.sense = np.sign(np.median([signed_area(s) for s in flats]))

    def at(self, z):
        """The arc fractions (side A, side B) at the height z."""
        if z <= self.z[0]:
            return self.fa[0], self.fb[0]
        if z >= self.z[-1]:
            return self.fa[-1], self.fb[-1]
        k = int(np.searchsorted(self.z, z)) - 1
        w = (z - self.z[k]) / (self.z[k + 1] - self.z[k])
        return (1.0 - w) * self.fa[k] + w * self.fa[k + 1], (1.0 - w) * self.fb[k] + w * self.fb[k + 1]

    def onto(self, s):
        """The section on this layout at its height: (loop, how: 'kept', 'resampled' or 'turned round')."""
        z = float(np.mean(s[:, 2]))
        p = closed(s)
        i_te = te_index(p)
        turned = np.sign(signed_area(p)) != self.sense
        if turned:                                         # the other side first: the LE stays first
            p = np.vstack((p[:1], p[1:-1][::-1], p[:1]))
            i_te = len(p) - 1 - i_te
        fa, fb = self.at(z)
        if len(p) == self.n_pts and i_te == self.i_te:
            a, b, _ = side_fractions(p, i_te)
            if max(np.abs(a - fa).max(), np.abs(b - fb).max()) <= SAME_TOL:
                return p, "turned round" if turned else "kept"
        sl = XL._SmoothLoop(p, i_te)
        out = np.vstack((sl.at(fa * sl.s_te), sl.at(sl.s_te + fb * (sl.S - sl.s_te))[1:]))
        out[-1] = out[0]
        if is_flat(p):
            out[:, 2] = z
        return out, "turned round" if turned else "resampled"


def build(modified_path, original_path=ORIGINAL):
    """(the loft result, a line saying what went in): the modified sections and the original's loops above
    them, the modified ones on the original's layout, lofted as two parts joined at the top modified
    section (xcad_loft.loft_split)."""
    modified = sorted(read_sections(modified_path), key=lambda s: float(np.mean(s[:, 2])))
    original = read_sections(original_path)
    bent = [k + 1 for k, s in enumerate(modified) if not is_flat(s)]
    if bent:
        raise ValueError(f"{modified_path}: {len(bent)} of its {len(modified)} sections are not horizontal (the "
                         f"first: section {bent[0]}, counted up the height); the modified file holds the horizontal "
                         f"sections from the root to H, the original file what lies above H")
    z = np.array([s[0, 2] for s in modified])
    if len(z) < 2 or np.any(np.diff(z) <= FLAT_TOL):
        raise ValueError(f"{modified_path}: at least two modified sections, each at a height of its own")
    counts = [len(s) for s in original]
    n_pts = max(set(counts), key=counts.count)             # the original file's layout
    flats = [s for s in original if is_flat(s) and len(s) == n_pts]
    layout = Layout(flats, n_pts)
    spacing = float(np.median(np.diff(layout.z)))
    top = float(z[-1])
    above = [s for s in original                           # the horizontal sections from half a step above H,
             if (s[0, 2] > top + 0.5 * spacing if is_flat(s) else s[:, 2].min() >= top - FLAT_TOL)]   # the rest
    if not above:
        raise ValueError(f"nothing of the original lies above the top modified section (z {top:.4f} mm): the "
                         f"unmodified loft needs the original's loops above H")
    mods = [layout.onto(s) for s in modified]
    ups = [(s, "kept") if len(s) == n_pts else layout.onto(s) for s in above]
    how = [h for _, h in mods]
    res = XL.loft_split([s for s, _ in mods] + [s for s, _ in ups], len(mods) - 1)
    said = ", ".join(f"{how.count(h)} {h}" for h in ("resampled", "turned round") if how.count(h))
    line = (f"{len(modified)} modified sections up to {top:.4f} mm"
            + (f" ({said} onto the original's layout of {n_pts} points)" if said else "")
            + f" + {len(above)} original loops above")
    return res, line


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    try:
        res, line = build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ORIGINAL)
    except ValueError as exc:
        sys.exit(f"temp_rudder_cad: {exc}")
    print(f"{line}; two lofts joined at the top modified section: {res.faces} faces, solid "
          f"{'valid' if res.valid else 'NOT valid'}{f' ({res.free_edges} free edges)' if res.free_edges else ''}, "
          f"volume {res.volume:.0f} mm^3, fit {res.fit_error:.3f} mm")
    for note in res.notes:
        print(f"  {note}")
    out = os.path.splitext(sys.argv[1])[0] + ".step"
    XL.write_step(res.shape, out)
    print(f"written {out}")


if __name__ == "__main__":
    main()
