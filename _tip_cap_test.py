"""
_tip_cap_test.py

Self-check for tip_cap.py, the rudder tip-cap model: the cross-section is
continuous and smooth where its pieces meet, the special cases come out right,
the loops meet the tip section where they should, and a fit to a cap built from
known parameters gives those parameters back.

Needs only numpy and scipy (and rot_axis.py / skew_symmetric_matrix.py are not
needed here): the tip section is an analytic NACA 0016.

    python _tip_cap_test.py
"""

from __future__ import annotations

import numpy as np

import tip_cap as T

_FAILED = []


def check(name, got, want, tol):
    got, want = np.asarray(got, float), np.asarray(want, float)
    err = float(np.abs(got - want).max())
    ok = err <= tol
    if not ok:
        _FAILED.append(name)
    print(f"  {'ok ' if ok else 'FAIL'}  {name:<52s} max error {err:.3e}   (tol {tol:.0e})")


def naca_base(chord=136.95, t=0.16, n=100, tilt_deg=3.58):
    """A NACA 00xx tip section, tilted like the rudder's, as a CapBase."""
    x = chord * 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
    xc = x / chord
    yt = 5.0 * t * chord * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc ** 2
                            + 0.2843 * xc ** 3 - 0.1036 * xc ** 4)
    th = np.radians(tilt_deg)
    return T.CapBase(x, yt, x, -yt, pivot=[14.0, 200.0, 0.0],
                     chord_dir=[np.cos(th), np.sin(th), 0.0],
                     normal=[-np.sin(th), np.cos(th), 0.0], thick_dir=[0.0, 0.0, 1.0])


RUDDER = T.CapParams(h_le=0.7, h_te=3.0, b1=0.15, b2=-0.1, crown_radius=60.0,
                     edge_radius=0.71, te_radius=3.0)


def test_cross_section():
    print("\ncross-section: wall, edge radius, crown")
    base = naca_base()
    for x in (10.0, 40.0, 110.0):
        pr = T.profile(x, RUDDER, base, n=4000)
        s, h = pr[:, 0], pr[:, 1]
        w, H = base.half_thickness(x), T.cap_top(x, RUDDER, base)
        check(f"x={x:g}: starts on the wall base (s = w, h = 0)", [s[0], h[0]], [w, 0.0], 1e-9)
        check(f"x={x:g}: ends on the crest (s = 0, h = H)", [s[-1], h[-1]], [0.0, H], 1e-9)
        check(f"x={x:g}: s never grows going up", max(np.diff(s).max(), 0.0), 0.0, 1e-12)
        # no jumps: consecutive points close together, and the direction turns smoothly
        seg = np.hypot(np.diff(s), np.diff(h))
        check(f"x={x:g}: continuous (largest step)", min(seg.max(), 0.05), seg.max(), 0.0)
        ang = np.unwrap(np.arctan2(np.diff(h), np.diff(s)))
        check(f"x={x:g}: tangent continuous (largest turn per step)",
              min(np.abs(np.diff(ang)).max(), 0.05), np.abs(np.diff(ang)).max(), 0.0)


def test_special_cases():
    print("\nspecial cases")
    base = naca_base()
    x = 50.0
    w = base.half_thickness(x)
    square = T.CapParams(h_le=1.0, h_te=1.0)                  # flat face, sharp edge
    pr = T.profile(x, square, base, n=2000)
    check("square tip: wall straight up to the face", pr[pr[:, 1] < 1.0 - 1e-9, 0], w, 1e-12)
    check("square tip: top is flat at the face height", T.cap_top(x, square, base), 1.0, 1e-12)
    rounded = T.CapParams(h_le=w, h_te=w, edge_radius=w)       # half-round at x = 50
    sec = T._section(w, float(T.cap_top(x, rounded, base)), rounded.edge_radius,
                     rounded.crown_radius)
    check("full round: edge radius clamps to the section", sec.r, 0.999 * w, 1e-9)
    check("ends: no cap height at the top of the LE or TE",
          [T.cap_top(0.0, RUDDER, base), T.cap_top(base.chord, RUDDER, base)], [0.0, 0.0], 0.0)


def test_loops():
    print("\nloops for a section-based rebuild")
    base = naca_base()
    L = T.loops(RUDDER, base, [0.0, 0.5, 1.0])
    n = len(base.x_up) + len(base.x_lo) - 1
    check("same point count as the tip section", [len(l) for l in L], n, 0)
    le = base.to_3d(0.0, 0.0, 0.0)
    te = base.to_3d(base.chord, base.upper(base.chord), 0.0)
    for v, l in zip((0.0, 0.5, 1.0), L):
        check(f"v={v:g}: starts and ends at the top of the LE", [l[0], l[-1]], [le, le], 1e-9)
        check(f"v={v:g}: TE point is the top of the TE", l[len(base.x_up) - 1], te, 1e-9)
    tip = np.vstack([base.to_3d(base.x_up, base.y_up, 0.0),
                     base.to_3d(base.x_lo[::-1][1:], base.y_lo[::-1][1:], 0.0)])
    check("v=0 is the tip section itself", L[0], tip, 1e-9)
    x, z, h = base.to_cap(L[2])
    check("v=1 closes onto the crest (upper = lower)",
          L[2][:len(base.x_up)], L[2][len(base.x_up) - 1:][::-1], 1e-9)


def test_fit_recovers_parameters():
    print("\nfit a cap built from known parameters")
    base = naca_base()
    stations = []
    for u in np.r_[np.linspace(0.02, 0.95, 24), 0.965, 0.975, 0.985, 0.992, 0.997]:
        x = u * base.chord
        pr = T.profile(x, RUDDER, base, n=600)
        stations.append(T.CapStation(x, pr[:, 0], pr[:, 1]))
    p, rep = T.fit(stations, base)
    check("face height at the LE end", p.h_le, RUDDER.h_le, 2e-3)
    check("face height at the TE end", p.h_te, RUDDER.h_te, 2e-3)
    check("bow terms", [p.b1, p.b2], [RUDDER.b1, RUDDER.b2], 5e-3)
    check("crown radius", p.crown_radius, RUDDER.crown_radius, 0.5)
    check("edge radius", p.edge_radius, RUDDER.edge_radius, 2e-3)
    check("TE corner radius", p.te_radius, RUDDER.te_radius, 2e-2)
    check("worst point deviation after the fit", max(r["max_dev"] for r in rep), 0.0, 1e-3)


def main():
    print("tip cap model self-check")
    test_cross_section()
    test_special_cases()
    test_loops()
    test_fit_recovers_parameters()
    print()
    if _FAILED:
        print(f"{len(_FAILED)} check(s) FAILED: " + ", ".join(_FAILED))
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
