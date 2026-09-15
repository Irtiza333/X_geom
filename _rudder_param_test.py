"""
_rudder_param_test.py

Self-check for Step 6 of Rudder_geom_extraction.py: the spanwise
parametrisation.

The real rudder is the awkward case to validate against, because its sweep is
the only parameter that is not zero - a sign error in the rake or the twist
would read as zero either way. So this builds synthetic section stacks with
prescribed chord, sweep, rake and twist wrapped around an analytic NACA
4-digit section, runs them through the same `spanwise_distributions` the CAD
path uses, and checks the numbers that come back are the ones that went in,
signs included.

Needs only numpy: the stacks are assembled directly as `Section` records, so
no CAD and no OCCT binding are involved.

    python _rudder_param_test.py
"""

from __future__ import annotations

import numpy as np

from Rudder_geom_extraction import (
    CHORD_AXIS, SPAN_AXIS, THICK_AXIS,
    Section, spanwise_distributions,
)


# --------------------------------------------------------------------------- #
# Synthetic geometry
# --------------------------------------------------------------------------- #
def naca4(m: float, p: float, t: float, n: int = 600, perpendicular: bool = False):
    """
    Upper and lower surfaces of a NACA 4-digit section, each LE -> TE in the
    chord frame, chord 1. The last thickness coefficient is -0.1036 rather than
    -0.1015, which closes the trailing edge to a single point.

    `perpendicular` selects how the thickness is hung on the camber line:

      False - straight up and down at each station, so the extractor's mean
              line, 0.5 (y_upper + y_lower) at constant x, is exactly the
              generating camber line and m and p are exact ground truth.
      True  - the real NACA construction, normal to the camber line. Realistic,
              but then the mean line taken at constant x is no longer exactly
              the generating one, so it is no use as an exact reference.

    Both are genuine airfoil shapes; the difference is second order in t.
    """
    x = (1.0 - np.cos(np.linspace(0.0, np.pi, n))) / 2.0
    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                    + 0.2843 * x ** 3 - 0.1036 * x ** 4)

    if m > 0.0:
        fore = x < p
        yc = np.where(fore,
                      m / p ** 2 * (2.0 * p * x - x ** 2),
                      m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x - x ** 2))
        dyc = np.where(fore,
                       2.0 * m / p ** 2 * (p - x),
                       2.0 * m / (1.0 - p) ** 2 * (p - x))
    else:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)

    th = np.arctan(dyc) if perpendicular else np.zeros_like(x)
    upper = np.column_stack([x - yt * np.sin(th), yc + yt * np.cos(th)])
    lower = np.column_stack([x + yt * np.sin(th), yc - yt * np.cos(th)])
    return upper, lower


def make_section(y, chord, x_le, z_le, twist_deg, upper, lower) -> Section:
    """
    One synthetic station: the unit section placed at `y`, scaled to `chord`,
    its LE put at (x_le, y, z_le) and rotated `twist_deg` nose-up about that LE.

    Nose-up means the chord line runs aft and down, so the TE ends up on the -Z
    side of the LE. That is the convention the extractor is expected to report.
    """
    th = np.radians(twist_deg)
    ex = np.array([np.cos(th), -np.sin(th)])      # LE -> TE
    ey = np.array([np.sin(th), np.cos(th)])       # section "up"

    def place(arc2d):
        pts = np.empty((len(arc2d), 3))
        xz = np.array([x_le, z_le]) + chord * (np.outer(arc2d[:, 0], ex)
                                               + np.outer(arc2d[:, 1], ey))
        pts[:, CHORD_AXIS] = xz[:, 0]
        pts[:, THICK_AXIS] = xz[:, 1]
        pts[:, SPAN_AXIS] = y
        return pts

    up, lo = place(upper), place(lower)
    outline = np.vstack([up, lo[::-1][1:-1]])     # closed loop, no repeat point
    le, te = up[0], up[-1]

    dummy = np.zeros((2, 2))
    return Section(y=float(y), le=le, te=te,
                   chord=float(te[CHORD_AXIS] - le[CHORD_AXIS]),
                   outline=outline, airfoil_raw=dummy, airfoil_norm=dummy)


def build_stack(ys, chord, x_le, z_le, twist, upper, lower):
    """`chord`, `x_le`, `z_le` and `twist` are callables of the station y."""
    return [make_section(y, chord(y), x_le(y), z_le(y), twist(y), upper, lower)
            for y in ys]


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
_FAILED = []


def check(name, got, want, tol):
    got, want = np.asarray(got, float), np.asarray(want, float)
    err = float(np.abs(got - want).max())
    ok = err <= tol
    if not ok:
        _FAILED.append(name)
    print(f"  {'ok ' if ok else 'FAIL'}  {name:<44s} max error {err:.3e}"
          f"   (tol {tol:.0e})")


def test_straight_swept_raked_twisted():
    """
    Everything constant and non-zero at once: 25 deg of aft sweep, 8 deg of
    dihedral, a fixed 3 deg nose-up twist and a linear taper. This is the case
    that catches a sign flip, because none of the answers is zero.
    """
    print("\nstraight rudder, sweep 25 deg aft, dihedral 8 deg, twist 3 deg nose-up")
    span, sweep, dih, tw = 200.0, 25.0, 8.0, 3.0
    ys = np.linspace(0.0, 180.0, 40)
    upper, lower = naca4(0.0, 0.0, 0.16)

    stack = build_stack(
        ys,
        chord=lambda y: 180.0 - 0.25 * y,
        x_le=lambda y: y * np.tan(np.radians(sweep)),
        z_le=lambda y: y * np.tan(np.radians(dih)),
        twist=lambda y: tw,
        upper=upper, lower=lower,
    )
    d = spanwise_distributions(stack, span)

    check("chord", d.chord, 180.0 - 0.25 * ys, 1e-9)
    check("chord_x (projection, shorter by cos twist)",
          d.chord_x, (180.0 - 0.25 * ys) * np.cos(np.radians(tw)), 1e-9)
    check("sweep_le cumulative", d.sweep_le, sweep, 1e-9)
    check("sweep_le local", d.sweep_le_local, sweep, 1e-9)
    check("rake", d.rake, ys * np.tan(np.radians(dih)), 1e-9)
    check("dihedral cumulative", d.dihedral, dih, 1e-9)
    check("dihedral local", d.dihedral_local, dih, 1e-9)
    check("twist", d.twist, tw, 1e-9)
    check("eta", d.eta, ys / span, 1e-12)

    # the quarter-chord line of a tapered wing is less swept than its LE
    c4 = ys * np.tan(np.radians(sweep)) + 0.25 * np.cos(np.radians(tw)) * (
        (180.0 - 0.25 * ys) - 180.0)
    want_c4 = np.degrees(np.arctan(np.gradient(c4, ys)))
    check("sweep_c4 local", d.sweep_c4_local, want_c4, 1e-9)


def test_curved_leading_edge():
    """
    A parabolic leading edge, x_le = k y^2. The cumulative sweep then follows
    arctan(k y) and the local sweep arctan(2 k y): the point of carrying both
    is that they separate exactly here, and by a known amount.
    """
    print("\ncurved leading edge, x_le = k y^2")
    span, k = 200.0, 4.0e-3
    ys = np.linspace(0.0, 180.0, 121)
    upper, lower = naca4(0.0, 0.0, 0.12)

    stack = build_stack(ys, chord=lambda y: 150.0, x_le=lambda y: k * y ** 2,
                        z_le=lambda y: 0.0, twist=lambda y: 0.0,
                        upper=upper, lower=lower)
    d = spanwise_distributions(stack, span)

    check("sweep_le cumulative = arctan(k y)",
          d.sweep_le, np.degrees(np.arctan(k * ys)), 1e-9)
    # the root value is the limit of the secant, i.e. the tangent there
    check("sweep_le local = arctan(2 k y)",
          d.sweep_le_local[1:-1], np.degrees(np.arctan(2.0 * k * ys[1:-1])), 1e-3)
    check("sweep_le at the root is the tangent, not 0/0",
          d.sweep_le[0], d.sweep_le_local[0], 1e-12)
    check("twist stays zero on a swept-back edge", d.twist, 0.0, 1e-9)


def test_section_shape():
    """
    Thickness, camber and nose radius, on a section whose answers are known in
    closed form. Measuring them in the section's own chord frame is the point:
    a twisted section must report the same shape as an untwisted one, with the
    rotation showing up only in the twist.
    """
    print("\nsection shape, 4412-like section at 0 deg and at 12 deg nose-up twist")
    m, p, t = 0.04, 0.4, 0.12
    upper, lower = naca4(m, p, t)
    ys = np.linspace(0.0, 100.0, 12)

    twists = {}
    for tw in (0.0, 12.0):
        stack = build_stack(ys, chord=lambda y: 120.0, x_le=lambda y: 0.3 * y,
                            z_le=lambda y: 0.0, twist=lambda y: tw,
                            upper=upper, lower=lower)
        d = spanwise_distributions(stack, 120.0)
        twists[tw] = d.twist
        tag = f"(twist {tw:g} deg)"
        check(f"t/c {tag}", d.t_over_c, t, 1e-4)
        check(f"x/c of max thickness {tag}", d.x_tmax, 0.30, 2e-3)
        check(f"max camber / c {tag}", d.camber, m, 1e-6)
        # The two parabolas of a NACA camber line meet at x = p with matching
        # slope but a jump in curvature, right where the peak is. That is the
        # worst case there is for a sub-grid parabolic peak fit, and it still
        # lands inside a fifth of the sampling interval.
        check(f"x/c of max camber {tag}", d.x_cmax, p, 2e-3)
        # NACA 4-digit nose radius is 1.1019 t^2
        check(f"LE radius / c {tag}", d.r_le, 1.1019 * t ** 2, 5e-5)
        check(f"twist is uniform up the span {tag}",
              d.twist - d.twist[0], 0.0, 1e-9)

    # A cambered section's geometric nose sits a little off its construction
    # axis, so the absolute twist carries a fixed offset - here 0.19 deg. It is
    # the same at every station and in both stacks, so the twist the rudder was
    # actually built with comes back exactly as a difference.
    check("twist difference between the two stacks",
          twists[12.0] - twists[0.0], 12.0, 1e-9)
    check("absolute twist offset is the section's, not the stack's",
          np.ptp(twists[0.0]), 0.0, 1e-9)


def test_symmetric_section_has_no_camber_position():
    """A symmetric section has no camber peak to locate; it must say so."""
    print("\nsymmetric section")
    upper, lower = naca4(0.0, 0.0, 0.16)
    stack = build_stack(np.linspace(0.0, 100.0, 8), chord=lambda y: 100.0,
                        x_le=lambda y: 0.0, z_le=lambda y: 0.0,
                        twist=lambda y: 0.0, upper=upper, lower=lower)
    d = spanwise_distributions(stack, 100.0)
    check("camber is zero", d.camber, 0.0, 1e-9)
    check("x/c of max camber is NaN", np.isfinite(d.x_cmax).sum(), 0, 0)


def main():
    print("Step 6 parametrisation self-check")
    test_straight_swept_raked_twisted()
    test_curved_leading_edge()
    test_section_shape()
    test_symmetric_section_has_no_camber_position()

    print()
    if _FAILED:
        print(f"{len(_FAILED)} check(s) FAILED: " + ", ".join(_FAILED))
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
