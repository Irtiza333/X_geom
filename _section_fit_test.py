"""
_section_fit_test.py

Self-check for the rudder section fit: Main_PSO (with bound_check, penalty and
repair), the Bezier replica (replica_funcs), the TE fillet (retruncate), the
sharpening (temp_funcs) and the whole match on section_cut_y80_selig.dat.

    python _section_fit_test.py
"""

from types import SimpleNamespace

import numpy as np

import replica_funcs as RF
from Main_PSO import PSO_main
from match_rudder_section import fit_error, match_rudder_section, mirror_selig
from penalty import penalty_check
from retruncate import circle_method, find_te_fillet, make_fillet, perp_method, trim_at
from temp_funcs import get_info_temp, untruncated_profile

_FAILED = []
BOUNDS = np.array([[0.00, 0.15], [0.05, 0.50], [0.30, 0.95], [0.00, 0.15]])


def check(name, got, want, tol):
    got, want = np.asarray(got, float), np.asarray(want, float)
    err = float(np.abs(got - want).max())
    ok = err <= tol
    if not ok:
        _FAILED.append(name)
    print(f"  {'ok ' if ok else 'FAIL'}  {name:<58s} max error {err:.3e}   (tol {tol:.0e})")


def test_pso():
    print("\nMain_PSO")
    box = np.array([[0.0, 1.0], [-2.0, 2.0]])
    sphere = lambda X: ((X - [0.3, -1.2])**2).sum(axis=1)
    r = PSO_main(sphere, 2, box, num_iter=200, stall_iter=0, verbose=False)
    check("minimum of a shifted sphere", r.best_pos, [0.3, -1.2], 1e-6)
    r = PSO_main(lambda X: -sphere(X), 2, box, type_of_PSO="maximization", num_iter=200,
                 stall_iter=0, verbose=False)
    check("maximum of the same, negated", r.best_pos, [0.3, -1.2], 1e-6)
    a = PSO_main(sphere, 2, box, num_iter=50, verbose=False)
    b = PSO_main(sphere, 2, box, num_iter=50, verbose=False)
    check("same seed, same answer", a.best_pos, b.best_pos, 0.0)
    outside = lambda X: ((X - [1.4, 0.5])**2).sum(axis=1)       # optimum beyond x0 = 1
    r = PSO_main(outside, 2, box, constraint_handling=3, num_iter=200, stall_iter=0, verbose=False)
    check("clip: optimum beyond the box lands on the bound", r.best_pos[0], 1.0, 0.0)
    check("clip: and the free variable at its optimum", r.best_pos[1], 0.5, 1e-6)
    r = PSO_main(outside, 2, box, constraint_handling=2, num_iter=200, stall_iter=0, verbose=False)
    check("penalty: stays close to the bound", r.best_pos, [1.0, 0.5], 2e-2)
    check("penalty_check: distance outside the bounds",
          penalty_check(np.array([[0.5, 0.0], [-0.25, 0.0], [1.5, 3.0]]), box), [0.0, 0.25, 1.5], 1e-15)

    class Surrogate:
        def predict(self, X, return_std=False):
            return sphere(X)
    r = PSO_main(Surrogate(), 2, box, num_iter=200, stall_iter=0, verbose=False)
    check("objective given as a surrogate with .predict", r.best_pos, [0.3, -1.2], 1e-6)
    r = PSO_main(lambda X: np.ones(len(X)), 2, box, stall_iter=15, verbose=False)
    check("stall stop after 15 iterations without improvement", r.n_iter, 17, 0)
    r = PSO_main(lambda X: ((X - [0.9, 0.5])**2).sum(axis=1), 2, box,
                 constrain_func=lambda X: X[:, 0] + X[:, 1], constraint_handling=3,
                 num_iter=300, stall_iter=0, verbose=False)
    check("inequality constraint x0 + x1 <= 0 by penalty", r.best_pos, [0.2, -0.2], 2e-3)


def test_replica():
    print("\nBezier replica")
    p = np.array([0.09056, 0.33810, 0.48857, 0.08919])
    ctrl = RF.ctrl_pts(p)[0]
    t = np.linspace(0.0, 1.0, 11)
    pts = ctrl.copy()[None].repeat(len(t), 0)
    for k in range(4, 0, -1):                                   # de Casteljau
        pts = (1 - t[:, None, None]) * pts[:, :k] + t[:, None, None] * pts[:, 1:k + 1]
    x, y = RF.build_half_airfoil(p, 10)
    check("samples match de Casteljau", np.column_stack((x, y)), pts[:, 0], 1e-15)
    check("control polygon scaled by the chord", RF.expand_ctrl_pts(p, 2.0), 2.0 * ctrl, 0.0)
    xq = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 97)))
    xd, yd = RF.build_half_airfoil(p, 200000)
    check("y at given x matches a dense sampling", RF.half_airfoil_y(p, xq)[0], np.interp(xq, xd, yd), 1e-9)
    xs, e = np.array([0.3, 0.9, 0.975]), 1e-6
    fd = (RF.y_and_slope(p, xs + e)[0] - RF.y_and_slope(p, xs - e)[0]) / (2 * e)
    check("slope matches finite differences", RF.y_and_slope(p, xs)[1], fd, 1e-8)
    J = RF.half_airfoil_jac(p, xq)
    Jfd = np.column_stack([(RF.half_airfoil_y(p + d, xq)[0] - RF.half_airfoil_y(p - d, xq)[0]) / 2e-7
                           for d in np.eye(4) * 1e-7])
    check("Jacobian matches finite differences", J, Jfd, 1e-8)
    check("LE radius is 4 Y1^2 / (3 X2)",
          (xd[1]**2 + yd[1]**2) / (2 * xd[1]), 4 * p[0]**2 / (3 * p[1]), 1e-6)
    rng = np.random.default_rng(0)
    Q = rng.uniform([-.1, -.2, -.2, -.1], [.2, 1.6, 1.3, .2], (2000, 4))
    xx = RF._curve(Q, np.broadcast_to(np.linspace(0, 1, 2001), (len(Q), 2001)))[0]
    check("is_valid agrees with a brute-force monotonic test",
          RF.is_valid(Q), (np.diff(xx, axis=1) >= -1e-15).all(axis=1), 0)
    check("every candidate inside the bounds is valid",
          RF.is_valid(rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], (10000, 4))), True, 0)


def wedge_section(m=0.18, x_te=1.02, R=0.0046):
    """Upper surface of a straight wedge (slope -m, sharp TE at x_te) with a
    round TE of radius R, as the section files store it (LE to TE)."""
    xc = (m * x_te - R * np.sqrt(1 + m * m)) / m
    phi = np.arctan(m)
    x_t = xc + R * np.sin(phi)
    xf = np.linspace(0.5, x_t, 40, endpoint=False)
    ang = np.linspace(np.pi / 2 - phi, 0.0, 7)[1:]
    x = np.r_[xf, xc + R * np.cos(ang)]
    y = np.r_[m * (x_te - xf), R * np.sin(ang)]
    return x, y, xc, x_t


def test_te():
    print("\nsharpening and TE fillet")
    x, y, xc, x_t = wedge_section()
    xs, ys, c = untruncated_profile(x, y, 1e-3)
    check("wedge: sharp TE where the flank meets y = 0", [xs[-1], ys[-1], c], [1.02, 0.0, 1.02 - 0.5], 1e-12)
    check("wedge: arc points dropped, flank kept", len(xs), 40, 0)
    te = find_te_fillet(mirror_selig(x, y))
    check("wedge: TE circle centre and radius", [te["xc"], te["R"]], [xc, 0.0046], 1e-12)
    check("wedge: fillet starts at the tangent point", te["x_start"], x_t, 1e-12)
    check("wedge: flank slope", te["flank_slope"], -0.18, 1e-12)
    try:
        untruncated_profile(np.linspace(0, 1, 50), np.sqrt(1 - np.linspace(0, 1, 50)**2), 1e-3)
        check("no straight flank raises", 0, 1, 0)
    except ValueError:
        check("no straight flank raises", 0, 0, 0)

    cut, slope = np.array([0.99, 0.004]), -0.2
    center, r = perp_method(cut, slope)
    check("perp_method: centre on y = 0, radius at right angles to the surface",
          [center[1], np.dot(cut - center, [1.0, slope])], [0.0, 0.0], 1e-15)
    center, r = circle_method(cut, (1.0, 0.0))
    check("circle_method: through both cut points and the TE point",
          [np.hypot(*(cut - center)), np.hypot(1.0 - center[0], 0.0)], [r, r], 1e-12)
    arc = make_fillet(center, cut)
    check("make_fillet: 24 points on the circle", [len(arc), *np.hypot(*(arc - center).T)], [24, *[r] * 24], 1e-12)
    check("make_fillet: TE to cut, angle rising", np.all(np.diff(np.arctan2(arc[:, 1], arc[:, 0] - center[0])) > 0), True, 0)
    sec = mirror_selig(*RF.build_half_airfoil([0.07, 0.2, 0.3, 0.12], 50))
    kept = trim_at(sec, cut)
    check("trim_at: upstream of the cut, from upper to lower cut point",
          [kept[1:-1, 0].max() < cut[0], *kept[0], *kept[-1]], [1, *cut, cut[0], -cut[1]], 0.0)


def test_match():
    print("\nfit of section_cut_y80_selig.dat")
    xr, yr, xs, ys, cs = get_info_temp("section_cut_y80_selig.dat", 0.001)
    check("sharp TE of the y = 80 section", [len(xs), xs[-1]], [96, 1.020868], 1e-6)
    pso = SimpleNamespace(swarm_size=50, num_iter=100, C1=2.05, C2=2.05, PSO_variant=1, w_max=0.9,
                          w_min=0.5, vmax_frac=0.2, constraint_handling=3, penalty_factor=5.0,
                          craziness=0, stall_iter=15, seed=42, ConsOn=0, Obj=0, polish=True,
                          verbose=False)
    te = find_te_fillet(mirror_selig(xr, yr))
    for method in (0, 1, 2):
        foil = SimpleNamespace(slope_tol=0.001, n_bez_samples=200, n_target_stations=0,
                               fillet_method=method)
        P, val, r, rc, rs, cp = match_rudder_section(xr, yr, xs, ys, cs, pso, foil, BOUNDS)
        if method == 0:
            check("best fit inside the bounds (X3 on its bound)", P, [0.07337, 0.21777, 0.3, 0.12354], 1e-5)
            check("fit error returned = error of the returned point", val,
                  fit_error(P, xs / cs, ys / cs)[0], 0.0)
            check("replica closed and in Selig order (one LE point)",
                  [*(rc[0] - rc[-1]), np.sum(rc[:, 0] == rc[:, 0].min()), np.argmin(rc[:, 0]) == len(rc) // 2],
                  [0, 0, 1, 1], 0.0)
            cut = rc[25]
            y, s = RF.y_and_slope(P, cut[0] / cs)
            check("method 0: fillet tangent to the replica at the cut",
                  np.dot(cut - (rc[0] - [r, 0]), [1.0, s[0]]), 0.0, 1e-12)
            again = match_rudder_section(xr, yr, xs, ys, cs, pso, foil, BOUNDS)
            check("same settings, same replica", again[3], rc, 0.0)
        elif method == 1:
            check("method 1: TE point is the original's", rc[0], [xr[-1], yr[-1]], 1e-12)
        else:
            check("method 2: TE radius is the original's", r, te["R"], 1e-12)


def main():
    print("section fit self-check")
    test_pso()
    test_replica()
    test_te()
    test_match()
    print()
    if _FAILED:
        print(f"{len(_FAILED)} check(s) FAILED: " + ", ".join(_FAILED))
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
