"""
bb2_aft_setup.py

BB2's aft half as XGeom set-up files (13 Nov 2026; the user: "the hull and the rudder is incorrect ... We will
only build the aft half of the sub for now, so just the hull aft half, rudder and propeller ... now just extract
the correct aft hull, rudder set up text files"; the rudder as a planar rudder component, the user's choice). From
the quarter IGES BB2_H3_DRDC_Q2R_quarter_hull_half_upstream.iges: the hull's aft half with the hub (y >= 0,
z >= 0, x 44 to 35078 mm) and one stern plane at 45 deg; mm, x towards the bow.

hull     Its sections are circles (r the same all round within 0.002 mm), so the aft half is a body of
         revolution: hull_modify without a nose (Ln 0: its front the flat cut at mid-length, x 35077.9), the
         parallel middle body to x 22202.5 (r* 4799.81 mm), the tail to the hub's forward end at x 1929.9, where
         the radius is drawing F6's hub base 797.22 mm (the hub is the propeller's: bb2_prop_setup.txt). The
         station table bb2_hull_aft_params.dat (the profile in the plane through the axis at 45 deg, N_TAIL
         cosine-spaced stations along the tail) and the set-up bb2_hull.txt, the tail's CST order the one that
         fits best (xgeom_from_cad.hull_setup). Frame: the hull tool's x (m, aft from the cut) is
         (35077.9 - x) / 1000, its y is -y.
rudder   The stern plane as a planar rudder (wing_modify, kind rudder), from its sections normal to its span in
         its 45 deg plane, s = (y + z) / sqrt(2) (bb2_sternplane_loft.flat_loop: the LE and the TE on the plane
         of symmetry; the sections are symmetric to 1e-5 of the chord). The root (eta 0) where its LE line
         meets the hull, s 3061.4: the root section lies outside the hull, its LE on it, and the vehicle extends
         it to the hull along the TE line (XGeom's TE rake 0; BB2's TE is square to the axis within 0.13 deg).
         The tip (eta 1) at the foot of the rounded cap, s 6811.5: the planar tip is flat, the 236 mm cap is
         left off; the TE turns forward into the cap's rounding over the last 20 mm, so the tip row is the
         section at S_LAST with the LE and TE lines carried on. Per row: chord, sweep (the LE aft of the root's),
         rake 0, pitch 0, camber 0; the thickness over the chord fitted by least squares as t/c times the
         section shape plus the changes dt1 .. dtk (wing_modify's Bernstein changes, zero at the LE and the
         TE). The section shape: the mean over the rows of the thickness over its maximum, written as
         bb2_sternplane_section.dat (Selig x y; its blunt TE's round kept; NACA 0016-like, 16.1 % at x/c 0.30);
         k the number of change points with the least error (xgeom_from_cad.pick). Each distribution in the
         form that fits best (xgeom_from_cad.wing_setup). Frame: the rudder's x (mm) aft from the root's LE,
         y the span from the root, z along (0, 1, -1) / sqrt(2).
check    bb2_aft_setup_check.png: the hull's profile against the IGES, the planform and sections rebuilt from the
         rudder's set-up against the CAD's; the deviations printed.

    python bb2_aft_setup.py <quarter.iges> [out_dir]          (out_dir: default bb2/outputs)
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import bb2_build as B                                      # noqa: E402
import bb2_sections as S                                   # noqa: E402
import bb2_sternplane_loft as L                            # noqa: E402
import hull_modify as HM                                   # noqa: E402
import vehicle_cad as VC                                   # noqa: E402
import wing_modify as WM                                   # noqa: E402
import xgeom_from_cad as FC                                # noqa: E402

N_TAIL = 201                 # stations along the tail
N_ROWS = 21                  # rows of the rudder's table, evenly along its span
N_X = 201                    # chord fractions where the thickness is fitted (cosine spacing)
S_LAST = 6790.0              # the last section before the TE turns into the cap's rounding
LINE_S = np.arange(3100.0, 6701.0, 200.0)    # sections that give the LE and TE lines
CHECK_S = (3100.0, 4000.0, 5000.0, 6000.0, 6700.0)
MAX_K = 8                    # change points tried: 0 .. MAX_K


def parts(path):
    """(hull shell, stern plane solid) of the quarter IGES: the longest shell, the shortest solid."""
    shape = S.read_cad(path)
    length = lambda s: VC._bbox(s)[1, 0] - VC._bbox(s)[0, 0]     # noqa: E731
    hull = max(S._explore(shape, "SHELL"), key=length)
    plane = min(S._explore(shape, "SOLID"), key=length)
    return hull, plane


# --------------------------------------------------------------------------
# The hull
# --------------------------------------------------------------------------

def aft_hull(hull):
    """(HullParams of the aft half, its frame: x_cut, x_pmb, x_fwd, the IGES profile (x, r) in mm)."""
    x_fwd = S.hub_forward_x(hull)
    x_pmb, r_pmb = B.middle_body(hull)
    x_cut = float(VC._bbox(hull)[1, 0])
    xi, ri = B.hull_profile(hull)
    lm, lt = (x_cut - x_pmb) / 1000.0, (x_pmb - x_fwd) / 1000.0
    xs = x_pmb - (x_pmb - x_fwd) * L.cosine(N_TAIL)             # IGES x along the tail, the middle body first
    rt = np.interp(xs, xi, ri) / 1000.0
    rt[0], rt[-1] = r_pmb / 1000.0, S.HUB_R_FWD / 1000.0
    x = np.r_[0.0, lm + (x_pmb - xs) / 1000.0]
    r = np.r_[r_pmb / 1000.0, rt]
    note = (f"BB2's aft half from {os.path.basename(SOURCE)} by bb2_aft_setup.py: no nose (the front the flat cut at "
            f"mid-length, IGES x {x_cut:.4f} mm), the middle body to x {x_pmb:.4f}, the tail to the hub's forward end "
            f"at x {x_fwd:.4f} (r {S.HUB_R_FWD:g} mm, drawing F6's hub base); a body of revolution; IGES x = "
            f"{x_cut:.4f} - 1000 x_m, y = -y")
    params = HM.HullParams(0.0, lm, lt, 0.0, x, r, r.copy(), True, None, [note])
    return params, {"x_cut": x_cut, "x_pmb": x_pmb, "x_fwd": x_fwd, "xi": xi, "ri": ri}


# --------------------------------------------------------------------------
# The stern plane
# --------------------------------------------------------------------------

def section(plane, s, xf):
    """The section at s: its LE and TE x (IGES), chord, and the thickness and camber over the chord at the
    fractions xf (each over the chord)."""
    lp = L.flat_loop(plane, s)                     # (x, w, s): the LE first, one side to the TE, back
    k = (len(lp) - 1) // 2
    le, te = lp[0], lp[k]
    c = float(le[0] - te[0])
    a, b = lp[:k + 1], lp[k:][::-1]                # each from the LE to the TE
    za = np.interp(xf, (le[0] - a[:, 0]) / c, a[:, 1])
    zb = np.interp(xf, (le[0] - b[:, 0]) / c, b[:, 1])
    zu, zl = (za, zb) if za.mean() > zb.mean() else (zb, za)
    return {"s": float(s), "x_le": float(le[0]), "x_te": float(te[0]), "w_le": float(le[1]), "chord": c,
            "T": (zu - zl) / c, "C": 0.5 * (zu + zl) / c, "loop": lp}


def lines(plane, xf):
    """The LE and TE lines, x = a s + b each (least squares over LINE_S), and the largest misfit (mm)."""
    secs = [section(plane, s, xf) for s in LINE_S]
    s = np.array([q["s"] for q in secs])
    le, te = np.polyfit(s, [q["x_le"] for q in secs], 1), np.polyfit(s, [q["x_te"] for q in secs], 1)
    miss = max(np.abs(np.polyval(le, s) - [q["x_le"] for q in secs]).max(),
               np.abs(np.polyval(te, s) - [q["x_te"] for q in secs]).max())
    return le, te, float(miss)


def root_s(le, xi, ri):
    """s where the LE line meets the hull (in the 45 deg plane the hull's surface lies at s = r(x))."""
    from scipy.optimize import brentq
    return float(brentq(lambda s: np.interp(np.polyval(le, s), xi, ri) - s, 2000.0, 4000.0))


def write_section(path, xf, tn, tc):
    """The section shape as Selig coordinates: the upper side from the TE to the LE, the lower back; the
    thickness tn (over its maximum) times tc, symmetric."""
    y = 0.5 * tc * tn
    xy = np.vstack((np.column_stack((xf[::-1], y[::-1])), np.column_stack((xf[1:], -y[1:]))))
    with open(path, "w", newline="\n") as fh:
        fh.write("bb2_sternplane_section (BB2's stern plane: its sections' mean thickness shape, bb2_aft_setup.py)\n")
        fh.write("\n".join(f"{a:.8f} {b:.8f}" for a, b in xy) + "\n")
    return path


def thickness_fit(T, T0, xf, k):
    """(t/c, dt (k,), largest misfit over the chord) of a thickness T over the chord: T0 t/c + the changes."""
    A = np.column_stack((T0, WM.change_basis(xf, k)))
    coef = np.linalg.lstsq(A, T, rcond=None)[0]
    return float(coef[0]), coef[1:], float(np.abs(A @ coef - T).max())


def stern_plane(plane, xi, ri, out):
    """(WingParams of the stern plane as a planar rudder, its frame and the misfits)."""
    xf = L.cosine(N_X)
    le, te, miss = lines(plane, xf)
    s0, s1 = root_s(le, xi, ri), L.S_CAP
    span = s1 - s0
    eta = np.linspace(0.0, 1.0, N_ROWS)
    secs = [section(plane, s, xf) for s in s0 + eta[:-1] * span] + [section(plane, S_LAST, xf)]
    camber = max(float(np.abs(q["C"]).max()) for q in secs)
    tn = np.mean([q["T"] / q["T"].max() for q in secs], axis=0)
    tc_mean = float(np.mean([q["T"].max() for q in secs]))
    sec_path = write_section(os.path.join(out, "bb2_sternplane_section.dat"), xf, tn, tc_mean)
    T0 = WM.base_shape(sec_path)[0](xf)
    fits = {k: [thickness_fit(q["T"], T0, xf, k) for q in secs] for k in range(MAX_K + 1)}
    errs = {k: max(f[2] for f in fits[k]) for k in fits}
    k, _ = FC.pick([(k, k, errs[k]) for k in errs])
    x_le = np.array([q["x_le"] for q in secs[:-1]] + [np.polyval(le, s1)])
    x_te = np.array([q["x_te"] for q in secs[:-1]] + [np.polyval(te, s1)])
    values = {"chord": x_le - x_te, "sweep": x_le[0] - x_le, "rake": np.zeros(N_ROWS), "pitch": np.zeros(N_ROWS),
              "thickness": np.array([f[0] for f in fits[k]]), "camber": np.zeros(N_ROWS)}
    for i in range(k):
        values[f"dt{i + 1}"] = np.array([f[1][i] for f in fits[k]])
        values[f"dc{i + 1}"] = np.zeros(N_ROWS)
    params = WM.WingParams(span, sec_path, eta, values, k, WM.REF_XC, WM.CAMBER_XC, "rudder", "")
    frame = {"s0": s0, "s1": s1, "x_le0": float(x_le[0]), "le": le, "te": te, "line_miss": miss, "camber": camber,
             "k": k, "errs": errs, "tc_mean": tc_mean}
    return params, frame


def to_iges(loops, frame):
    """Rudder-frame points (mm: x aft from the root's LE, y the span, z across) in the IGES frame."""
    p = np.asarray(loops, dtype=float)
    x = frame["x_le0"] - p[..., 0]
    s = frame["s0"] + p[..., 1]
    return x[..., None] * np.array([1.0, 0.0, 0.0]) + s[..., None] * L.E_SPAN + p[..., 2:3] * L.E_THICK


def rudder_check(plane, design, params, frame):
    """{s: largest distance (mm) from the rebuilt section's points to the CAD section's polyline, in the
    section's plane}, and the planform's misfit (LE and TE x, mm, over the span)."""
    out = {}
    for s in CHECK_S:
        eta = (s - frame["s0"]) / (frame["s1"] - frame["s0"])
        loop = to_iges(WM.section_loops(design, params, [eta], 181), frame)[0]
        cad = L.flat_loop(plane, s)
        q = np.column_stack((loop[:, 0], loop @ L.E_THICK))
        seg_a, seg_b = cad[:-1, :2], cad[1:, :2]
        d = seg_b - seg_a
        t = np.clip(np.einsum("ijk,jk->ij", q[:, None, :] - seg_a[None], d) / np.einsum("jk,jk->j", d, d), 0.0, 1.0)
        near = seg_a[None] + t[..., None] * d[None]
        out[s] = float(np.linalg.norm(q[:, None, :] - near, axis=2).min(axis=1).max())
    eta = np.linspace(0.0, 1.0, 101)
    s = frame["s0"] + eta * (frame["s1"] - frame["s0"])
    v = design.values(eta)
    x_le = frame["x_le0"] - v["sweep"]
    x_te = x_le - v["chord"]
    plan = float(max(np.abs(x_le - np.polyval(frame["le"], s)).max(), np.abs(x_te - np.polyval(frame["te"], s)).max()))
    return out, plan


def plot_check(path, hull_design, hull_frame, design, params, frame, plane):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(14, 10))
    FigureCanvasAgg(fig)
    ax1, ax2 = fig.add_subplot(2, 2, (1, 2)), fig.add_subplot(2, 2, 3)
    ax3 = fig.add_subplot(2, 2, 4)
    hf = hull_frame
    xs = np.linspace(0.0, hull_design.length, 2001)
    ax1.plot(hf["xi"], hf["ri"], "-", color="#9a9994", lw=4, label="IGES (the profile at 45 deg)")
    ax1.plot(hf["x_cut"] - 1000.0 * xs, 1000.0 * hull_design.radius("r", xs), "-", color="#2a78d6", lw=1.4,
             label="bb2_hull.txt")
    for x in (hf["x_cut"], hf["x_pmb"], hf["x_fwd"]):
        ax1.axvline(x, color="#c9c8c3", lw=0.8, ls=":")
    eta = np.linspace(0.0, 1.0, 101)
    v = design.values(eta)
    s = frame["s0"] + eta * (frame["s1"] - frame["s0"])
    x_le = frame["x_le0"] - v["sweep"]
    for xx, yy in ((x_le, s), (x_le - v["chord"], s)):
        ax1.plot(xx, yy, "-", color="#e34948", lw=1.4)
    ax1.plot([], [], "-", color="#e34948", label="bb2_sternplane.txt: LE and TE in the 45 deg plane (s)")
    for sc in (L.S_CAP, frame["s0"]):
        lp = L.flat_loop(plane, sc - 0.5 if sc == L.S_CAP else sc)
        ax1.plot([lp[:, 0].max(), lp[:, 0].min()], [sc, sc], ":", color="#9a9994", lw=1.0)
    ax1.set_xlabel("x (mm, the IGES frame: towards the bow)")
    ax1.set_ylabel("r, s (mm)")
    ax1.set_aspect("equal")
    ax1.invert_xaxis()
    ax1.grid(True, color="#e4e3df")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_title(f"aft hull: L {hull_design.length:.4f} m, middle body {hull_design.middle:.4f} m, tail "
                  f"{hull_design.tail:.4f} m; the stern plane from s {frame['s0']:.1f} (its LE on the hull) to "
                  f"{frame['s1']:.1f} (the cap's foot)", fontsize=10, loc="left")
    xr = np.linspace(hf["x_fwd"], hf["x_cut"], 3001)
    ax2.plot(xr, 1000.0 * hull_design.radius("r", (hf["x_cut"] - xr) / 1000.0) - np.interp(xr, hf["xi"], hf["ri"]),
             "-", color="#2a78d6")
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("hull radius: set-up - IGES (mm)")
    ax2.invert_xaxis()
    ax2.grid(True, color="#e4e3df")
    for i, sc in enumerate(CHECK_S):
        e = (sc - frame["s0"]) / (frame["s1"] - frame["s0"])
        loop = WM.section_loops(design, params, [e], 181)[0]
        cad = L.flat_loop(plane, sc)
        col = ["#2a78d6", "#1f8a70", "#e8a33d", "#8e5ac8", "#e34948"][i]
        ax3.plot((frame["x_le0"] - loop[:, 0]) - (frame["x_le0"] - 0.0), loop[:, 2], "-", color=col, lw=1.0,
                 label=f"s {sc:.0f}")
        ax3.plot(cad[:, 0] - frame["x_le0"], cad[:, 1], ":", color="black", lw=0.8)
    ax3.set_aspect("equal")
    ax3.set_xlabel("x - the root's LE x (mm)")
    ax3.set_ylabel("w (mm)")
    ax3.grid(True, color="#e4e3df")
    ax3.legend(fontsize=7, ncol=3)
    ax3.set_title("sections: set-up (colour) and CAD (dotted)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


SOURCE = ""


def main():
    global SOURCE
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    SOURCE = sys.argv[1]
    out = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "outputs"))
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    hull, plane = parts(SOURCE)
    params, hf = aft_hull(hull)
    pp = HM.write_params(os.path.join(out, "bb2_hull_aft_params.dat"), params)
    setup, orders = FC.hull_setup(pp, os.path.join(out, "bb2_hull.txt"), os.path.basename(pp))
    hd = HM.load_component(setup).design
    xr = np.linspace(hf["x_fwd"], hf["x_cut"], 3001)
    h_err = np.abs(1000.0 * hd.radius("r", (hf["x_cut"] - xr) / 1000.0) - np.interp(xr, hf["xi"], hf["ri"])).max()
    print(f"hull: the cut at x {hf['x_cut']:.4f}, the middle body from x {hf['x_pmb']:.4f} (r* {1000 * params.r[0]:.3f} "
          f"mm), the hub's forward end at x {hf['x_fwd']:.4f}: L {hd.length:.4f} m; "
          + ", ".join(f"{c} order {v[0]}" for c, v in orders.items())
          + f"; largest difference from the IGES profile {h_err:.3f} mm ({time.time() - t0:.0f} s)")
    print(f"  {pp}\n  {setup}")
    wp, fr = stern_plane(plane, hf["xi"], hf["ri"], out)
    print(f"stern plane: LE line x = {fr['le'][0]:.6f} s + {fr['le'][1]:.3f}, TE line x = {fr['te'][0]:.6f} s + "
          f"{fr['te'][1]:.3f} (misfit {fr['line_miss']:.3f} mm); root s {fr['s0']:.3f} (the LE on the hull), tip s "
          f"{fr['s1']:.1f}: span {wp.span:.2f} mm, chord {wp.values['chord'][0]:.2f} to {wp.values['chord'][-1]:.2f} mm; "
          f"camber at most {fr['camber']:.1e} of the chord (0 taken); t/c {wp.values['thickness'].min():.5f} to "
          f"{wp.values['thickness'].max():.5f}; change points {fr['k']} (thickness misfit "
          + ", ".join(f"k {k} {100 * e:.3f} %" for k, e in fr["errs"].items()) + " of the chord)")
    rsetup, forms = FC.wing_setup(wp, os.path.join(out, "bb2_sternplane.txt"), "rudder", "bb2_sternplane",
                                  [f"BB2's stern plane from {os.path.basename(SOURCE)} by bb2_aft_setup.py: sections "
                                   f"normal to its span s = (y + z)/sqrt(2) from s {fr['s0']:.3f} (its LE on the hull) "
                                   f"to {fr['s1']:.1f} (the foot of its rounded cap, left off: the tip flat)",
                                   f"frame: x = {fr['x_le0']:.4f} - x_rudder (mm), s = {fr['s0']:.3f} + y_rudder, w = "
                                   f"z_rudder along (0, 1, -1)/sqrt(2); the TE square to the axis (XGeom's TE rake 0)"])
    ld = WM.load_component(rsetup)
    dev, plan = rudder_check(plane, ld.design, ld.params, fr)
    print("  forms: " + "; ".join(f"{c} " + ("2 segments" if f[0][0] == 2 else f"{f[0][1] + 1} points")
                                  + f" ({100 * f[1]:.2f} %)" for c, f in forms.items()))
    print(f"  rebuilt against the CAD: planform (LE and TE) within {plan:.3f} mm; sections within "
          + ", ".join(f"{d:.2f} mm at s {s:.0f}" for s, d in dev.items()))
    print(f"  {rsetup}\n  {os.path.join(out, 'bb2_sternplane_section.dat')}")
    png = plot_check(os.path.join(out, "bb2_aft_setup_check.png"), hd, hf, ld.design, ld.params, fr, plane)
    print(f"  {png}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
