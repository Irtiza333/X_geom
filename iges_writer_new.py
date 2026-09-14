"""
iges_writer_new.py

Minimal, dependency-free IGES 5.3 writer for Rational B-Spline Surfaces
(entity type 128). Produces the file structure TM 2013-180 Annex B expects
for the five DRDC blade surfaces:

  - five type-128 surfaces labelled "Blade" with subscripts 1..5
    (1 = TE surface, 2 = central pressure, 3 = LE surface,
     4 = central suction, 5 = tip surface);
  - the propeller accuracy written into the Global Section
    "Minimum User-Intended Resolution" field (Annex B item 1).

Only numpy is required, so the CAD export can run in ANY Python
environment; the pythonOCC path in X_CAD_new.py remains available for
sewing/validity checks when OCC is installed.

Surface input format: dict with
    ku, kv   : spline degrees (u = rows, v = cols)
    tu, tv   : full clamped knot vectors (scipy tck convention)
    poles    : (nu, nv, 3) control points
"""

import numpy as np
from datetime import datetime


# ---------------------------------------------------------------------------
# low-level record helpers
# ---------------------------------------------------------------------------
def _hollerith(s):
    return f"{len(s)}H{s}"


def _fmt_real(x):
    """IGES real: compact, always with a decimal point."""
    s = f"{float(x):.10G}"
    if "E" not in s and "." not in s:
        s += "."
    return s


def _pack_section(fields, letter, start_seq, col=72):
    """Pack comma-separated fields into fixed-width section lines."""
    lines = []
    cur = ""
    for i, f in enumerate(fields):
        tok = f + ("," if i < len(fields) - 1 else ";")
        if len(cur) + len(tok) > col:
            lines.append(cur)
            cur = tok
        else:
            cur += tok
    if cur:
        lines.append(cur)
    out = []
    for j, ln in enumerate(lines):
        out.append(f"{ln:<{col}}{letter}{start_seq + j:7d}")
    return out


def _pack_pdata(fields, de_pointer, start_seq):
    """Parameter-data lines: data in cols 1-64, DE pointer in 65-72."""
    lines = []
    cur = ""
    for i, f in enumerate(fields):
        tok = f + ("," if i < len(fields) - 1 else ";")
        if len(cur) + len(tok) > 64:
            lines.append(cur)
            cur = tok
        else:
            cur += tok
    if cur:
        lines.append(cur)
    out = []
    for j, ln in enumerate(lines):
        out.append(f"{ln:<64}{de_pointer:8d}P{start_seq + j:7d}")
    return out


# ---------------------------------------------------------------------------
# entity 128 parameter data
# ---------------------------------------------------------------------------
def _surface_128_fields(srf):
    ku, kv = int(srf["ku"]), int(srf["kv"])
    tu = np.asarray(srf["tu"], dtype=float)
    tv = np.asarray(srf["tv"], dtype=float)
    poles = np.asarray(srf["poles"], dtype=float)
    nu, nv, _ = poles.shape
    assert len(tu) == nu + ku + 1, "u knot vector inconsistent with poles"
    assert len(tv) == nv + kv + 1, "v knot vector inconsistent with poles"

    K1, K2 = nu - 1, nv - 1
    M1, M2 = ku, kv
    fields = ["128", str(K1), str(K2), str(M1), str(M2),
              "0", "0", "1", "0", "0"]        # PROP1..5: open, polynomial
    fields += [_fmt_real(x) for x in tu]
    fields += [_fmt_real(x) for x in tv]
    fields += ["1." for _ in range(nu * nv)]  # unit weights
    # control points: FIRST index (u) varies fastest per the IGES spec
    for j in range(nv):
        for i in range(nu):
            fields += [_fmt_real(poles[i, j, 0]),
                       _fmt_real(poles[i, j, 1]),
                       _fmt_real(poles[i, j, 2])]
    fields += [_fmt_real(tu[M1]), _fmt_real(tu[K1 + 1]),
               _fmt_real(tv[M2]), _fmt_real(tv[K2 + 1])]
    return fields


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------
def write_iges_128(surfaces, path, accuracy=1.0e-6, units="MM",
                   label="Blade", product="DRDC 5-surface blade"):
    """Write a list of B-spline surfaces to an IGES file.

    surfaces : list of surface dicts (see module docstring); subscripts are
               assigned 1..N in list order (Annex B ordering expected).
    accuracy : Minimum User-Intended Resolution (same units as the data).
    """
    now = datetime.now().strftime("%Y%m%d.%H%M%S")
    units_flag = {"MM": 2, "M": 6, "IN": 1}[units]

    start = [f"{'DRDC five-surface propeller blade (tip-smoothed)':<72}S{1:7d}"]

    gfields = [
        "1H,", "1H;",
        _hollerith(product), _hollerith(str(path).replace("\\", "/")[-60:]),
        _hollerith("blade_surface_new/tip_surfaces_new"),
        _hollerith("iges_writer_new 1.0"),
        "32", "38", "6", "308", "15",
        _hollerith(product),
        "1.", str(units_flag), _hollerith(units),
        "1", "0.01",
        _hollerith(now),
        _fmt_real(accuracy),          # Minimum User-Intended Resolution
        "0.",
        _hollerith("irtiza"), _hollerith(""),
        "11", "0",
        _hollerith(now),
    ]
    global_lines = _pack_section(gfields, "G", 1)

    # build P-section first to know line counts, then D-section
    p_lines = []
    d_lines = []
    p_seq = 1
    for k, srf in enumerate(surfaces):
        de_ptr = 2 * k + 1              # sequence of the entity's first D line
        fields = _surface_128_fields(srf)
        ent_lines = _pack_pdata(fields, de_ptr, p_seq)
        n_p = len(ent_lines)

        d1 = (f"{128:8d}{p_seq:8d}{0:8d}{0:8d}{0:8d}{0:8d}{0:8d}{0:8d}"
              f"00000000D{de_ptr:7d}")
        d2 = (f"{128:8d}{0:8d}{0:8d}{n_p:8d}{0:8d}{'':8}{'':8}"
              f"{label:>8}{k + 1:8d}D{de_ptr + 1:7d}")
        d_lines += [d1, d2]
        p_lines += ent_lines
        p_seq += n_p

    n_s, n_g, n_d, n_p_total = 1, len(global_lines), len(d_lines), len(p_lines)
    terminate = (f"S{n_s:7d}G{n_g:7d}D{n_d:7d}P{n_p_total:7d}"
                 f"{'':40}T{1:7d}")

    with open(path, "w", newline="\n") as fh:
        for ln in start + global_lines + d_lines + p_lines + [terminate]:
            fh.write(ln + "\n")
    return path


# ---------------------------------------------------------------------------
# scipy bridge
# ---------------------------------------------------------------------------
def surface_from_grid(grid, u_params=None, v_params=None, scale=1.0):
    """Interpolating bicubic B-spline through a structured (nu, nv, 3) grid,
    returned in the writer's surface-dict format (uses scipy)."""
    from scipy.interpolate import RectBivariateSpline
    grid = np.asarray(grid, dtype=float) * scale
    nu, nv, _ = grid.shape
    u = np.linspace(0.0, 1.0, nu) if u_params is None else np.asarray(u_params)
    v = np.linspace(0.0, 1.0, nv) if v_params is None else np.asarray(v_params)
    ku = min(3, nu - 1)
    kv = min(3, nv - 1)
    splines = [RectBivariateSpline(u, v, grid[:, :, c], kx=ku, ky=kv, s=0)
               for c in range(3)]
    tu, tv = splines[0].tck[0], splines[0].tck[1]
    n_cu = len(tu) - ku - 1
    n_cv = len(tv) - kv - 1
    poles = np.stack([s.get_coeffs().reshape(n_cu, n_cv) for s in splines],
                     axis=-1)
    return dict(ku=ku, kv=kv, tu=tu, tv=tv, poles=poles)


def eval_surface(srf, u, v):
    """Evaluate a writer surface dict (for verification)."""
    from scipy.interpolate import bisplev
    tu, tv = srf["tu"], srf["tv"]
    ku, kv = srf["ku"], srf["kv"]
    out = np.empty((len(np.atleast_1d(u)), len(np.atleast_1d(v)), 3))
    for c in range(3):
        coeffs = srf["poles"][:, :, c].ravel()
        out[:, :, c] = bisplev(np.atleast_1d(u), np.atleast_1d(v),
                               (tu, tv, coeffs, ku, kv))
    return out
