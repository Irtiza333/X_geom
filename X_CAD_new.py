"""
X_CAD_new.py

DRDC-method CAD export, replacing the loft-and-cap approach of the original
X_CAD.py (which this file supersedes; the original X_CAD.py is untouched).

Two entry points:
  X_CAD(grids, x1)                       -> grids to IGES (main function)
  X_CAD_from_design(pitch_con, chord_con, x1)
                                         -> design vector to IGES (wrapper:
                                            X_blade -> build_drdc_grids ->
                                            X_CAD); drop-in for cad_worker.

Input to X_CAD is the five structured surface grids produced by
tip_surfaces_new.build_drdc_grids(). Each grid is interpolated by a
tensor-product cubic B-spline (scipy, s=0 so the grid points are ON the
surface), converted to an OCC Geom_BSplineSurface, made into a face, sewed,
and written to IGES (OpenCascade through cad_kernel since 27 Nov 2026, so
pythonocc-core or cadquery-ocp; this module imports no OCC itself). Because
adjacent grids share identical boundary point rows/columns, the faces match
along their common edges and the sewing merges them; there is no filling
solver, no loft, no cap, and no degenerate tip ring anywhere.

Surface naming follows TM 2013-180 Annex B:
  Blade 1 = trailing edge surface   ('te_strip')
  Blade 2 = central pressure side   ('central_pressure')
  Blade 3 = leading edge surface    ('le_strip')
  Blade 4 = central suction side    ('central_suction')
  Blade 5 = tip surface             ('tip')
Orientation flips (to satisfy the outward-normal and parameter-direction
conventions of Annex B) are collected in ORIENTATIONS below so they can be
adjusted in one place once checked in the target CAD/mesh tool.
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline

import cad_kernel as CK
from pipeline_config import cad_output_paths

MM_PER_M = 1000.0
SEW_TOL = 1.0e-3          # mm; grids share boundary points exactly

# (flip_rows, flip_cols, transpose) applied to each grid before splining.
# Adjust here if the meshing tool needs the Annex B parameter directions.
ORIENTATIONS = {
    "te_strip": (False, False, False),
    "central_pressure": (False, False, False),
    "le_strip": (False, False, False),
    "central_suction": (False, False, False),
    "tip": (False, False, False),
}

ANNEX_B_ORDER = [
    ("Blade 1", "te_strip"),
    ("Blade 2", "central_pressure"),
    ("Blade 3", "le_strip"),
    ("Blade 4", "central_suction"),
    ("Blade 5", "tip"),
]


def _clamped_knots(t):
    """Full clamped knot vector with every knot within 1e-12 of the first of
    its run made equal to it: the kernel counts equal knots as one knot of
    that multiplicity."""
    out = []
    for kn in t:
        out.append(out[-1] if out and abs(kn - out[-1]) < 1e-12 else float(kn))
    return np.array(out)


def _occ_surface_from_tck(tu, tv, coeffs, ku, kv, transpose=False):
    """Build a Geom_BSplineSurface (cad_kernel.bspline_surface) from scipy
    tck data.

    coeffs : (n_cu, n_cv, 3). transpose=True swaps the role of the two
    parameter directions (used by the runtime self-check below).
    """
    if transpose:
        coeffs = np.transpose(coeffs, (1, 0, 2))
        tu, tv = tv, tu
        ku, kv = kv, ku
    # the kernel's pole grid holds n_v rows of n_u poles
    return CK.bspline_surface(np.transpose(coeffs, (1, 0, 2)), _clamped_knots(tu),
                              _clamped_knots(tv), ku, kv)


def grid_to_bspline_surface(grid, u_params=None, v_params=None,
                            check_tol_mm=1.0e-3):
    """Interpolating bicubic B-spline surface through a structured grid,
    SELF-VERIFIED at runtime.

    The spline is fitted with scipy using OUR parameter values (u_params /
    v_params; this matters because the wrap grids are clustered 100:1 near
    the edges and any interpolator that picks its own parameterization
    overshoots into large lobes there). The scipy spline is converted to an
    OCC Geom_BSplineSurface, then the OCC surface is EVALUATED at every grid
    parameter and compared against the grid points. If the deviation exceeds
    check_tol_mm the conversion is retried transposed; if it still fails, a
    RuntimeError is raised rather than exporting garbage geometry.
    """
    grid = np.asarray(grid, dtype=float)
    nu, nv, _ = grid.shape
    u = np.linspace(0.0, 1.0, nu) if u_params is None else np.asarray(u_params, dtype=float)
    v = np.linspace(0.0, 1.0, nv) if v_params is None else np.asarray(v_params, dtype=float)

    ku = min(3, nu - 1)
    kv = min(3, nv - 1)
    splines = [RectBivariateSpline(u, v, grid[:, :, c] * MM_PER_M,
                                   kx=ku, ky=kv, s=0) for c in range(3)]
    tu, tv = splines[0].tck[0], splines[0].tck[1]
    n_cu = len(tu) - ku - 1
    n_cv = len(tv) - kv - 1
    coeffs = np.stack([s.get_coeffs().reshape(n_cu, n_cv) for s in splines],
                      axis=-1)

    # sanity: the scipy spline itself must interpolate the grid
    chk = np.stack([splines[c](u, v) for c in range(3)], axis=-1)
    scipy_err = np.abs(chk - grid * MM_PER_M).max()
    if scipy_err > check_tol_mm:
        raise RuntimeError(
            f"scipy surface fit failed to interpolate (err {scipy_err:.3g} mm)")

    for transpose in (False, True):
        srf = _occ_surface_from_tck(tu, tv, coeffs, ku, kv, transpose)
        worst = 0.0
        for i in range(nu):
            ui = np.full(nv, u[i])
            p = CK.surface_points(srf, *((ui, v) if not transpose else (v, ui)))
            worst = max(worst, float(np.abs(p - grid[i] * MM_PER_M).max()))
            if worst > check_tol_mm:
                break
        if worst <= check_tol_mm:
            if transpose:
                print("grid_to_bspline_surface: NOTE transposed conversion "
                      "was required.")
            return srf
    raise RuntimeError(
        f"OCC B-spline conversion failed self-check (worst dev {worst:.3g} "
        "mm); refusing to export bad geometry.")


def _apply_orientation(grid, name):
    fr, fc, tr = ORIENTATIONS.get(name, (False, False, False))
    g = grid
    if fr:
        g = g[::-1, :, :]
    if fc:
        g = g[:, ::-1, :]
    if tr:
        g = np.transpose(g, (1, 0, 2))
    return np.ascontiguousarray(g)


def _trim_blade_at_hub(blade_shape, hub_radius_m, ring_x_mm):
    """Cut away the blade portion INSIDE the hub cylinder.

    The tool is an analytic cylinder SOLID at the design hub radius along +x
    (tool only; it is never exported). BRepAlgoAPI_Cut(blade, cyl) keeps the
    part of the blade outside the cylinder, so the root-extended band below
    the hub surface is removed and the new root edge is OCC's intersection
    curve of blade and hub cylinder. Units: mm (the faces are built in mm).
    """
    R = hub_radius_m * MM_PER_M
    x_lo = float(ring_x_mm[0]) - 200.0
    x_hi = float(ring_x_mm[1]) + 200.0
    tool = CK.cylinder((x_lo, 0.0, 0.0), (1.0, 0.0, 0.0), R, x_hi - x_lo)
    try:
        out = CK.cut(blade_shape, tool)
    except RuntimeError:
        raise RuntimeError("root trim failed: the cut did not build") from None
    n = CK.count_faces(out)
    if n < 5:
        raise RuntimeError(f"root trim produced only {n} faces; refusing")
    return out


def _write_iges_brep(shape, path):
    """IGES writer in BRep mode (1).

    REQUIRED whenever faces carry trim loops: the default Faces-mode writer
    exports only each face's underlying surface and DISCARDS trimming (the
    documented unbounded-cylinder failure earlier in this project). BRep
    mode writes trimmed faces as MSBO entities, which carry the loops.
    """
    CK.write_iges(shape, path, brep=True)


def X_CAD(grids, x1, output_dir=None, hub=True, hub_height=None,
          hub_center=0.0, n_blades=5, trim_root_at_hub=False):
    """Build the five-surface DRDC blade (plus hub) and write it to IGES.

    grids      : dict from tip_surfaces_new.build_drdc_grids()
    x1         : case id (used for the output filename via pipeline_config)
    hub        : include the Sec. 10 hub sector (default True)
    hub_height : total hub height in metres; None = 2x the root ring's
                 axial span, centred on the ring. The radius is measured
                 from the blade root ring and takes no input.
    n_blades   : Z; the sector spans exactly 2 pi / Z and Z rotated copies
                 reproduce the full hub (TM 2013-178 Sec. 10, Fig. 9).

    ROOT HANDLING. Two settings control it and they MUST agree:

      blade_surface_new.ROOT_EXTENSION
        > 0  the five blade patches start that far BELOW the design root
             radius while the hub keeps the design radius
             (meta['hub_radius']), so blade and hub clearly intersect;
        = 0  the patches start exactly AT the design root radius, so the
             root ring only touches the hub surface tangentially (the two
             splines part by ~1 um between shared samples).

      trim_root_at_hub
        True   cut the overhang off here, against an analytic cylinder,
               and write in IGES BRep mode so the trim loops survive;
        False  export untrimmed and leave any trimming to the mesher.

    Setting trim_root_at_hub=True with ROOT_EXTENSION = 0 is INVALID:
    there is no overhang to remove and the cut runs against a boundary
    coincident with the blade's own root edge, which is a degenerate
    boolean. The default is False, matching ROOT_EXTENSION = 0.

    The hub goes through the SAME grid -> self-verified-B-spline path as
    the blade patches, so it arrives in the IGES as bounded entity-128
    surfaces. Deliberately NOT an analytic Geom_CylindricalSurface: the
    Faces-mode IGES writer discards trimming, and an analytic cylinder then
    arrives unbounded (the earlier "Model Size adjusted to 100000" and
    stack-of-rings failures).
    """
    t_common = grids.get("meta", {}).get("t_common")

    if hub:
        from hub_new import hub_grids
        hgrids, hinfo = hub_grids(grids, hub_height=hub_height,
                                  hub_center=hub_center, n_blades=n_blades)

    faces = []
    for label, key in ANNEX_B_ORDER:
        grid = _apply_orientation(np.asarray(grids[key], dtype=float), key)
        # the wrap grids are sampled at the clustered t_common columns; the
        # spline MUST use those same parameter values
        v_params = None
        if key in ("te_strip", "le_strip", "tip") and t_common is not None:
            v_params = t_common
        srf = grid_to_bspline_surface(grid, v_params=v_params)
        faces.append((label, CK.natural_face(srf, 1.0e-6)))

    if hub:
        for label, key in (("Hub", "hub_sector"),
                           ("Hub cap lo", "hub_cap_lo"),
                           ("Hub cap hi", "hub_cap_hi")):
            srf = grid_to_bspline_surface(hgrids[key])
            faces.append((label, CK.natural_face(srf, 1.0e-6)))

    # sewed, then ShapeFix_Shape
    shape = CK.sew([f for _, f in faces], SEW_TOL)

    paths = cad_output_paths(x1, output_dir)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    iges_path = paths["iges"]

    if trim_root_at_hub:
        hub_radius = grids.get("meta", {}).get("hub_radius")
        if hub_radius is None:
            raise RuntimeError("trim_root_at_hub needs meta['hub_radius'] "
                               "(regenerate grids with the current "
                               "tip_surfaces_new)")
        # The blade must actually overhang the hub, or the cut is
        # degenerate: see the ROOT HANDLING note in this function's
        # docstring. Measured, not assumed, so an inconsistent
        # ROOT_EXTENSION cannot slip through.
        ring_r = np.hypot(
            *np.vstack([np.asarray(grids[k], dtype=float)[0, :, :]
                        for k in ("te_strip", "central_pressure",
                                  "le_strip", "central_suction")])[:, 1:].T)
        overhang = float(hub_radius - ring_r.max())
        if overhang <= 1.0e-6:
            raise RuntimeError(
                f"trim_root_at_hub=True but the blade root ring sits "
                f"{overhang * 1000.0:.4f} mm inside the hub radius: there "
                f"is nothing to trim and the cut would run against the "
                f"blade's own root edge. Set ROOT_EXTENSION > 0 in "
                f"blade_surface_new, or trim_root_at_hub=False.")
        # blade faces only: separate the blade shell from the already-added
        # hub faces by rebuilding the blade shell alone
        blade_shape = CK.sew([f for label, f in faces
                              if label.startswith("Blade")], SEW_TOL)

        ring = np.vstack([np.asarray(grids[k], dtype=float)[0, :, :]
                          for k in ("te_strip", "central_pressure",
                                    "le_strip", "central_suction")])
        ring_x_mm = (ring[:, 0].min() * MM_PER_M,
                     ring[:, 0].max() * MM_PER_M)
        blade_trimmed = _trim_blade_at_hub(blade_shape, hub_radius, ring_x_mm)

        parts = [blade_trimmed]
        if hub:                     # the hub's faces sewed, not fixed
            parts.append(CK.sew([f for label, f in faces
                                 if label.startswith("Hub")], SEW_TOL,
                                fix=False))
        shape = CK.compound(parts)

        _write_iges_brep(shape, iges_path)
        print(f"[X_CAD] blade root TRIMMED at the hub cylinder "
              f"(R = {hub_radius:.6f} m); trimmed blade faces: "
              f"{CK.count_faces(blade_trimmed)}. Written in IGES BRep mode "
              f"(trim loops preserved).")
        print(f"IGES file written (trimmed blade"
              + (" + hub sector + 2 caps" if hub else "") + f"): {iges_path}")
        return shape

    CK.write_iges(shape, iges_path)        # Faces mode: untrimmed surfaces
    n_srf = len(faces)
    print(f"IGES file written ({n_srf} faces: 5 blade, untrimmed"
          + (" + hub sector + 2 caps" if hub else "") + f"): {iges_path}")
    return shape


def X_CAD_from_design(pitch_con, chord_con, x1, output_dir=None,
                      tip_config=None, write_dat=False, verbose=True,
                      hub=True, hub_height=None, hub_center=0.0,
                      n_blades=5, trim_root_at_hub=False,
                      para_control=7, max_pitch=1.4):
    """Convenience wrapper: design vector -> DRDC grids -> IGES.

    Drop-in for workers that previously called
    X_CAD(points, case_id): call X_CAD_from_design(pitch_con, chord_con,
    case_id) instead. para_control and max_pitch go to X_blade (0: the
    original blade).
    """
    from x_blade_new import X_blade
    from tip_surfaces_new import build_drdc_grids, TipConfig

    res = X_blade(pitch_con, chord_con, x1, return_blade_surface=True,
                  write_dat=write_dat, para_control=para_control,
                  max_pitch=max_pitch)
    blade_surface = res[-1]
    constraint_violation = res[2]
    if blade_surface is None or constraint_violation:
        raise RuntimeError(
            f"Case {x1}: design rejected (violation={constraint_violation}); "
            "no CAD generated.")
    grids = build_drdc_grids(blade_surface, tip_config or TipConfig(),
                             verbose=verbose)
    return X_CAD(grids, x1, output_dir=output_dir, hub=hub,
                 hub_height=hub_height, hub_center=hub_center,
                 n_blades=n_blades, trim_root_at_hub=trim_root_at_hub)
