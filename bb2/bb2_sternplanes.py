"""
bb2_sternplanes.py

BB2's own stern planes on the vehicles of bb2_build.py. The propeller is optimized in this set-up, so the stern
planes are BB2's (user, 22 Oct 2026), not the wind-tunnel rudder: its planform differs from BB2's (the angle of its
tip cut to its TE, and how fast its LE converges on the TE), so no placement lines both up (bb2_rudder.py
compares them).

The stern plane is the solid of hull.iges at 45 deg; its root runs into the hull. It is copied to
45, 135, 225 and 315 deg (turned about the axis; its sections are symmetric, so the turned copies are also the
mirror images), each cut by the rebuilt hull with the hub (bb2_hull_hub.step) and fused, as that part outside the
hull, with the vehicles (the same union). The part of hull.iges's stern plane outside hull.iges's own hull is the
check: the rebuilt hull lies within 0.96 mm of it. The quadrant y >= 0, z >= 0 (hull.iges's) of the hull, the hub
and the stern plane is written as IGES with solids, and so is the set-up with the propeller periodic (user, 22 Oct
2026): the quadrant of the hull without the hub and the stern plane, with the one blade on its 1/6 of the hub of
bb2_build.py (bb2_blade_sector.step, blade 1 of prop.stp) as a second solid; the two meet where the hull ends and
the hub begins (x = the hub's forward end), the stationary part and the rotating one. The sector is
turned about the axis so that the strip's face at the hull's end is centred on the quadrant (SECTOR_CENTRE_DEG,
user, 22 Oct 2026: it stuck out of the hull's quarter face): 15 to 75 deg. The strip twists along the hub, so the
blade then reaches past the quadrant, and its clocking against the stern planes is no longer one of
prop.stp's (a whole multiple of 60 deg would keep it).

Writes into the output folder (bb2_build.py's, with bb2_vehicle.step, bb2_vehicle6.step, bb2_hull.step,
bb2_hull_hub.step and bb2_blade_sector.step): bb2_sternplane.step/.iges (the stern plane of hull.iges as it is),
bb2_vehicle_rudders.step/.iges (one blade), bb2_vehicle6_rudders.step/.iges (six blades),
bb2_hull_hub_rudder_quadrant.iges, bb2_hull_rudder_quadrant.step/.iges (the quadrant of the hull without the hub
and the stern plane) and bb2_quadrant_sector.step/.iges (that and the blade sector, two solids).

    python bb2_sternplanes.py hull.iges [out_dir]        (about 2 min)
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                  # the X_geom folder
sys.path.insert(0, HERE)

import bb2_build as B                                      # noqa: E402
import bb2_sections as S                                   # noqa: E402
import vehicle_cad as VC                                   # noqa: E402
import xcad_loft as XL                                     # noqa: E402

E_SPAN = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)         # the stern plane's span direction (45 deg)
SECTOR_CENTRE_DEG = 45.0                                   # the strip's face at the hull's end centred here (deg
                                                           # about the axis, from +y towards +z)


def boolean(name, a, b):
    op = XL._occ("BRepAlgoAPI", f"BRepAlgoAPI_{name}")(a, b)
    op.Build()
    if not op.IsDone():
        raise RuntimeError(f"BRepAlgoAPI_{name} failed")
    return op.Shape()


def end_face_span(sector, x_end, tol=0.01):
    """The angles (deg, about the axis from +y towards +z) where the sector's face at x_end (the strip's piece of
    the hub's forward end) begins and ends, from its edges."""
    pts = [S._polyline(e, 200) for f in S._explore(sector, "FACE")
           if np.ptp(VC._bbox(f)[:, 0]) < tol and abs(VC._bbox(f)[0, 0] - x_end) < tol
           for e in S._explore(f, "EDGE")]
    if not pts:
        raise RuntimeError(f"the sector has no face at x {x_end:.3f} mm")
    p = np.vstack(pts)
    p = p[np.hypot(p[:, 1], p[:, 2]) > 1.0]
    th = np.arctan2(p[:, 2], p[:, 1])
    mid = np.arctan2(np.sin(th).mean(), np.cos(th).mean())
    rel = np.degrees(np.arctan2(np.sin(th - mid), np.cos(th - mid)))
    return np.degrees(mid) + rel.min(), np.degrees(mid) + rel.max()


def one_solid(shape):
    """The shape, checked to be one solid."""
    n = len(S._explore(shape, "SOLID"))
    if n != 1:
        raise RuntimeError(f"the fuse left {n} solids")
    return shape


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    hull_path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "outputs")
    t0 = time.time()
    solids = S._explore(S.read_cad(hull_path), "SOLID")
    width = lambda s: VC._bbox(s)[1, 0] - VC._bbox(s)[0, 0]
    plane, hull_q = min(solids, key=width), max(solids, key=width)
    hull_hub = S._explore(S.read_cad(os.path.join(out, "bb2_hull_hub.step")), "SOLID")[0]
    n, ok, vol = VC.solid_report(plane)
    if n != 1 or not ok:
        raise RuntimeError(f"hull.iges's stern plane: {n} solids, valid {ok}")
    v, _ = XL.tessellate(plane, linear=1.0, angular=0.3)
    s = v @ E_SPAN
    print(f"BB2's stern plane (hull.iges): valid, {vol:.6e} mm^3, x {v[:, 0].min():.1f} to {v[:, 0].max():.1f} mm, "
          f"s {s.min():.1f} to {s.max():.1f} mm")
    XL.write_step(plane, os.path.join(out, "bb2_sternplane.step"))
    XL.write_iges(plane, os.path.join(out, "bb2_sternplane.iges"))

    # the four stern planes outside the rebuilt hull, and the check against hull.iges's own hull
    exposed = [boolean("Cut", VC.transformed(plane, VC.turn_x(0.5 * np.pi * k), np.zeros(3)), hull_hub)
               for k in range(4)]
    vols = [VC.volume(e) for e in exposed]
    own = VC.volume(boolean("Cut", plane, hull_q))
    if max(vols) > 1.001 * min(vols) or max(vols) > 0.999 * vol:
        raise RuntimeError(f"the stern planes' cuts by the hull differ (volumes {vols})")
    print(f"outside the rebuilt hull: {vols[0]:.6e} mm^3 each (the four within {max(vols) / min(vols) - 1:.1e}); "
          f"outside hull.iges's own hull {own:.6e} mm^3 ({vols[0] / own - 1:+.2e})")

    # the vehicles and the quadrant
    for name in ("vehicle", "vehicle6"):
        vehicle = one_solid(VC.fuse([S.read_cad(os.path.join(out, f"bb2_{name}.step"))] + exposed))
        print(f"{name} with BB2's four stern planes: solids, valid, volume mm^3 {VC.solid_report(vehicle)}")
        XL.write_step(vehicle, os.path.join(out, f"bb2_{name}_rudders.step"))
        XL.write_iges(vehicle, os.path.join(out, f"bb2_{name}_rudders.iges"))
    q = B.quadrant(one_solid(VC.fuse([hull_hub, exposed[0]])))
    print(f"hull, hub and stern plane in the quadrant y >= 0, z >= 0: {VC.solid_report(q)}")
    B.write_iges_solids(q, os.path.join(out, "bb2_hull_hub_rudder_quadrant.iges"))

    # the propeller periodic: the quadrant of the hull without the hub and the stern plane, and the blade sector
    hull = S._explore(S.read_cad(os.path.join(out, "bb2_hull.step")), "SOLID")[0]
    q = B.quadrant(one_solid(VC.fuse([hull, exposed[0]])))
    n, ok, vq = VC.solid_report(q)
    if n != 1 or not ok:
        raise RuntimeError(f"the hull and stern plane's quadrant: {n} solids, valid {ok}")
    XL.write_step(q, os.path.join(out, "bb2_hull_rudder_quadrant.step"))
    B.write_iges_solids(q, os.path.join(out, "bb2_hull_rudder_quadrant.iges"))
    sector = S._explore(S.read_cad(os.path.join(out, "bb2_blade_sector.step")), "SOLID")[0]
    ends = VC._bbox(q)[0, 0], VC._bbox(sector)[1, 0]
    if abs(ends[0] - ends[1]) > 0.01:
        raise RuntimeError(f"the hull ends at x {ends[0]:.3f} mm, the hub strip at {ends[1]:.3f}")
    lo, hi = end_face_span(sector, ends[1])
    turn = SECTOR_CENTRE_DEG - 0.5 * (lo + hi)
    sector = VC.transformed(sector, VC.turn_x(np.radians(turn)), np.zeros(3))
    lo2, hi2 = end_face_span(sector, ends[1])
    print(f"blade sector: its strip's face at the hull's end from {lo:.2f} to {hi:.2f} deg (prop.stp's blade 1), "
          f"turned by {turn:.2f} deg to {lo2:.2f} to {hi2:.2f} deg")
    both = VC.compound([q, sector])
    XL.write_step(both, os.path.join(out, "bb2_quadrant_sector.step"))
    B.write_iges_solids(both, os.path.join(out, "bb2_quadrant_sector.iges"))
    v, _ = XL.tessellate(sector, linear=1.0, angular=0.2)
    blade = np.hypot(v[:, 1], v[:, 2]) > 1.1 * S.HUB_R_FWD
    th = np.degrees(np.arctan2(v[blade, 2], v[blade, 1]))
    print(f"hull without the hub and stern plane in the quadrant: valid, {vq:.6e} mm^3; with the blade sector "
          f"({VC.solid_report(sector)[2]:.6e} mm^3; the blade from {th.min():.1f} to {th.max():.1f} deg about the "
          f"axis): {VC.solid_report(both)[:2]}")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
