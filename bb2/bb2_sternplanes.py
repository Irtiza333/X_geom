"""
bb2_sternplanes.py

BB2's own stern planes on the vehicles of bb2_build.py. The propeller is optimized in this set-up, so the stern
planes are BB2's (user, 22 Oct 2026), not the wind-tunnel rudder: its planform differs from BB2's (its tip cut at
94.5 deg to its TE against BB2's 90.6, its LE converging on the TE at 12.1 deg against 11.6), so no placement lines
both up (bb2_rudder.py compares them).

The stern plane is the solid of hull.iges at 45 deg; its root runs 150 to 200 mm into the hull. It is copied to
45, 135, 225 and 315 deg (turned about the axis; its sections are symmetric, so the turned copies are also the
mirror images), each cut by the rebuilt hull with the hub (bb2_hull_hub.step) and fused, as that part outside the
hull, with the vehicles (the same union). The part of hull.iges's stern plane outside hull.iges's own hull is the
check: the rebuilt hull lies within 0.96 mm of it. The quadrant y >= 0, z >= 0 (hull.iges's) of the hull, the hub
and the stern plane is written as IGES with solids.

Writes into the output folder (bb2_build.py's, with bb2_vehicle.step, bb2_vehicle6.step and bb2_hull_hub.step):
bb2_sternplane.step/.iges (the stern plane of hull.iges as it is), bb2_vehicle_rudders.step/.iges (one blade),
bb2_vehicle6_rudders.step/.iges (six blades), bb2_hull_hub_rudder_quadrant.iges.

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


def boolean(name, a, b):
    op = XL._occ("BRepAlgoAPI", f"BRepAlgoAPI_{name}")(a, b)
    op.Build()
    if not op.IsDone():
        raise RuntimeError(f"BRepAlgoAPI_{name} failed")
    return op.Shape()


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
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
