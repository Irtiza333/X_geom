"""
_cad_compare.py

The samples' CAD, recorded so a change to the CAD code can be checked for
changing nothing (27 Nov 2026, the CAD kernel: cad_kernel.py). Each case is
built the way the design tool builds it and read back from its STEP file: its
faces, volume, area, validity and bounding box, and a digest of the STEP
file's DATA section (its header carries the time) and of an IGES file's
directory and parameter lines.

    python _cad_compare.py before.json           record (about 5 minutes)
    python _cad_compare.py after.json            ... after the change, on the same machine
    python _cad_compare.py --diff before.json after.json

The cases: the rudder modified below H (one loft, two lofts, the half body),
the wing, the fin, the planar rudders (rudder.txt, the wind-tunnel rudder with
its tilted tip and cap), the blended wing body, SUBOFF's hull, the MSc blade
(the DRDC sector, the whole propeller as a solid, a cone hub's sector). Needs
pythonocc-core or cadquery-ocp. The digests depend on the OCCT version: compare
records made on one machine.
"""

import hashlib
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def step_digest(path):
    with open(path, errors="replace") as fh:
        text = fh.read()
    return hashlib.sha1(text[text.index("DATA;"):].encode()).hexdigest()


def iges_digest(path):
    with open(path, errors="replace") as fh:
        keep = [ln[:72] for ln in fh.read().splitlines() if len(ln) >= 73 and ln[72] in "DP"]
    return hashlib.sha1("\n".join(keep).encode()).hexdigest()


def measure(step):
    import cad_kernel as CK
    shape = CK.read_step(step)
    vol, area = CK.mass_properties(shape)
    rec = {"faces": CK.count_faces(shape), "volume_mm3": round(vol, 6), "area_mm2": round(area, 6),
           "valid": CK.is_valid(shape), "box_mm": np.round(CK.bounding_box(shape), 6).tolist(),
           "step_sha1": step_digest(step)}
    iges = os.path.splitext(step)[0] + ".iges"
    if os.path.exists(iges):
        rec["iges_sha1"] = iges_digest(iges)
    return rec


def record(path):
    os.chdir(HERE)                                             # the adapters' files are relative to this folder
    sys.path.insert(0, HERE)
    import xgeom_blade as XBL
    import xgeom_hull as XH
    import xgeom_rudder as XR
    import xgeom_wing as XW
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        def run(name, fn):
            t0 = time.time()
            try:
                rec = fn()
            except Exception as exc:                            # noqa: BLE001 - recorded, the rest still run
                rec = {"error": f"{type(exc).__name__}: {exc}"}
            rec["seconds"] = round(time.time() - t0, 1)
            out[name] = rec
            print(f"{name}: " + ", ".join(f"{k} {v}" for k, v in rec.items() if k != "box_mm"), flush=True)

        def rudder(case, half=False, g0=False):
            ad = XR.RudderAdapter(out_dir=tmp)
            if g0:                                              # not tangent at H: two lofts
                ad.set_tangent_all(False)
                ad.set("LE.dx3", -10.0)
            return measure(ad.build_xcad(case, plot=False, cad=True, half=half)["step"])

        def built(ad, case, **kw):
            ad.out_dir = tmp
            res = ad.build_task(case, **kw)()
            if res.get("error") or not res.get("view"):
                raise RuntimeError(res.get("message", "no CAD"))
            return measure(res["view"])

        def blade(part, case, cone=False):
            ad = XBL.BladeAdapter(params=os.path.join(HERE, "msc_blade.txt"))
            if cone:
                ad.set_hub_shape("cone")
            return built(ad, case, part=part)

        run("rudder_one_loft", lambda: rudder("r1"))
        run("rudder_two_lofts", lambda: rudder("r2", g0=True))
        run("rudder_half", lambda: rudder("r3", half=True))
        run("wing", lambda: built(XW.WingAdapter(params=os.path.join(HERE, "wing.txt")), "w"))
        run("fin", lambda: built(XW.FinAdapter(params=os.path.join(HERE, "fin.txt")), "f"))
        run("rudder_planar", lambda: built(XW.PlanarRudderAdapter(params=os.path.join(HERE, "rudder.txt")), "pr"))
        run("wind_tunnel_rudder", lambda: built(XW.PlanarRudderAdapter(
            params=os.path.join(HERE, "outputs", "wind_tunnel_rudder.txt")), "wt"))
        run("bwb", lambda: built(XW.BwbAdapter(params=os.path.join(HERE, "bwb_glider.txt")), "bwb"))
        run("hull", lambda: built(XH.HullAdapter(params=os.path.join(HERE, "suboff_hull.txt")), "h"))
        run("blade_sector_drdc", lambda: blade("sector", "bs"))
        run("blade_whole", lambda: blade("whole", "bw"))
        run("blade_cone_sector", lambda: blade("sector", "bc", cone=True))
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def diff(path_a, path_b):
    with open(path_a) as fh:
        a = json.load(fh)
    with open(path_b) as fh:
        b = json.load(fh)
    same = 0
    for k in a:
        ka = {x: y for x, y in a[k].items() if x != "seconds"}
        kb = {x: y for x, y in b.get(k, {}).items() if x != "seconds"}
        if ka == kb and "error" not in ka:
            same += 1
            print(f"  same      {k}: {ka['faces']} faces, volume {ka['volume_mm3']} mm^3, STEP data "
                  f"{ka['step_sha1'][:10]}" + (f", IGES {ka['iges_sha1'][:10]}" if "iges_sha1" in ka else ""))
        else:
            print(f"  DIFFERENT {k}: " + "; ".join(f"{x} {ka.get(x)} -> {kb.get(x)}" for x in sorted(set(ka) | set(kb))
                                                   if ka.get(x) != kb.get(x)))
    print(f"{same} of {len(a)} cases the same")
    return same == len(a)


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--diff":
        raise SystemExit(0 if diff(sys.argv[2], sys.argv[3]) else 1)
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    record(sys.argv[1])


if __name__ == "__main__":
    main()
