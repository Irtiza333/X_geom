"""
vehicle_cad.py

The vehicle as one CAD solid (xgeom_tool.py, the Vehicle tab's Vehicle CAD): each component's own CAD, placed
as the Vehicle tab places it (xgeom_vehicle.assemble) and fused, in mm in the hull's frame (x aft from the nose,
y to starboard, z up).

    hull       the hull's STEP solid (hull_modify.build_solid), as it is
    rudder     the rudder's STEP solid (lofted through its XCAD file by xcad_loft; its XCAD frame: x chordwise,
               y the thickness reversed, z the height), with its flat root face extruded down the height by the
               root extrusion's depth (extended_rudder: the prism's sides and end sewn on in place of the root
               face); each rudder is a copy turned, pitched, scaled and moved so that its root LE sits at the
               vehicle's root LE, its chord along ec, its height along eh and its thickness along et (rudder_trsf)
    propeller  the blade's five DRDC faces (the propeller's STEP: blade_modify.write_cad, X_CAD_new) sewn and
               closed at the root by the piece of the hub cylinder inside the root ring (blade_solid); Z copies
               around the axis fused with the hub cylinder (the hub radius and height: propeller_solid); scaled to
               the vehicle's D and moved to the propeller plane
The placed parts are written as <case>_vehicle_parts.step, read back (they fuse about three times faster so) and
fused (BRepAlgoAPI_Fuse with a fuzzy value, then ShapeUpgrade_UnifySameDomain); parts that do not touch stay
separate solids of the result (counted in the report).

run_vehicle (the tool's button) runs each component's own build, writes <case>_vehicle_setup.json (the three
STEP files and the placement in numbers) and builds the vehicle from it in a process of its own, so the window
stays responsive through the fuse:

    python vehicle_cad.py outputs/vehicle/<case>_vehicle_setup.json      the vehicle again from that file

Output (outputs/vehicle): <case>_vehicle.step (the fused solid), <case>_vehicle_parts.step (the parts placed, not
fused) and <case>_vehicle.json (the set-up and the result). Works on pythonocc-core or cadquery-ocp through
xcad_loft's binding helpers; the propeller's own CAD needs pythonocc-core (X_CAD_new).
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import subprocess
import sys
import time

import numpy as np

import xcad_loft as XL

FUZZY_MM = 1.0e-3                  # the fuse's fuzzy value: the sewing tolerance of the blade and the rudder
SEW_MM = 1.0e-3


# --------------------------------------------------------------------------
# Small OCC helpers (either binding)
# --------------------------------------------------------------------------

def _shapes(shape, kind):
    """The sub-shapes of a kind ('SOLID', 'SHELL', 'FACE', 'WIRE', 'EDGE'), each cast to its type."""
    TopExp_Explorer = XL._occ("TopExp", "TopExp_Explorer")
    top = XL._occ("TopAbs", f"TopAbs_{kind}")
    cast = XL._topods(kind.capitalize())
    out, ex = [], TopExp_Explorer(shape, top)
    while ex.More():
        out.append(cast(ex.Current()))
        ex.Next()
    return out


def _trsf(m, t):
    """gp_Trsf of x -> m x + t (m a rotation times a positive scale, t in mm)."""
    gp_Trsf = XL._occ("gp", "gp_Trsf")
    m, t = np.asarray(m, dtype=float), np.asarray(t, dtype=float)
    tr = gp_Trsf()
    tr.SetValues(*(float(v) for v in np.column_stack((m, t)).ravel()))
    return tr


def transformed(shape, m, t):
    """A copy of the shape moved by x -> m x + t."""
    BRepBuilderAPI_Transform = XL._occ("BRepBuilderAPI", "BRepBuilderAPI_Transform")
    return BRepBuilderAPI_Transform(shape, _trsf(m, t), True).Shape()


def turn_x(angle):
    """The rotation by angle (rad) about +x (y towards z), as xgeom_vehicle turns the blades."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _bbox(shape):
    Bnd_Box = XL._occ("Bnd", "Bnd_Box")
    mod = importlib.import_module(f"{XL._PKG}.BRepBndLib")
    owner = getattr(mod, "brepbndlib", None) or getattr(mod, "BRepBndLib")
    b = Bnd_Box()
    XL._static(owner, "AddOptimal")(shape, b, False, False)        # the surfaces, not their control nets
    return np.array(b.Get(), dtype=float).reshape(2, 3)


def is_valid(shape):
    BRepCheck_Analyzer = XL._occ("BRepCheck", "BRepCheck_Analyzer")
    return bool(BRepCheck_Analyzer(shape).IsValid())


def _sewn_solid(faces, tol=SEW_MM):
    """One solid from faces that close a volume (sewing, ShapeFix)."""
    Sewing, MakeSolid = XL._occ("BRepBuilderAPI", "BRepBuilderAPI_Sewing", "BRepBuilderAPI_MakeSolid")
    ShapeFix_Shell, ShapeFix_Solid = XL._occ("ShapeFix", "ShapeFix_Shell", "ShapeFix_Solid")
    sew = Sewing(tol)
    for f in faces:
        sew.Add(f)
    sew.Perform()
    shells = _shapes(sew.SewedShape(), "SHELL")
    if len(shells) != 1:
        raise RuntimeError(f"sewing left {len(shells)} shells, expected 1")
    fs = ShapeFix_Shell(shells[0])
    fs.Perform()
    solid = MakeSolid(fs.Shell()).Solid()
    fx = ShapeFix_Solid(solid)
    fx.Perform()
    return fx.Solid(), int(sew.NbFreeEdges())


def fuse(shapes, fuzzy=FUZZY_MM):
    """The union of the shapes (the first the argument, the others the tools), same-domain faces unified."""
    TopTools_ListOfShape = XL._occ("TopTools", "TopTools_ListOfShape")
    BRepAlgoAPI_Fuse = XL._occ("BRepAlgoAPI", "BRepAlgoAPI_Fuse")
    ShapeUpgrade_UnifySameDomain = XL._occ("ShapeUpgrade", "ShapeUpgrade_UnifySameDomain")
    args, tools = TopTools_ListOfShape(), TopTools_ListOfShape()
    args.Append(shapes[0])
    for s in shapes[1:]:
        tools.Append(s)
    op = BRepAlgoAPI_Fuse()
    op.SetArguments(args)
    op.SetTools(tools)
    op.SetFuzzyValue(float(fuzzy))
    op.SetRunParallel(True)
    op.Build()
    errors = getattr(op, "HasErrors", None)                     # not in every binding
    if not op.IsDone() or (errors is not None and errors()):
        raise RuntimeError("the boolean fuse failed")
    unify = ShapeUpgrade_UnifySameDomain(op.Shape(), True, True, False)
    unify.Build()
    return unify.Shape()


# --------------------------------------------------------------------------
# The rudder
# --------------------------------------------------------------------------

def _root_face(solid, axis=2):
    """The rudder's flat root face: the planar face lowest along axis."""
    BRep_Tool = XL._occ("BRep", "BRep_Tool")
    GeomAdaptor_Surface = XL._occ("GeomAdaptor", "GeomAdaptor_Surface")
    GeomAbs_Plane = XL._occ("GeomAbs", "GeomAbs_Plane")
    surface = XL._static(BRep_Tool, "Surface")
    best, low = None, np.inf
    for f in _shapes(solid, "FACE"):
        if GeomAdaptor_Surface(surface(f)).GetType() != GeomAbs_Plane:
            continue
        mid = _bbox(f).mean(axis=0)[axis]
        if mid < low:
            best, low = f, mid
    if best is None:
        raise RuntimeError("the rudder has no flat root face")
    return best


def extended_rudder(solid, depth, axis=2):
    """The rudder with its root extrusion, one solid: its faces but the root face, sewn to the sides and the
    end of the root face's prism (depth mm down the axis); the same as the union of the two, without a
    boolean."""
    BRepPrimAPI_MakePrism = XL._occ("BRepPrimAPI", "BRepPrimAPI_MakePrism")
    gp_Vec = XL._occ("gp", "gp_Vec")
    root = _root_face(solid, axis)
    v = [0.0, 0.0, 0.0]
    v[axis] = -float(depth)
    mk = BRepPrimAPI_MakePrism(root, gp_Vec(*v))
    first, last = mk.FirstShape(), mk.LastShape()
    sides = [f for f in _shapes(mk.Shape(), "FACE") if not (f.IsSame(first) or f.IsSame(last))]
    faces = [f for f in _shapes(solid, "FACE") if not f.IsSame(root)] + sides + _shapes(last, "FACE")
    out, free = _sewn_solid(faces)
    if free:
        raise RuntimeError(f"the rudder with its root extrusion has {free} free edges")
    return out


def rudder_trsf(frame, scale, xa, y0):
    """(m, t) taking the rudder's XCAD frame (mm: x chordwise, y the thickness reversed, z the height) to the
    vehicle (mm): the working frame (x, z, -y) of xgeom_vehicle.place_rudder, its root LE (xa, y0) to the frame's
    a, x along ec, the height along eh, the thickness along et, sizes times scale."""
    a, ec, eh, et = (np.asarray(v, dtype=float) for v in frame)
    m = scale * np.column_stack((ec, -et, eh))
    t = 1000.0 * a - scale * (xa * ec + y0 * eh)
    return m, t


# --------------------------------------------------------------------------
# The propeller
# --------------------------------------------------------------------------

def blade_solid(cad_shape, hub_radius):
    """The blade as a solid (mm, the blade's frame): the five DRDC blade faces of blade_modify.write_cad's shape
    (the shell that is not the hub sector) closed at the root by the piece of the hub cylinder (radius
    hub_radius, mm) inside the root ring, which lies on that cylinder."""
    gp_Pnt, gp_Dir, gp_Ax3 = XL._occ("gp", "gp_Pnt", "gp_Dir", "gp_Ax3")
    Geom_CylindricalSurface = XL._occ("Geom", "Geom_CylindricalSurface")
    MakeFace = XL._occ("BRepBuilderAPI", "BRepBuilderAPI_MakeFace")
    ShapeFix_Face = XL._occ("ShapeFix", "ShapeFix_Face")
    ShapeAnalysis_FreeBounds = XL._occ("ShapeAnalysis", "ShapeAnalysis_FreeBounds")
    shells = _shapes(cad_shape, "SHELL")
    reach = [_bbox(s) for s in shells]
    k = int(np.argmax([np.abs(b[:, 1:]).max() for b in reach]))      # the blade reaches out to the tip
    blade = shells[k]
    faces = _shapes(blade, "FACE")
    wires = _shapes(ShapeAnalysis_FreeBounds(blade, SEW_MM, False, False).GetClosedWires(), "WIRE")
    if len(wires) != 1:
        raise RuntimeError(f"the blade's open boundary has {len(wires)} loops, expected 1 (the root ring)")
    ring = _bbox(wires[0])
    mid = ring.mean(axis=0)
    away = -mid[1:] / max(np.hypot(*mid[1:]), 1e-12)                  # the cylinder's seam opposite the root
    surf = Geom_CylindricalSurface(gp_Ax3(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0),
                                          gp_Dir(0.0, float(away[0]), float(away[1]))), float(hub_radius))
    mk = MakeFace(surf, wires[0], True)
    if not mk.IsDone():
        raise RuntimeError("the blade's root face on the hub cylinder failed")
    fix = ShapeFix_Face(mk.Face())
    fix.Perform()
    solid, free = _sewn_solid(faces + [fix.Face()])
    if free:
        raise RuntimeError(f"the blade closed at its root still has {free} free edges")
    return solid


def propeller_solid(blade, n_blades, hub_radius, hub_height, hub_center=0.0):
    """Z copies of the blade solid around +x (360/Z apart, as the Vehicle tab draws them) fused with the hub
    cylinder (radius, height and centre in mm)."""
    gp_Pnt, gp_Dir, gp_Ax2 = XL._occ("gp", "gp_Pnt", "gp_Dir", "gp_Ax2")
    BRepPrimAPI_MakeCylinder = XL._occ("BRepPrimAPI", "BRepPrimAPI_MakeCylinder")
    hub = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(float(hub_center - 0.5 * hub_height), 0.0, 0.0),
                                          gp_Dir(1.0, 0.0, 0.0)), float(hub_radius), float(hub_height)).Solid()
    blades = [transformed(blade, turn_x(2.0 * np.pi * k / n_blades), np.zeros(3)) for k in range(int(n_blades))]
    return fuse([hub] + blades)


# --------------------------------------------------------------------------
# The vehicle
# --------------------------------------------------------------------------

def volume(shape, eps=1.0e-5):
    """The volume (mm^3), adaptive integration to the relative error eps."""
    GProp_GProps = XL._occ("GProp", "GProp_GProps")
    mod = importlib.import_module(f"{XL._PKG}.BRepGProp")
    fn = XL._static(getattr(mod, "brepgprop", None) or getattr(mod, "BRepGProp"), "VolumeProperties")
    props = GProp_GProps()
    try:
        fn(shape, props, float(eps))
    except TypeError:                                           # a binding without the eps overload
        fn(shape, props)
    return abs(props.Mass())


def solid_report(shape):
    """(number of solids, valid, volume mm^3)."""
    return len(_shapes(shape, "SOLID")), is_valid(shape), volume(shape)


def place_parts(hull, rudder, propeller, setup):
    """The parts placed in the vehicle (mm, the hull's frame), by name: 'hull'; 'rudder 1' .. with their root
    prisms; 'propeller'. rudder: the rudder's solid (XCAD frame, mm); propeller: propeller_solid's (the blade
    frame, mm); setup: vehicle_setup's placement numbers. A part that is None is left out."""
    out = {}
    if hull is not None:
        out["hull"] = hull
    ru = setup.get("rudder")
    if rudder is not None and ru:
        cache = {}
        for k, (frame, depth) in enumerate(zip(ru["frames"], ru["depth"])):
            d = round(1000.0 * depth / ru["scale"], 9)                # the extrusion in the rudder's own mm
            if d not in cache:
                cache[d] = extended_rudder(rudder, d)
            out[f"rudder {k + 1}"] = transformed(cache[d], *rudder_trsf(frame, ru["scale"], ru["xa"], ru["y0"]))
    pr = setup.get("propeller")
    if propeller is not None and pr:
        out["propeller"] = transformed(propeller, pr["scale"] * np.eye(3), [1000.0 * pr["x"], 0.0, 0.0])
    return out


def compound(shapes):
    BRep_Builder = XL._occ("BRep", "BRep_Builder")
    TopoDS_Compound = XL._occ("TopoDS", "TopoDS_Compound")
    c, b = TopoDS_Compound(), BRep_Builder()
    b.MakeCompound(c)
    for s in shapes:
        b.Add(c, s)
    return c


def build_vehicle(out_dir, case, placed):
    """Fuse the placed parts and write <case>_vehicle.step (the fused solid) and <case>_vehicle_parts.step (the
    parts, placed, not fused). Returns the paths and the report."""
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    names = list(placed)
    paths = {"parts": XL.write_step(compound([placed[n] for n in names]),
                                    os.path.join(out_dir, f"{case}_vehicle_parts.step"))}
    parts = _shapes(XL.read_step(paths["parts"]), "SOLID")       # as written: they fuse about 3 times faster
    if len(parts) != len(names):
        raise RuntimeError(f"{paths['parts']} holds {len(parts)} solids, not {len(names)}")
    shape = fuse(parts) if len(parts) > 1 else parts[0]
    n, valid, vol = solid_report(shape)
    paths["vehicle"] = XL.write_step(shape, os.path.join(out_dir, f"{case}_vehicle.step"))
    return {"paths": paths, "shape": shape, "solids": n, "valid": valid, "volume": vol, "parts": names,
            "seconds": time.time() - t0}


ORDER = ("hull", "rudder", "blade")                 # the components' builds, the quick ones first
NAMES = {"hull": "hull", "rudder": "rudder", "blade": "propeller"}


def vehicle_setup(case, steps, parts, info, placement):
    """What the vehicle's CAD needs, in numbers (JSON): the case, the components' STEP files (steps, by kind),
    the placement, and as xgeom_vehicle.assemble has them the rudders' frames, root extrusions and root LE in
    their own frame, and the propeller's plane, scale, blades and hub (mm in its own frame)."""
    out = {"case": case, "written": time.strftime("%d %b %Y %H:%M"),
           "units": "mm, the hull's frame (x aft from the nose, y to starboard, z up)",
           "steps": {NAMES[k]: os.path.abspath(v) for k, v in steps.items()},
           "placement": dataclasses.asdict(placement)}
    ru = parts.get("rudder")
    if "rudder" in steps and ru is not None and "rudder_frames" in info:
        root = np.asarray(ru["root"], dtype=float)
        n = len(info["rudder_frames"])
        out["rudder"] = {"xa": float(root[:, 0].min()), "y0": float(root[:, 1].mean()),
                         "scale": float(placement.rudder_scale),
                         "frames": [[np.asarray(v, dtype=float).tolist() for v in f] for f in info["rudder_frames"]],
                         "depth": [float(v) for v in info["rudder_depth"]], "x_te": float(info["rudder_x_te"]),
                         "angle_deg": [(placement.rudder_angle + 360.0 * k / n) % 360.0 for k in range(n)],
                         **{q: [float(v) for v in info[f"rudder_{q}"]] for q in ("x_le", "rho_le", "rho_te", "pitch")}}
    b = parts.get("blade")
    if "blade" in steps and b is not None and "prop_x" in info:
        out["propeller"] = {"x": float(info["prop_x"]), "d": float(info["prop_d"]),
                            "scale": float(info["prop_scale"]), "blades": int(b["blades"]),
                            "hub_radius_mm": 1000.0 * float(b["hub"][0]), "hub_height_mm": 1000.0 * float(b["hub"][1])}
    return out


def build_from_setup(setup, out_dir, progress=print):
    """The vehicle from a vehicle_setup record: the parts as solids, placed and fused; writes the STEP files
    and <case>_vehicle.json (the set-up and the result). Returns that record."""
    case, steps = setup["case"], setup["steps"]
    progress("the parts as solids")
    hull = _shapes(XL.read_step(steps["hull"]), "SOLID")[0]
    rudder = _shapes(XL.read_step(steps["rudder"]), "SOLID")[0] if "rudder" in steps and "rudder" in setup else None
    propeller = None
    pr = setup.get("propeller")
    if "propeller" in steps and pr:
        progress(f"the propeller as a solid (the blade closed at its root, {pr['blades']} blades and the hub)")
        blade = blade_solid(XL.read_step(steps["propeller"]), pr["hub_radius_mm"])
        propeller = propeller_solid(blade, pr["blades"], pr["hub_radius_mm"], pr["hub_height_mm"])
    progress("the parts placed")
    placed = place_parts(hull, rudder, propeller, setup)
    progress(f"the fuse of {len(placed)} parts")
    out = build_vehicle(out_dir, case, placed)
    rec = dict(setup, parts=out["parts"], solids=out["solids"], valid=out["valid"], volume_mm3=out["volume"],
               fuse_seconds=out["seconds"], files=dict(out["paths"]))
    rec["files"]["json"] = os.path.join(out_dir, f"{case}_vehicle.json")
    with open(rec["files"]["json"], "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def run_vehicle(out_dir, case, tasks, parts, info, placement, progress=print, separate=True):
    """The Vehicle CAD: every component's own build (tasks: the adapters' build_task(case) by kind; each writes
    its case and its CAD into its own folder), then <case>_vehicle_setup.json in out_dir and the vehicle built
    from it (build_from_setup), in a process of its own unless separate is False. parts, info and placement as
    xgeom_vehicle.assemble had them when the build was asked for. progress(text) is told what runs. Returns a
    dict as the adapters' builds do: report, message, error, warn, view (the vehicle's STEP), and the record."""
    t0 = time.time()
    steps = {}
    for k in ORDER:
        if k in tasks:
            progress(f"the {NAMES[k]}'s case and CAD")
            out = tasks[k]()
            if out.get("error") or not out.get("view"):
                raise RuntimeError(f"the {NAMES[k]}: {out.get('message', 'no CAD')}")
            steps[k] = out["view"]
    os.makedirs(out_dir, exist_ok=True)
    setup = vehicle_setup(case, steps, parts, info, placement)
    path = os.path.join(out_dir, f"{case}_vehicle_setup.json")
    with open(path, "w") as fh:
        json.dump(setup, fh, indent=1)
    if separate:
        here = os.path.dirname(os.path.abspath(__file__))
        proc = subprocess.Popen([sys.executable, os.path.join(here, "vehicle_cad.py"), path, "--out", out_dir],
                                cwd=here, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lines = []
        for line in proc.stdout:
            lines.append(line.rstrip())
            if line.startswith("progress: "):
                progress(line[len("progress: "):].strip())
        if proc.wait() != 0:
            tail = [q for q in lines if q.strip()][-12:]
            raise RuntimeError("the vehicle's CAD failed (vehicle_cad.py " + os.path.basename(path) + "):\n"
                               + "\n".join(tail))
        with open(os.path.join(out_dir, f"{case}_vehicle.json")) as fh:
            rec = json.load(fh)
    else:
        rec = build_from_setup(setup, out_dir, progress)
    one, files = rec["solids"] == 1, rec["files"]
    msg = (f"{os.path.basename(files['vehicle'])} ({'one solid' if one else str(rec['solids']) + ' solids'}, "
           f"{'valid' if rec['valid'] else 'NOT valid'}, V {rec['volume_mm3'] / 1e9:.5f} m^3, mm), "
           f"{os.path.basename(files['parts'])} (the parts placed, not fused); each component's CAD in its own "
           f"folder; OCC viewer shows the vehicle")
    if not one:
        msg += f"; warning: {rec['solids']} separate solids (a part does not touch the hull)"
    report = [f"vehicle {case}: {', '.join(rec['parts'])} fused in {rec['fuse_seconds']:.1f} s; {rec['solids']} "
              f"solid(s), {'valid' if rec['valid'] else 'NOT valid'}, V {rec['volume_mm3'] / 1e9:.6f} m^3",
              "  components: " + ", ".join(f"{k} {v}" for k, v in rec["steps"].items()),
              "  written: " + ", ".join([path] + list(files.values())), f"  in {time.time() - t0:.1f} s"]
    return {"report": report, "message": msg, "error": False, "warn": not (one and rec["valid"]),
            "view": files["vehicle"], "record": rec}


def main():
    ap = argparse.ArgumentParser(description="The vehicle's CAD from a <case>_vehicle_setup.json (the Vehicle CAD "
                                             "of xgeom_tool.py writes it): the parts placed and fused.")
    ap.add_argument("setup", help="<case>_vehicle_setup.json")
    ap.add_argument("--out", default=None, help="output folder (default: the set-up's folder)")
    args = ap.parse_args()
    with open(args.setup) as fh:
        setup = json.load(fh)
    out_dir = args.out or os.path.dirname(os.path.abspath(args.setup))
    rec = build_from_setup(setup, out_dir, lambda text: print(f"progress: {text}", flush=True))
    print(f"done: {rec['files']['vehicle']}: {rec['solids']} solid(s), {'valid' if rec['valid'] else 'NOT valid'}, "
          f"V {rec['volume_mm3'] / 1e9:.6f} m^3", flush=True)


if __name__ == "__main__":
    main()
