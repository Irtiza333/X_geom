"""
_cad_kernel_test.py

Self-check for cad_kernel.py, XGeom's CAD kernel (the one module that calls
OpenCascade): each function on simple shapes whose answers are known (a box, a
cube sewn from six faces, a loft between two squares, a B-spline face, STEP and
IGES written and read back), and that the modules moved over to it call no
OpenCascade themselves. Without pythonocc-core or cadquery-ocp only the second
part runs.

    python _cad_kernel_test.py
"""

import os
import re
import sys
import tempfile

import numpy as np

import cad_kernel as CK

_FAILED = []
HERE = os.path.dirname(os.path.abspath(__file__))
MOVED = ("xcad_loft.py",)          # the modules that reach OpenCascade only through the kernel (27 Nov 2026)


def check(name, got, want, tol):
    got, want = np.asarray(got, float), np.asarray(want, float)
    err = float(np.abs(got - want).max()) if got.size else 0.0
    ok = err <= tol
    if not ok:
        _FAILED.append(name)
    print(f"  {'ok ' if ok else 'FAIL'}  {name:<62s} max error {err:.3e}   (tol {tol:.0e})")


def test_seam():
    print("the seam: the modules moved over call no OpenCascade themselves")
    calls = re.compile(r"\b(_occ|_static|_topods)\(|^\s*(from|import)\s+(OCC|OCP)\b"
                       r"|importlib\.import_module\(f?\"\{?_PKG", re.M)
    found = {}
    for name in MOVED:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            found[name] = len(calls.findall(fh.read()))
    check("no _occ, _static, _topods call and no OCC or OCP import in " + ", ".join(found),
          list(found.values()), [0] * len(found), 0)


def square(z, a=10.0):
    """A square of side a at height z, as a closed wire of four straight edges."""
    p = [(0.0, 0.0, z), (a, 0.0, z), (a, a, z), (0.0, a, z)]
    return CK.wire([CK.line_edge(p[i], p[(i + 1) % 4]) for i in range(4)])


def cube_faces(a=10.0):
    """The six faces of the cube [0, a]^3, each bounded by its own straight edges."""
    c = np.array([[x, y, z] for z in (0.0, a) for y in (0.0, a) for x in (0.0, a)])
    quads = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    return [CK.wire_face([CK.line_edge(c[q[i]], c[q[(i + 1) % 4]]) for i in range(4)]) for q in quads]


def test_kernel(tmp):
    print(f"\nthe kernel's functions ({CK.binding()})")
    b = CK.box((0.0, 0.0, 0.0), (10.0, 20.0, 30.0))
    vol, area = CK.mass_properties(b)
    check("box: volume, area, 6 faces, valid, its bounding box",
          [vol, area, CK.count_faces(b), CK.is_valid(b), *CK.bounding_box(b).ravel()],
          [6000.0, 2200.0, 6, 1, 0, 0, 0, 10, 20, 30], 1e-6)
    inner = CK.common(b, CK.box((5.0, 5.0, 5.0), (50.0, 50.0, 50.0)))
    check("common: the part of the box inside another, 5 x 15 x 25",
          [CK.mass_properties(inner)[0], CK.is_valid(inner)], [1875.0, 1], 1e-6)
    cube, free = CK.sew_solid(cube_faces(), 1.0e-6)
    check("sew_solid: a cube from six plane faces (wire_face), closed, 1000 mm^3",
          [CK.mass_properties(cube)[0], free, CK.count_faces(cube), CK.is_valid(cube)], [1000.0, 0, 6, 1], 1e-6)
    prism = CK.loft([square(0.0), square(5.0)], True, True)
    check("loft: two squares 5 apart, ruled and closed: a 10 x 10 x 5 block",
          [CK.mass_properties(prism)[0], CK.count_faces(prism), CK.is_valid(prism)], [500.0, 6, 1], 1e-6)
    face = CK.planar_face(square(0.0, 4.0))
    check("planar_face: a 4 x 4 square", CK.mass_properties(face)[1], 16.0, 1e-9)
    k = np.array([0.0] * 4 + [1.0] * 4)                         # one cubic span each way: a flat patch
    u = np.linspace(0.0, 1.0, 4)
    grid = np.array([[[3.0 * x, 2.0 * y, 0.0] for x in u] for y in u])
    patch = CK.surface_face(grid, k, k)
    edge = CK.bspline_edge(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]), k, 3)
    check("surface_face: the flat 3 x 2 patch of a 4 x 4 pole grid; bspline_edge: a straight cubic, 3 long (its "
          "box takes in the edge's tolerance, 1e-7)",
          [CK.mass_properties(patch)[1], *CK.bounding_box(edge).ravel()[[0, 3]]], [6.0, 0.0, 3.0], 1e-6)
    step, iges = CK.write_step(cube, os.path.join(tmp, "cube.step")), os.path.join(tmp, "cube.iges")
    CK.write_iges(cube, iges)
    s_back, i_back = CK.read_step(step), CK.read_iges(iges)
    CK.write_iges(cube, os.path.join(tmp, "cube_brep.iges"), brep=True)
    b_back = CK.read_iges(os.path.join(tmp, "cube_brep.iges"))
    check("STEP and IGES written and read back: the cube's volume and faces (IGES as faces, and as a BRep "
          "solid)", [CK.mass_properties(s_back)[0], CK.count_faces(s_back), CK.count_faces(i_back),
                     CK.mass_properties(b_back)[0]], [1000.0, 6, 6, 1000.0], 1e-6)
    verts, tris = CK.tessellate(cube, 0.5, 0.5)
    normals = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
    outward = np.einsum("ij,ij->i", normals, verts[tris].mean(axis=1) - 5.0) > 0.0
    check("tessellate: the cube's triangles, every one facing outward", [len(tris) >= 12, outward.all()], [1, 1], 0)


def main():
    print("cad_kernel self-check")
    test_seam()
    if not CK.available():
        print("\n       the kernel's functions skipped: no pythonocc-core or cadquery-ocp")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            test_kernel(tmp)
    print()
    if _FAILED:
        print(f"{len(_FAILED)} check(s) FAILED: " + ", ".join(_FAILED))
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    sys.exit(main())
