"""
_airfoils_test.py

Self-check for airfoils.py: the NACA 4- and 5-digit equations, the Selig and
Lednicer file formats, normalization, the lookup by name, the section tables
for para.py (they give the airfoil back), the install from an archive, and,
when the UIUC database is installed in airfoils/, every file in it.

    python _airfoils_test.py
"""

import os
import sys
import tempfile
import zipfile

import numpy as np
from scipy.interpolate import PchipInterpolator

import airfoils as AF

_FAILED = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(name, got, want, tol):
    got, want = np.asarray(got, float), np.asarray(want, float)
    err = float(np.abs(got - want).max()) if got.size else 0.0
    ok = err <= tol
    if not ok:
        _FAILED.append(name)
    print(f"  {'ok ' if ok else 'FAIL'}  {name:<66s} max error {err:.3e}   (tol {tol:.0e})")


def refused(fn, *args):
    try:
        fn(*args)
        return 0
    except ValueError:
        return 1


def given_back(xy, path):
    """Largest difference between the airfoil and the section that para.py
    builds from its table at the airfoil's own t/c and f/c (the table's
    stations, chord fractions)."""
    tab = np.loadtxt(path, delimiter=",", skiprows=1)
    yc, _, _, t_max, _ = AF.shape(xy)
    cmax = tab[:, 1].max()
    cam = (tab[:, 1] / cmax if cmax > 0 else 0.0 * tab[:, 1]) * cmax / 100.0
    th = tab[:, 3] * t_max * np.cos(np.arctan(tab[:, 2]))
    up, lo = AF.surfaces(xy)
    fu = PchipInterpolator(np.sqrt(np.clip(up[:, 0], 0, None)), up[:, 1])
    fl = PchipInterpolator(np.sqrt(np.clip(lo[:, 0], 0, None)), lo[:, 1])
    q = np.sqrt(tab[:, 0])
    return max(np.abs(cam + th - fu(q)).max(), np.abs(cam - th - fl(q)).max())


def test_naca():
    print("\nNACA equations")
    x = np.linspace(0.0, 1.0, 4001)
    out = {}
    for name in ("naca2416", "naca23012", "naca0012"):
        yc, _, th, t_max, c_max = AF.shape(AF.normalize(AF.naca(name, 161)), x)
        out[name] = (t_max, x[np.argmax(th)], c_max, x[np.argmax(yc)], np.abs(yc).max())
    check("naca2416: thickness 0.16 at x 0.30, camber 0.02 at x 0.40",
          [out["naca2416"][0], out["naca2416"][1], out["naca2416"][2], out["naca2416"][3]],
          [0.16, 0.30, 0.02, 0.40], 0.01)
    check("naca23012: thickness 0.12 at x 0.30, camber 0.0184 at x 0.15",
          [out["naca23012"][0], out["naca23012"][1], out["naca23012"][2], out["naca23012"][3]],
          [0.12, 0.30, 0.0184, 0.15], 0.01)
    check("naca0012: no camber", out["naca0012"][4], 0.0, 1e-12)
    a = AF.naca("naca2416")
    check("names: 'NACA 2416', 'naca-2416', 'naca2416.dat' are naca2416",
          [np.abs(AF.naca(n) - a).max() for n in ("NACA 2416", "naca-2416", "naca2416.dat")], 0.0, 0.0)
    check("not NACA names: none; a 5-digit camber line that does not exist refused",
          [AF.naca("clarky") is None, AF.naca("naca241") is None, refused(AF.naca, "naca26012")], [1, 1, 1], 0)


def test_files(tmp):
    print("\ncoordinate files")
    xy = AF.naca("naca2412", 61)
    p = AF.write_file(os.path.join(tmp, "s.dat"), xy, "NACA 2412")
    desc, back = AF.read_file(p)
    up, lo = xy[:61][::-1], xy[60:]
    with open(os.path.join(tmp, "l.dat"), "w") as fh:
        fh.write("NACA 2412 (Lednicer)\n61. 61.\n\n" + "\n".join(f"{x:.6f} {y:.6f}" for x, y in up)
                 + "\n\n" + "\n".join(f"{x:.6f} {y:.6f}" for x, y in lo) + "\n")
    _, led = AF.read_file(os.path.join(tmp, "l.dat"))
    check("Selig file round trip (6 decimals); Lednicer file read in Selig order",
          [desc == "NACA 2412", np.abs(back - xy).max(), led.shape == xy.shape, np.abs(led - xy).max()],
          [1, 0, 1, 0], 1e-6)
    sym = AF.naca("naca0012", 61)
    a = np.deg2rad(3.0)
    moved = (sym @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]).T) * 250.0 + [40.0, -7.0]
    check("normalize: an airfoil moved, turned by 3 deg and scaled x250 comes back; a normalized one is kept",
          [np.abs(AF.normalize(moved) - sym).max(), np.abs(AF.normalize(xy) - xy).max()], 0.0, 1e-9)
    b = np.deg2rad(0.1)
    off = (sym - [1.0, 0.0]) @ np.array([[np.cos(b), np.sin(b)], [-np.sin(b), np.cos(b)]]).T + [1.0, 0.0]
    check("normalize: the LE 1.7e-3 off the origin (turned 0.1 deg about the TE, as s1223) is put at (0, 0)",
          np.abs(AF.normalize(off) - sym).max(), 0.0, 1e-9)


def test_tables(tmp):
    print("\nsection tables (para.py's format)")
    from para import para
    worst = {}
    for name in ("naca2416", "naca23012", "naca4412", "naca0012"):
        p = AF.section_table(name, folder=tmp)
        tab = np.loadtxt(p, delimiter=",", skiprows=1)
        worst[name] = (tab, given_back(AF.normalize(AF.naca(name, 81)), p))
    tab = worst["naca2416"][0]
    check("26 rows at airfoil_data_fixed.csv's stations, the LE row 0, thickness peak 0.5",
          [tab.shape[0], np.abs(tab[:, 0] - AF.STATIONS).max(), np.abs(tab[0]).max(), tab[:, 3].max()],
          [26, 0, 0, 0.5], 1e-3)
    check("each table gives its airfoil back at its own t/c and f/c (chord fractions)",
          [worst[n][1] for n in worst], 0.0, 5e-5)
    check("naca0012: the camber columns are 0", np.abs(worst["naca0012"][0][:, 1:3]).max(), 0.0, 0.0)
    fn = lambda v: (lambda r: np.full(np.shape(r), v, dtype=float))          # noqa: E731
    r = np.linspace(0.2, 1.0, 9)
    for name, fc in (("naca2416", 0.02), ("naca0012", 0.02)):
        pts = para(fn(fc), fn(1.0), fn(0.2), fn(0.12), fn(0.0), fn(0.0), r, 0, write_dat=False,
                   airfoil_path=os.path.join(tmp, "sections", f"{name}.dat"))
        worst[name + "_para"] = (pts.shape, bool(np.all(np.isfinite(pts))))
    check("para.py runs on the tables: 53 points per section, finite (also the symmetric one)",
          [worst["naca2416_para"][0][0], worst["naca2416_para"][1], worst["naca0012_para"][1]], [9 * 53, 1, 1], 0)


def test_lookup(tmp):
    print("\nlookup by name")
    folder = os.path.join(tmp, "db")
    os.makedirs(folder)
    AF.write_file(os.path.join(folder, "myfoil.dat"), AF.naca("naca4415", 41), "MY FOIL")
    AF.write_file(os.path.join(folder, "naca2412.dat"), AF.naca("naca2412", 41), "NACA 2412 (file)")
    _, src1 = AF.coordinates("MyFoil", folder)
    _, src2 = AF.coordinates("naca2412", folder)
    _, src3 = AF.coordinates("naca2416", folder)
    try:
        AF.coordinates("myfol", folder)
        msg = ""
    except ValueError as exc:
        msg = str(exc)
    check("a file by name (any case), a file before the NACA equations, the equations, close names",
          [src1.endswith("myfoil.dat"), src2.endswith("naca2412.dat"), src3 == "NACA 4-digit equations",
           "close: myfoil" in msg], [1, 1, 1, 1], 0)
    zp = os.path.join(tmp, "coord_seligFmt.zip")
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("coord_seligFmt/a18.dat", "A18\n1 0\n0.5 0.05\n0 0\n0.5 -0.03\n1 0\n")
        zf.writestr("coord_seligFmt/E387.dat", open(os.path.join(folder, "myfoil.dat")).read())
        zf.writestr("coord_seligFmt/readme.txt", "not a coordinate file")
    inst = os.path.join(tmp, "inst")
    n, bad = AF.install(zp, inst)
    index = open(os.path.join(inst, "index.dat")).read().splitlines()
    check("install: the .dat files (lower case), index.dat, README.md; the airfoils found by name",
          [n, len(bad), len(index), os.path.exists(os.path.join(inst, "e387.dat")),
           os.path.exists(os.path.join(inst, "README.md")), AF.coordinates("E387", inst)[1].endswith("e387.dat")],
          [2, 0, 3, 1, 1, 1], 0)
    src, inst2 = os.path.join(tmp, "coord_seligFmt"), os.path.join(tmp, "inst2")
    os.makedirs(src)
    AF.write_file(os.path.join(src, "S1223.dat"), AF.naca("naca4415", 41), "S1223")
    n2, _ = AF.install(src, inst2)
    n3, bad3 = AF.install(inst2, inst2)
    check("install from the unpacked folder (copied, lower case), then in place (index and README only)",
          [n2, os.path.exists(os.path.join(inst2, "s1223.dat")), n3, len(bad3), sorted(os.listdir(inst2)) ==
           ["README.md", "index.dat", "s1223.dat"]], [1, 1, 1, 0, 1], 0)


def test_database():
    print("\nthe installed database (airfoils/)")
    files = AF._files(AF.AIRFOIL_DIR)
    if len(files) < 100:
        print(f"       skipped: {len(files)} files in {AF.AIRFOIL_DIR} (python airfoils.py --install coord_seligFmt.zip)")
        return
    bad, worse = [], []
    for key, path in sorted(files.items()):
        try:
            xy = AF.normalize(AF.read_file(path)[1])
            up, lo = AF.surfaces(xy)
            if len(up) < 3 or len(lo) < 3:
                raise ValueError("a surface with fewer than 3 points")
        except Exception as exc:
            bad.append(f"{key}: {exc}")
    for b in bad[:20]:
        print("       could not read " + b)
    tmp = tempfile.mkdtemp()
    names = ({"naca2412", "naca23012", "clarky", "e387", "s1223", "sd7037"} | set(sorted(files)[::25])) & set(files)
    names -= {b.split(":")[0] for b in bad}
    negative = []
    for name in sorted(names):
        try:
            path = AF.section_table(name, out_dir=tmp)
        except ValueError as exc:
            if "negative" not in str(exc):
                raise
            negative.append(name)
            continue
        worse.append(given_back(AF.normalize(AF.read_file(files[name])[1]), path))
    print(f"       {len(files)} files, {len(bad)} not readable; {len(names)} section tables, {len(negative)} refused "
          f"(camber line negative{': ' + ', '.join(negative) if negative else ''})")
    check("every file read (fewer than 1 % not); the tables give their airfoils back (2e-4: a nearly symmetric one "
          "loses its camber)", [len(bad) / len(files) < 0.01, max(worse) < 2e-4], [1, 1], 0)


def main():
    print("airfoils self-check")
    with tempfile.TemporaryDirectory() as tmp:
        test_naca()
        test_files(tmp)
        test_tables(tmp)
        test_lookup(tmp)
    test_database()
    print()
    if _FAILED:
        print(f"{len(_FAILED)} check(s) FAILED: " + ", ".join(_FAILED))
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
