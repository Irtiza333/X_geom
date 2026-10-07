"""
airfoils.py

Airfoil sections by name, offline: the UIUC Airfoil Coordinates Database
(m-selig.ae.illinois.edu/ads/coord_database.html, about 1,650 airfoils) kept in
the airfoils/ folder beside this file, and the NACA 4- and 5-digit equations for
a NACA name the folder has no file for.

    xy, source = coordinates("naca2416")          # (n, 2), Selig order, chord 1
    path = section_table("naca2416")              # para.py's section table (26 rows)

Names: case, spaces, '-' and '_' and a '.dat' ending are ignored ("NACA 2416",
"naca-2416" and "naca2416.dat" are one name). A file airfoils/<name>.dat wins;
then naca + 4 digits (mptt) or 5 digits (LPSTT, L P S = 2 3 0 for the 230
series) are computed; anything else is an error naming the closest files. Own
airfoils: put <name>.dat (Selig or Lednicer format) into airfoils/.

The database is installed once from the UIUC archive (coord_seligFmt.zip, the
whole database in Selig format, from the page's Archives section) or the folder
it unpacks to:

    python airfoils.py --install coord_seligFmt.zip      (or: coord_seligFmt, or airfoils)
    python airfoils.py naca24                     list the names that contain 'naca24'

which puts the coordinate files into airfoils/ (airfoils itself: they are
there already), writes airfoils/index.dat (name, points, description) and
airfoils/README.md (source, date).

Section table (para.py's and blade_surface_new.BladeSurface's format, comma
separated, one header line): the 26 stations x/c of airfoil_data_fixed.csv; the
camber line (% c; para.py normalizes it by its maximum and scales it by f/c);
its slope dyc/dx (0 at the LE, as in airfoil_data_fixed.csv); the half thickness
over the maximum full thickness (peaks at 0.5; para.py scales it by t/c). So a
named airfoil gives the shapes, the blade's t/c and f/c the sizes. The camber is
the mean of the upper and lower surfaces at the same x, the thickness their
difference. Tables are written to airfoils/sections/ when first asked for (again
when the coordinate file is newer).
"""

from __future__ import annotations

import datetime
import difflib
import os
import re
import sys
import zipfile

import numpy as np
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
AIRFOIL_DIR = os.path.join(HERE, "airfoils")
STATIONS = np.array([0.0, 0.005, 0.0075, 0.0125, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                     0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])   # airfoil_data_fixed.csv
SOURCE_URL = "https://m-selig.ae.illinois.edu/ads/coord_database.html"


def norm_name(name):
    """The lookup form of a name: lower case, no spaces, '-', '_' or '.dat'."""
    n = os.path.basename(str(name)).strip().lower()
    if n.endswith(".dat"):
        n = n[:-4]
    return re.sub(r"[\s_\-]+", "", n)


# --------------------------------------------------------------------------
# Coordinate files
# --------------------------------------------------------------------------

def read_file(path):
    """(description, xy) of a coordinate file: Selig format (a name line, then
    x y from the TE over the upper surface to the LE and back) or Lednicer
    format (a name line, the point counts of the two surfaces, then each surface
    from the LE to the TE); returned in Selig order. Lines that are not two
    numbers are skipped."""
    with open(path, errors="replace") as fh:
        lines = fh.read().splitlines()
    desc = lines[0].strip() if lines else ""
    pts = []
    for line in lines[1:]:
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            pts.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    xy = np.array(pts, dtype=float)
    if len(xy) and xy[0, 0] > 1.5 and xy[0, 1] > 1.5:                 # Lednicer: the counts
        nu, nl = int(round(xy[0, 0])), int(round(xy[0, 1]))
        up, lo = xy[1:1 + nu], xy[1 + nu:1 + nu + nl]
        xy = np.vstack((up[::-1], lo[1:] if np.allclose(lo[0], up[0]) else lo))
    if len(xy) < 5:
        raise ValueError(f"{path}: fewer than 5 coordinate points")
    return desc, xy


def write_file(path, xy, desc):
    """A coordinate file in Selig format."""
    with open(path, "w") as fh:
        fh.write(desc + "\n" + "\n".join(f"{x:.6f} {y: .6f}" for x, y in xy) + "\n")
    return path


def naca(name, n=81):
    """NACA 4-digit (mptt) or 5-digit (LPSTT) coordinates, Selig order, n points
    per surface (cosine spacing), the standard thickness (finite TE); None if
    the name is not one. 5-digit: L the design lift coefficient over 0.15, P the
    position of the maximum camber over 0.05 (1 .. 5), S 0 (normal) or 1
    (reflexed), TT the thickness in % c."""
    m4 = re.fullmatch(r"naca(\d)(\d)(\d\d)", norm_name(name))
    m5 = re.fullmatch(r"naca(\d)(\d)([01])(\d\d)", norm_name(name))
    if not (m4 or m5):
        return None
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
    t = int((m4 or m5).groups()[-1]) / 100.0
    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)
    if m4:
        m, p = int(m4.group(1)) / 100.0, int(m4.group(2)) / 10.0
        if m > 0.0 and p > 0.0:
            front = x < p
            yc = np.where(front, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
            dyc = np.where(front, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
        else:
            yc, dyc = np.zeros_like(x), np.zeros_like(x)
    else:
        L, P, S = (int(g) for g in m5.groups()[:3])
        normal = {1: (0.0580, 361.4), 2: (0.1260, 51.64), 3: (0.2025, 15.957), 4: (0.2900, 6.643), 5: (0.3910, 3.230)}
        reflex = {2: (0.1300, 51.99, 0.000764), 3: (0.2170, 15.793, 0.00677), 4: (0.3180, 6.520, 0.0303),
                  5: (0.4410, 3.191, 0.1355)}
        table = reflex if S else normal
        if P not in table:
            raise ValueError(f"{name}: no NACA 5-digit camber line for the digits {L}{P}{S}")
        r, k1 = table[P][:2]
        k1 *= L * 0.15 / 0.3                                  # the k1 above are for a design cl of 0.3
        front = x < r
        if S == 0:
            yc = np.where(front, k1 / 6 * (x ** 3 - 3 * r * x ** 2 + r ** 2 * (3 - r) * x), k1 * r ** 3 / 6 * (1 - x))
            dyc = np.where(front, k1 / 6 * (3 * x ** 2 - 6 * r * x + r ** 2 * (3 - r)), -k1 * r ** 3 / 6)
        else:
            k21 = table[P][2]
            yc = np.where(front, k1 / 6 * ((x - r) ** 3 - k21 * (1 - r) ** 3 * x - r ** 3 * x + r ** 3),
                          k1 / 6 * (k21 * (x - r) ** 3 - k21 * (1 - r) ** 3 * x - r ** 3 * x + r ** 3))
            dyc = np.where(front, k1 / 6 * (3 * (x - r) ** 2 - k21 * (1 - r) ** 3 - r ** 3),
                           k1 / 6 * (3 * k21 * (x - r) ** 2 - k21 * (1 - r) ** 3 - r ** 3))
    th = np.arctan(dyc)
    up = np.column_stack((x - yt * np.sin(th), yc + yt * np.cos(th)))
    lo = np.column_stack((x + yt * np.sin(th), yc - yt * np.cos(th)))
    return np.vstack((up[::-1], lo[1:]))


def _files(folder):
    """{lookup name: path} of the .dat files in the folder."""
    if not os.path.isdir(folder):
        return {}
    return {norm_name(f): os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".dat")
            and f.lower() != "index.dat"}


def coordinates(name, folder=None):
    """(xy, source) of a named airfoil: Selig order, normalized (LE at (0, 0),
    TE midpoint at (1, 0)). source: the file it came from, or 'NACA 4-digit
    equations' / 'NACA 5-digit equations'."""
    folder = folder or AIRFOIL_DIR
    key = norm_name(name)
    quick = os.path.join(folder, key + ".dat")                 # the installed names are lower case
    if os.path.isfile(quick):
        return normalize(read_file(quick)[1]), quick
    files = _files(folder)
    if key in files:
        return normalize(read_file(files[key])[1]), files[key]
    xy = naca(key)
    if xy is not None:
        return normalize(xy), f"NACA {len(key) - 4}-digit equations"
    close = difflib.get_close_matches(key, list(files), n=6, cutoff=0.6)
    raise ValueError(f"airfoil '{name}': not in {folder} ({len(files)} files) and not a NACA 4- or 5-digit name"
                     + (f"; close: {', '.join(close)}" if close else "")
                     + ("" if files else "; the database is not installed (python airfoils.py --install "
                                         "coord_seligFmt.zip)"))


def describe(name, folder=None):
    """One line on where a named airfoil comes from."""
    try:
        _, src = coordinates(name, folder)
    except ValueError as exc:
        return str(exc)
    if src.startswith("NACA"):
        return f"{norm_name(name)} ({src})"
    desc = read_file(src)[0]
    return f"{norm_name(name)} ({os.path.basename(src)}" + (f": {desc}" if desc else "") + ")"


def normalize(xy, tol=2e-3, le_tol=1e-4):
    """Coordinates with the LE at (0, 0) and the TE midpoint at (1, 0),
    consecutive duplicate points dropped. Coordinates already so (a point
    within le_tol of (0, 0), the TE midpoint within tol of (1, 0); most of the
    database) are kept as they are, the authors' chord line; others are moved,
    turned and scaled with the LE at the smallest x (in the database: an LE
    up to 0.2 % c off the origin, a turn of up to about 0.1 deg)."""
    xy = np.asarray(xy, dtype=float)
    keep = np.ones(len(xy), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(xy, axis=0)) > 1e-12, axis=1)
    xy = xy[keep]
    te = 0.5 * (xy[0] + xy[-1])
    if np.hypot(te[0] - 1.0, te[1]) < tol and np.min(np.hypot(*xy.T)) < le_tol:
        return xy                                             # a point at (0, 0) and the TE at (1, 0)
    le = xy[int(np.argmin(xy[:, 0]))]
    chord = te - le
    c = float(np.hypot(*chord))
    cos, sin = chord / c
    rot = np.array([[cos, sin], [-sin, cos]])
    return (xy - le) @ rot.T / c


# --------------------------------------------------------------------------
# Section tables for para.py
# --------------------------------------------------------------------------

def surfaces(xy):
    """(upper, lower) of normalized coordinates, each from the LE to the TE
    with x rising (points that would turn x back are dropped)."""
    i = int(np.argmin(np.hypot(*xy.T)))                       # the LE, (0, 0)
    xy = np.array(xy, dtype=float)
    xy[i] = 0.0                                               # (within normalize's le_tol)
    a, b = xy[:i + 1][::-1], xy[i:]
    if np.mean(a[:, 1]) < np.mean(b[:, 1]):                   # the upper surface first
        a, b = b, a
    out = []
    for s in (a, b):
        s = s[np.concatenate(([True], s[1:, 0] > np.maximum.accumulate(s[:, 0])[:-1]))]
        out.append(s)
    return out


def shape(xy, x=STATIONS):
    """Camber line (chord fractions), its slope, half thickness, and the
    maximum full thickness and camber, at the stations x; from normalized
    coordinates (the surfaces interpolated against sqrt(x), monotone). Camber
    and thickness are the mean and half difference of the surfaces at the same
    x: para.py and BladeSurface put a section together that way (thickness
    added at the station itself), so the table gives the airfoil back."""
    up, lo = surfaces(xy)
    fu = PchipInterpolator(np.sqrt(np.clip(up[:, 0], 0.0, None)), up[:, 1], extrapolate=True)
    fl = PchipInterpolator(np.sqrt(np.clip(lo[:, 0], 0.0, None)), lo[:, 1], extrapolate=True)
    q = np.sqrt(np.asarray(x, dtype=float))
    yc, th = 0.5 * (fu(q) + fl(q)), 0.5 * (fu(q) - fl(q))
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(q > 0, 0.5 * (fu(q, 1) + fl(q, 1)) / (2.0 * q), 0.0)
    qf = np.linspace(0.0, 1.0, 4001)
    t_max = float(np.max(fu(qf) - fl(qf)))
    c_max = float(np.max(0.5 * (fu(qf) + fl(qf))))
    return yc, slope, th, t_max, c_max


def section_table(name, folder=None, out_dir=None):
    """The path of para.py's section table of a named airfoil (see the module
    docstring), written to <folder>/sections/<name>.dat if it is not there or
    is older than the coordinate file or this file."""
    folder = folder or AIRFOIL_DIR
    xy, src = coordinates(name, folder)
    out_dir = out_dir or os.path.join(folder, "sections")
    path = os.path.join(out_dir, f"{norm_name(name)}.dat")
    made_from = [f for f in (src, os.path.abspath(__file__)) if os.path.isfile(f)]
    if os.path.exists(path) and all(os.path.getmtime(path) >= os.path.getmtime(f) for f in made_from):
        return path
    yc, slope, th, t_max, c_max = shape(xy)
    if t_max <= 0.0:
        raise ValueError(f"airfoil '{name}': no thickness")
    tiny = 1e-3 * t_max                                       # camber below this: the file's rounding
    if c_max <= tiny and np.min(yc) < -tiny:
        raise ValueError(f"airfoil '{name}': its camber line is negative; f/c scales a camber line with a "
                         f"positive maximum")
    symmetric = c_max <= tiny                                 # no camber shape: f/c has no effect
    if symmetric:
        yc, slope = np.zeros_like(yc), np.zeros_like(slope)
    # para.py and BladeSurface multiply the thickness by cos(atan(slope)): divided out here
    rows = np.column_stack((STATIONS, 100.0 * yc, slope, th / t_max / np.cos(np.arctan(slope))))
    rows[0] = 0.0                                              # the LE row, as in airfoil_data_fixed.csv
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"x/c,camber (% c; {norm_name(name)}: t/c {t_max:.5f} f/c {max(c_max, 0.0):.5f}),"
                 f"camber slope,half thickness / max thickness\n")
        fh.write("\n".join(",".join(f"{v:.6f}" for v in row) for row in rows) + "\n")
    return path


# --------------------------------------------------------------------------
# The database
# --------------------------------------------------------------------------

def install(source, folder=None):
    """Put the UIUC database into the folder: from the archive
    (coord_seligFmt.zip; the .dat files without the archive's folder names) or
    from an unpacked folder of .dat files (copied; nothing to copy when it is
    the folder itself); files of the same name are replaced, names in lower
    case. Then index.dat (name, points, description) and README.md. Returns
    (files, files that could not be read)."""
    folder = folder or AIRFOIL_DIR
    os.makedirs(folder, exist_ok=True)
    n = 0
    if os.path.isdir(source):
        here = os.path.samefile(source, folder)               # the files are there already
        for f in sorted(os.listdir(source)):
            if f.lower().endswith(".dat") and f.lower() != "index.dat":
                if not here:
                    with open(os.path.join(source, f), "rb") as src, open(os.path.join(folder, f.lower()), "wb") as fh:
                        fh.write(src.read())
                n += 1
    else:
        with zipfile.ZipFile(source) as zf:
            for info in zf.infolist():
                base = os.path.basename(info.filename)
                if info.is_dir() or not base.lower().endswith(".dat"):
                    continue
                with open(os.path.join(folder, base.lower()), "wb") as fh:
                    fh.write(zf.read(info))
                n += 1
    bad = write_index(folder)
    with open(os.path.join(folder, "README.md"), "w") as fh:
        fh.write(f"# Airfoil coordinates\n\nThe UIUC Airfoil Coordinates Database, Selig format (the archive "
                 f"coord_seligFmt.zip on {SOURCE_URL}), UIUC Applied Aerodynamics Group, M. Selig. Installed "
                 f"{datetime.date.today().isoformat()} with `python airfoils.py --install`: {n} files.\n\n"
                 "`index.dat` lists every file (name, points, description). Own airfoils: add `<name>.dat` (Selig "
                 "or Lednicer format); the blade parameter file names them with `# airfoil <name>`. `sections/` "
                 "holds the section tables made from them for para.py (made again when needed).\n")
    return n, bad


def write_index(folder=None):
    """index.dat: name, number of points and description of every file;
    returns the files that could not be read."""
    folder = folder or AIRFOIL_DIR
    rows, bad = [], []
    for key, path in sorted(_files(folder).items()):
        try:
            desc, xy = read_file(path)
            rows.append(f"{key:<24s} {len(xy):5d}  {desc}")
        except Exception as exc:
            bad.append(f"{os.path.basename(path)}: {exc}")
    with open(os.path.join(folder, "index.dat"), "w") as fh:
        fh.write("# name                   points  description\n" + "\n".join(rows) + "\n")
    return bad


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--install":
        n, bad = install(sys.argv[2])
        print(f"{n} coordinate files in {AIRFOIL_DIR}" + (f"; {len(bad)} could not be read:" if bad else ""))
        for b in bad:
            print("  " + b)
    elif len(sys.argv) == 2:
        q = norm_name(sys.argv[1])
        names = sorted(k for k in _files(AIRFOIL_DIR) if q in k)
        print("\n".join(names) if names else f"no file name contains '{q}'"
              + (" (a NACA name is computed)" if naca(q) is not None else ""))
    else:
        print(__doc__)
