"""
blade_sections.py

A blade whose section shape changes along the span: the per-radius section
tables of a propeller backed out of its CAD (blade_extraction.py writes them,
one para.py table per r/R station in one file; the component file names it
under section_tables) as a BladeSurface, SectionsBladeSurface, cubic in r/R
between the tables. The one-table BladeSurface of the MSc code is the special
case of one section family; blade_modify.blade_surface builds this one when
the parameters carry the tables, so the design tool's views, the CAD and the
checks follow the shapes the propeller has (BB2's propeller: one table is off
by 23 mm at the root, the tables within 1 mm; bb2/bb2_build.py built the BB2
set-up files with the same class).

    tables = read_tables("bb2_prop_section_tables.dat")      # [(r/R, table text)]
    blade = SectionsBladeSurface(tables, *design.props(), d=design.diameter, r_root=design.r0,
                                 root_extension=ext, airfoil_path=params.section_path())
"""

from __future__ import annotations

import io

import numpy as np
from scipy.interpolate import RectBivariateSpline

from blade_surface_new import BladeSurface, CanonicalSectionFamily

N_V = 801                    # stations around a section behind the shapes


def read_tables(path):
    """[(r/R, table text)] of a section tables file (each text a section table pandas reads as para.py does):
    '# r/R <value>' before each table's header line and 26 rows."""
    out, cur = [], None
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("# r/R"):
                cur = [float(s.split()[2]), []]
                out.append(cur)
            elif cur is not None and s and not s.startswith("#"):
                cur[1].append(s)
    if not out:
        raise ValueError(f"{path}: no '# r/R <value>' blocks")
    return [(r, "\n".join(lines) + "\n") for r, lines in out]


class SectionsBladeSurface(BladeSurface):
    """BladeSurface with a section shape per r/R: tables [(r/R, section table text)], the camber and thickness
    distributions of each (CanonicalSectionFamily) on N_V stations around the section, cubic in r/R between the
    tables, the first and last held beyond them. The rest is BladeSurface's (the distributions, the tip taper and
    closure, the root extension)."""

    def __init__(self, tables, *args, **kw):
        super().__init__(*args, **kw)
        tables = list(tables)
        if len(tables) == 1:                                 # one shape everywhere
            tables = [(0.0, tables[0][1]), (1.0, tables[0][1])]
        rk = np.array([r for r, _ in tables], dtype=float)
        if np.any(np.diff(rk) <= 0.0):
            raise ValueError("the section tables' r/R must rise from table to table")
        ky = min(3, len(rk) - 1)                             # cubic across the tables, lower with fewer than 4
        v = np.linspace(0.0, 1.0, N_V)                       # 0 at the LE, 1 at the TE: |1 - 2 xi|
        cam, th = [], []
        for _, text in tables:
            fam = CanonicalSectionFamily(None, None, airfoil_path=io.StringIO(text))
            s = fam.s_end * 0.5 * (1.0 - np.cos(np.pi * v))
            cam.append(fam._camber_dist(s))
            th.append(fam._thickness_dist(s) * np.cos(np.arctan(fam._camber_slope(s))))
        self._cam_vr = RectBivariateSpline(v, rk, np.array(cam).T, ky=ky)
        self._th_vr = RectBivariateSpline(v, rk, np.array(th).T, ky=ky)
        self._r_range = (float(rk[0]), float(rk[-1]))

    def _shape(self, xi, r):
        """(s, F_cam, F_th) of CanonicalSectionFamily.basis, at r/R."""
        v = np.abs(1.0 - 2.0 * xi)
        rr = np.clip(r, *self._r_range)
        s = self.canon.s_end * 0.5 * (1.0 - np.cos(np.pi * v))
        f = self._cam_vr(v.ravel(), rr.ravel(), grid=False).reshape(v.shape)
        t = self._th_vr(v.ravel(), rr.ravel(), grid=False).reshape(v.shape)
        return s, f, np.where(xi < 0.5, -1.0, 1.0) * t

    def _canonical(self, xi, r):
        """BladeSurface._canonical with the shape at r/R (the tip closure as there)."""
        xi = np.atleast_1d(np.asarray(xi, dtype=float))
        r = np.atleast_1d(np.asarray(r, dtype=float))
        xi, r = np.broadcast_arrays(xi, r)
        s, F_cam, F_th = self._shape(xi, r)
        ys = np.asarray(F_cam * self.MaxCamber(r) + F_th * self.MaxThickness(r), dtype=float)
        mask = r > self._rc
        if np.any(mask):
            c = self._closure_constants()
            rc, rt = self._rc, self.r_tip
            dr = rt - rc
            y0 = F_cam[mask] * c["mc0"]
            s0 = F_cam[mask] * c["dmc0"]
            y1 = F_cam[mask] * c["mc1"]
            s1 = 2.0 * (y1 - y0) / dr - s0
            u = np.clip((r[mask] - rc) / dr, 0.0, 1.0)
            h00 = 2 * u ** 3 - 3 * u ** 2 + 1
            h10 = u ** 3 - 2 * u ** 2 + u
            h01 = -2 * u ** 3 + 3 * u ** 2
            h11 = u ** 3 - u ** 2
            ys_cam = h00 * y0 + h10 * dr * s0 + h01 * y1 + h11 * dr * s1
            g = np.sqrt(np.clip(1.0 - u ** 3, 0.0, None))
            ys = ys.copy()
            ys[mask] = ys_cam + F_th[mask] * np.asarray(self.MaxThickness(r[mask]), dtype=float) * g
        return np.stack([s, ys], axis=-1)
