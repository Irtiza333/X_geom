"""
blade_vary.py

What changes in a blade's component file (blade_modify.py, xgeom_files.py): its
vary lines, which parameters change and the range of their values, one line per
parameter in the header:

    # vary  pitch       Y  0.6  1.4      Y: a design variable, its values in 0.6 .. 1.4 (the range is optional)
    # vary  thickness   N                N: kept as the file has it

A name not listed is N. The vary lines are for hand-written files and table
files (no var lines); a set-up the tool saves has var lines instead, one per
variable with its free flag ('# var pitch.v1 1.24 0.6 1.4 Y'), and read_free
takes what changes from those: a parameter with a free variable. blade_modify.py
reads the vary lines through xgeom_files.py for the XGeom blade
(blade_control.py, the design tool); Initial_sampling.py reads them here for the
MSc pipeline (para_control from pitch and chord, their ranges as bounds), else
the free var lines. Standard library only: a pipeline folder needs this file and
the component file (msc_blade.txt), nothing else of XGeom.
"""

from dataclasses import dataclass

CURVES = ("pitch", "chord", "thickness", "camber", "skew", "rake")   # the radial distributions
GLOBAL_SLOTS = ("radius", "hub_radius", "hub_height")             # the blade's own design variables, m
HUB_SLOTS = ("hub_base", "hub_top", "hub_x")                      # a conic hub's (prop_hub.py): its base and top
                                                                  # radii and the x of its base plane, m
VARY_NAMES = CURVES + GLOBAL_SLOTS + HUB_SLOTS                     # the parameters a '# vary' line names


@dataclass
class Vary:
    """A parameter's '# vary' line: changed (on, Y) or kept (N), and the range of its values (low, high) when
    given."""
    on: bool
    lo: float = None
    hi: float = None


def read_vary(path):
    """{name: Vary} of a parameter file's '# vary <name> Y|N [low high]' lines, every VARY_NAMES (a name not
    listed: N); {} when the file has none."""
    out = {}
    with open(path) as fh:
        for k, line in enumerate(fh, start=1):
            s = line.strip()
            parts = s[1:].split() if s.startswith("#") else []
            if not parts or parts[0] != "vary":
                continue
            where = f"{path}, line {k}"
            if len(parts) not in (3, 5):
                raise ValueError(f"{where}: '# vary <name> Y|N [low high]'")
            name, flag = parts[1], parts[2].upper()
            if name not in VARY_NAMES:
                raise ValueError(f"{where}: '{name}' is not one of {', '.join(VARY_NAMES)}")
            if name in out:
                raise ValueError(f"{where}: {name} has a vary line already")
            if flag not in ("Y", "N", "YES", "NO"):
                raise ValueError(f"{where}: Y or N, not '{parts[2]}'")
            lo = hi = None
            if len(parts) == 5:
                try:
                    lo, hi = float(parts[3]), float(parts[4])
                except ValueError:
                    raise ValueError(f"{where}: low and high are numbers") from None
                if not lo < hi:
                    raise ValueError(f"{where}: low must lie below high")
            out[name] = Vary(flag in ("Y", "YES"), lo, hi)
    if out:
        out = {n: out.get(n, Vary(False)) for n in VARY_NAMES}
    return out


def read_free(path):
    """{name: Vary} of a component file's '# var <name> <value> <low> <high> Y|N [<relation>]' lines, every
    VARY_NAMES: a parameter is on (Y, no range) when one of its variables is free (pitch.v1, ... for a
    distribution); {} when the file has no var line."""
    on = {}
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            parts = s[1:].split() if s.startswith("#") else []
            if len(parts) < 6 or parts[0] != "var":
                continue
            name = parts[1].split(".")[0]
            on[name] = on.get(name, False) or parts[5].upper() in ("Y", "YES", "1", "TRUE")
    if not on:
        return {}
    return {n: Vary(on.get(n, False)) for n in VARY_NAMES}
