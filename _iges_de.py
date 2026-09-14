import sys

TYPE_NAMES = {100: "circle", 102: "compcurve", 108: "plane", 110: "line",
              120: "surfrev", 122: "tabcyl", 124: "transform", 126: "bspline_curve",
              128: "bspline_surface", 141: "boundary", 142: "curve_on_surf",
              143: "bounded_surface", 144: "trimmed_surface", 502: "vertexlist",
              504: "edgelist", 508: "loop", 510: "face", 514: "shell", 186: "manifold_solid"}


def parse_de(path):
    """Return list of (DE_pointer, entity_type) from the Directory Entry section."""
    de = []
    with open(path, "r", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]
    d_lines = [ln for ln in lines if len(ln) >= 73 and ln[72] == "D"]
    # DE records are pairs; first line of each pair holds the entity type in cols 1-8
    # and its sequence number (the DE pointer) in cols 74-80.
    for i in range(0, len(d_lines), 2):
        first = d_lines[i]
        try:
            etype = int(first[0:8].strip())
            seq = int(first[73:80].strip())
        except ValueError:
            continue
        de.append((seq, etype))
    return de


for p in sys.argv[1:]:
    try:
        de = parse_de(p)
        surf = [(seq, t, TYPE_NAMES.get(t, str(t))) for seq, t in de
                if t in (143, 144, 128, 108)]
        trimmed = [seq for seq, t in de if t == 144]
        print(f"\n{p}")
        print(f"  total DE entities: {len(de)}")
        print(f"  trimmed-surface (144) DE pointers: {trimmed}")
        print(f"  TrimSurf names Pointwise would assign: {['TrimSurf-%d' % s for s in trimmed]}")
        from collections import Counter
        c = Counter(t for _, t in de)
        print("  entity-type histogram:", {TYPE_NAMES.get(k, k): v for k, v in sorted(c.items())})
    except Exception as e:
        print(p, "ERROR", repr(e))
