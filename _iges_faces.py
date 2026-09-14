import sys
from OCC.Core.IGESControl import IGESControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID


def faces(path):
    r = IGESControl_Reader()
    r.ReadFile(path)
    r.TransferRoots()
    shape = r.OneShape()
    out = {}
    for name, kind in [("solids", TopAbs_SOLID), ("shells", TopAbs_SHELL), ("faces", TopAbs_FACE)]:
        n, ex = 0, TopExp_Explorer(shape, kind)
        while ex.More():
            n += 1; ex.Next()
        out[name] = n
    return out


for p in sys.argv[1:]:
    try:
        print(p, "->", faces(p))
    except Exception as e:
        print(p, "ERROR", repr(e))
