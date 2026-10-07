# Airfoil coordinates

The UIUC Airfoil Coordinates Database, Selig format (the archive coord_seligFmt.zip on https://m-selig.ae.illinois.edu/ads/coord_database.html), UIUC Applied Aerodynamics Group, M. Selig. Installed 2026-10-06 with `python airfoils.py --install`: 1665 files.

`index.dat` lists every file (name, points, description). Own airfoils: add `<name>.dat` (Selig or Lednicer format); the blade parameter file names them with `# airfoil <name>`. `sections/` holds the section tables made from them for para.py (made again when needed).
