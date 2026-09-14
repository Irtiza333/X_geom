import os, sys
os.add_dll_directory(os.path.join(sys.prefix, "Library", "bin"))

import numpy as np
# try:
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_Sewing,
)
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCC.Core.BRepFill import BRepFill_Filling
from OCC.Core.TopoDS import topods
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.GeomAbs import GeomAbs_C0
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Circ, gp_Ax2
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Extend.DataExchange import write_iges_file
from OCC.Core.ShapeFix import ShapeFix_Shape
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_SHELL

from pipeline_config import cad_output_paths


# except ImportError:
#     from OCP.BRepBuilderAPI import (
#         BRepBuilderAPI_MakeEdge,
#         BRepBuilderAPI_MakeWire,
#         BRepBuilderAPI_MakeFace,
#         BRepBuilderAPI_MakeSolid,
#         BRepBuilderAPI_Sewing,
#     )
#     from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
#     from OCP.BRepFill import BRepFill_Filling
#     from OCP.TopoDS import topods
#     from OCP.GeomAPI import GeomAPI_PointsToBSpline
#     from OCP.GeomAbs import GeomAbs_C0
#     from OCP.gp import gp_Pnt, gp_Dir, gp_Circ, gp_Ax2
#     from OCP.TColgp import TColgp_Array1OfPnt
#     from OCP.Display.SimpleGui import init_display
#     from OCP.Extend.DataExchange import write_iges_file
#     from OCP.ShapeFix import ShapeFix_Shape
#     from OCP.TopExp import TopExp_Explorer
#     from OCP.TopAbs import TopAbs_EDGE, TopAbs_SHELL

def X_CAD(points, x1, output_dir=None):
    class AirfoilBlade:
        def __init__(self, points_np, sections=48, points_per_section=53, display=None):
            self.points_np = points_np  # shape: (sections*points_per_section, 3)
            self.sections = sections
            self.points_per_section = points_per_section
            self.display = display

        def create_spline(self, points):
            points_array = TColgp_Array1OfPnt(1, len(points))
            for i, pnt in enumerate(points):
                points_array.SetValue(i + 1, pnt)
            spline_builder = GeomAPI_PointsToBSpline(points_array, 3,5, GeomAbs_C0,1e-6)
            return spline_builder.Curve()

        def create_face(self, splines):
            filling = BRepFill_Filling()
            for spline in splines:
                edge = BRepBuilderAPI_MakeEdge(spline).Edge()
                # self.display.DisplayShape(edge, update=True)  # Visualize edges
                filling.Add(edge, GeomAbs_C0)
            filling.Build()
            if not filling.IsDone():
                raise RuntimeError("Filling failed to create a face.")
            face = filling.Face()
            # self.display.DisplayShape(face, update=True)  # Visualize face
            return face

        def create_shell(self):
            loft = BRepOffsetAPI_ThruSections(True, False)

            for i in range(self.sections):
                start_idx = i * self.points_per_section
                end_idx = (i + 1) * self.points_per_section
                sec = self.points_np[start_idx:end_idx, :] * 1000.0  # mm

                p_10_to_1_and_53_to_42 = np.vstack([sec[2::-1, :], sec[51:49:-1, :]])
                p_10_to_27 = sec[2:26, :]
                p_27_to_42 = sec[27:51, :]
                p_26_to_28 = sec[25:28, :]

                points_10_to_1_and_53_to_42 = [gp_Pnt(x, y, z) for x, y, z in p_10_to_1_and_53_to_42]
                points_10_to_27 = [gp_Pnt(x, y, z) for x, y, z in p_10_to_27]
                points_27_to_42 = [gp_Pnt(x, y, z) for x, y, z in p_27_to_42]
                points_26_to_28 = [gp_Pnt(x, y, z) for x, y, z in p_26_to_28]

                try:
                    spline_1 = self.create_spline(points_10_to_1_and_53_to_42)
                    spline_2 = self.create_spline(points_10_to_27)
                    spline_3 = self.create_spline(points_27_to_42)
                    spline_4 = self.create_spline(points_26_to_28)

                    face = self.create_face([spline_1, spline_2, spline_4, spline_3])

                    wire_builder = BRepBuilderAPI_MakeWire()
                    explorer = TopExp_Explorer(face, TopAbs_EDGE)
                    while explorer.More():
                        edge = topods.Edge(explorer.Current())
                        wire_builder.Add(edge)
                        explorer.Next()

                    if not wire_builder.IsDone():
                        raise RuntimeError("Failed to create a wire from the face.")

                    wire = wire_builder.Wire()
                    # self.display.DisplayShape(wire, update=True)  # Display wire
                    loft.AddWire(wire)
                except RuntimeError as e:
                    print(f"Failed to create face or wire for section {i}: {e}")

            loft.Build()
            if not loft.IsDone():
                raise RuntimeError("Shell creation failed.")
            shell = loft.Shape()
            # self.display.DisplayShape(shell, update=True)  # Display shell
            return shell


        def create_last_cap(self):
            """
            Create the last cap of the airfoil blade using multiple spline segments.
            """
            sec = self.points_np[-self.points_per_section:, :] * 1000.0
        
            # Define spline segments for the last cap
            p_10_to_1_and_53_to_42 = np.vstack([sec[2::-1, :], sec[51:49:-1, :]])
            p_10_to_27 = sec[2:26, :]
            p_27_to_42 = sec[27:51, :]
            p_26_to_28 = sec[25:28, :]

            points_10_to_1_and_53_to_42 = [gp_Pnt(x, y, z) for x, y, z in p_10_to_1_and_53_to_42]
            points_10_to_27 = [gp_Pnt(x, y, z) for x, y, z in p_10_to_27]
            points_27_to_42 = [gp_Pnt(x, y, z) for x, y, z in p_27_to_42]
            points_26_to_28 = [gp_Pnt(x, y, z) for x, y, z in p_26_to_28]
        
            try:
                # Create splines for the last cap
                spline_1 = self.create_spline(points_10_to_1_and_53_to_42)
                spline_2 = self.create_spline(points_10_to_27)
                spline_3 = self.create_spline(points_27_to_42)
                spline_4 = self.create_spline(points_26_to_28)
        
                # Create the face from splines
                face = self.create_face([spline_1, spline_2, spline_4, spline_3])
                #self.display.DisplayShape(face, update=True)  # Visualize last cap
                return face
            except RuntimeError as e:
                print(f"Failed to create last cap: {e}")
                return None
            
    def create_fixed_solid(blade_shell, last_cap):
        sewing = BRepBuilderAPI_Sewing(1e-6)
        sewing.Add(blade_shell)
        # sewing.Add(first_cap)
        sewing.Add(last_cap)
        # sewing.Add(cylinder)
        sewing.Perform()

        sewed_shape = sewing.SewedShape()
        fixer = ShapeFix_Shape(sewed_shape)
        fixer.Perform()
        return fixer.Shape()        

    # Build and export IGES when called (headless; no GUI)
    total_points = points.shape[0]
    points_per_section = 53
    sections = total_points // points_per_section

    blade = AirfoilBlade(points, sections=sections, points_per_section=points_per_section)
    blade_shell = blade.create_shell()
    last_cap = blade.create_last_cap()

    if not last_cap:
        raise RuntimeError("X_CAD: Failed to create last cap")

    combined_solid = create_fixed_solid(blade_shell, last_cap)

    paths = cad_output_paths(x1, output_dir)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    iges_path = paths["iges"]
    write_iges_file(combined_solid, str(iges_path))
    print(f"IGES file written: {iges_path}")
    return combined_solid
