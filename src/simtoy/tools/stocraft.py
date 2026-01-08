import pygfx as gfx
from rendercanvas.offscreen import RenderCanvas
import math as m

class Stocraft(gfx.WorldObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 创建一个平面 作为 Stocraft 的可视化表示
        plane = gfx.plane_geometry()
        material = gfx.MeshBasicMaterial(color=(0.5,0.5,0.5,0.5))
        mesh = gfx.Mesh(plane, material)
        # self.add(mesh)
        
        start = m.log(1,1.1)
        end = m.log(100,1.1)

        # 创建Ruler 作为 x，y，z轴的可视化表示
        axis_x = gfx.Ruler(start_pos=(0,0,0), end_pos=(1,0,0))
        self.add(axis_x)

        axis_y = gfx.Ruler(start_value=start,start_pos=(0,0,0), end_pos=(0,end,0))
        axis_y.local.scale_y = 1 / end
        self.add(axis_y)

        axis_z = gfx.Ruler(start_pos=(0,0,0), end_pos=(0,0,1))
        self.add(axis_z)
    
    def set_code(self, code: str):
        self.code = code

    def step(self, dt: float,camera: gfx.Camera, canvas : RenderCanvas):
        for ob in self.children:
            if isinstance(ob, gfx.Ruler):
                ob.update(camera, canvas.get_logical_size())
