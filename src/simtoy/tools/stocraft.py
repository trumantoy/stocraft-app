import pygfx as gfx
from rendercanvas.offscreen import RenderCanvas
import math as m
import subprocess as sp
import threading
from importlib.resources import files
import pandas as pd
import numpy as np

class Stocraft(gfx.WorldObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 创建一个平面 作为 Stocraft 的可视化表示
        # plane = gfx.plane_geometry()
        # material = gfx.MeshBasicMaterial(color=(0.5,0.5,0.5,0.5))
        # mesh = gfx.Mesh(plane, material)
        # self.add(mesh)
        
        start = m.log(1,1.01)
        end = m.log(100,1.01)

        # 创建Ruler 作为 x，y，z轴的可视化表示
        axis_x = gfx.Ruler(start_pos=(0,0,0), end_pos=(1,0,0))
        self.add(axis_x)

        axis_y = gfx.Ruler(start_value=start,start_pos=(0,0,0), end_pos=(0,end,0))
        axis_y.local.scale_y = 1 / end
        self.add(axis_y)

        axis_z = gfx.Ruler(start_pos=(0,0,0), end_pos=(0,0,1))
        self.add(axis_z)

        self.process = None

    def set_code(self, code: str):
        self.code = code

        if self.process:
            self.process.terminate()

        file = files("simtoy.data.stocraft") / "stocraft.py"
        self.process = sp.Popen(["python", file.as_posix(), 'measure','--code', code],stdin=sp.PIPE,stdout=sp.PIPE,encoding='utf-8')
        self.df = pd.DataFrame()
        measuring = threading.Thread(target=self.measure, args=(self.process,self.df),daemon=True)
        measuring.start()

    def measure(self, p:sp.Popen, df:pd.DataFrame):
        line = p.stdout.readline().strip()
        df.index = [i for i in range(10000)]
        df[line.split(',')] = np.full((10000,4),np.nan)
        df[df.columns[0]] = df[df.columns[0]].astype(str)
        df[df.columns[3]] = df[df.columns[3]].astype(str)

        end = 0
        while line != '-':
            line = p.stdout.readline().strip()
            if not line or line == '-': continue
            row = line.split(',')
            id,time,price,count,kind = int(row[0]),row[1],float(row[2]),float(row[3]),row[4]
            df.loc[id] = [time,price,count,kind]
            end = id
            
        print(df.loc[:end])

    def step(self, dt: float,camera: gfx.Camera, canvas : RenderCanvas):
        for ob in self.children:
            if isinstance(ob, gfx.Ruler):
                ob.update(camera, canvas.get_logical_size())
                
