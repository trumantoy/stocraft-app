from datetime import datetime, timedelta
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
        self.steps = list()
     
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

        grid_xy = gfx.Grid(
            gfx.box_geometry(),
            gfx.GridMaterial(
                major_step=1,
                minor_step=0.1,
                thickness_space="world",
                axis_thickness=0.005,
                major_thickness=0.005,
                minor_thickness=0.001,
            ),
            orientation="xy",
        )
        self.add(grid_xy)

        self.process_measure = self.up_process = None

    def __del__(self):
        if self.up_process: self.up_process.terminate()

    def step(self, dt: float,camera: gfx.Camera, canvas : RenderCanvas):
        for ob in self.children:
            if isinstance(ob, gfx.Ruler):
                ob.update(camera, canvas.get_logical_size())
        
        if not self.steps:
            return 
        
        f = self.steps.pop(0)
        if f(): self.steps.append(f)

    def cmd_up(self, days = 1, func = None):
        if self.process_up:
            self.process_up.terminate()

        # 得到当天的时间，如果超过15点，取当天，否则取前一天
        now = datetime.now()
        if now.hour >= 15:
            date = now.date().strftime('%Y%m%d')
        else:
            date = (now.date() - timedelta(days=1)).strftime('%Y%m%d')

        def f():
            file = files("simtoy.data.stocraft") / "stocraft.py"
            self.process_up = sp.Popen(["python", file.as_posix(), 'up','--date',f'{date}','--days',f'{days}'],stdin=sp.PIPE,stdout=sp.PIPE,encoding='utf-8')
            print(' '.join(["python", file.as_posix(), 'up','--date',f'{date}','--days',f'{days}']),flush=True)

            while self.up_process.poll() is None:
                line = self.up_process.stdout.readline().strip()
                if not line: continue 
                print(line)
                if line != '-': 
                    func(line)
                else:
                    self.up_process.stdin.write('exit\n')
                    self.up_process.stdin.flush()

        # self.steps.append(f)
        threading.Thread(target=f,daemon=True).start()

    def cmd_measure(self,code,days=1,func=None):
        if self.process_measure:
            self.process_measure.terminate()

        def f():
            file = files("simtoy.data.stocraft") / "stocraft.py"
            cmd = ["python", file.as_posix(), 'measure','--code', code]
            self.process_measure = sp.Popen(cmd,stdin=sp.PIPE,stdout=sp.PIPE,encoding='utf-8')
            print(' '.join(cmd),flush=True)
    
            while self.process_measure.poll() is None:
                line = self.process_measure.stdout.readline().strip()
                if not line: continue 
                print(line)
                if line != '-': 
                    func(line)
                else:
                    self.process_measure.stdin.write('exit\n')
                    self.process_measure.stdin.flush()

        threading.Thread(target=f,daemon=True).start()    