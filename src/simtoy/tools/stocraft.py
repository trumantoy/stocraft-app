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
     
        price_start = m.log(1,1.01)
        price_end = m.log(100,1.01)
        amount_start = 0
        amount_end = 100
        now = datetime.now()
        h9 = datetime.strptime(f'{now.date()} 09:30:00','%Y-%m-%d %H:%M:%S')
        h11 = datetime.strptime(f'{now.date()} 11:30:00','%Y-%m-%d %H:%M:%S')
        h13 = datetime.strptime(f'{now.date()} 13:00:00','%Y-%m-%d %H:%M:%S')
        h15 = datetime.strptime(f'{now.date()} 15:00:00','%Y-%m-%d %H:%M:%S')
        time_start = 0
        time_end = (h11 - h9).total_seconds() + (h15 - h13).total_seconds()


        # 创建Ruler 作为 x，y，z轴的可视化表示
        self.axis_x = gfx.Ruler(start_pos=(time_start,0,0), end_pos=(time_end,0,0))
        self.axis_x.local.scale_x = 1 / (time_end-time_start)
        self.add(self.axis_x)

        self.axis_y = gfx.Ruler(start_pos=(0,0,0), end_pos=(0,price_end,0))
        self.axis_y.local.scale_y = 1 / (price_end-price_start)
        self.add(self.axis_y)

        axis_z = gfx.Ruler(start_pos=(0,0,amount_start), end_pos=(0,0,amount_end))
        axis_z.local.x = axis_z.local.y = 1
        axis_z.local.scale_z = 1 / (amount_end-amount_start)
        self.add(axis_z)

        grid_xy = gfx.Grid(
            gfx.box_geometry(),
            gfx.GridMaterial(
                major_step=(1,1),
                minor_step=(0.1,10 / (price_end-price_start)),
                thickness_space="world",
                axis_thickness=0.005,
                major_thickness=0.005,
                minor_thickness=0.001,
                infinite=False
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

        rows = []
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
                    row = line.split(',')
                    if rows: self.add_feature(row)
                    func(line)
                    rows.append(row)
                else:
                    self.process_measure.stdin.write('exit\n')
                    self.process_measure.stdin.flush()

        threading.Thread(target=f,daemon=True).start()

    def add_feature(self,row):
        编号,起时,终时,起价,终价,总价,均价,均点,起点,终点 = *row,

        now = datetime.now()
        h9 = datetime.strptime(f'{now.date()} 09:30:00','%Y%m%d %H:%M:%S')
        h15 = datetime.strptime(f'{now.date()} 15:00:00','%Y%m%d %H:%M:%S')
        
        # 如何得到当前时间的秒数
        起时 = datetime.strptime(起时,'%Y%m%d %H:%M:%S')
        起时 = (起时 - 起时).total_seconds()
    
        h9 = datetime.strptime(f'{now.date()} 09:30:00','%Y-%m-%d %H:%M:%S')
        h11 = datetime.strptime(f'{now.date()} 11:30:00','%Y-%m-%d %H:%M:%S')
        h13 = datetime.strptime(f'{now.date()} 13:00:00','%Y-%m-%d %H:%M:%S')
        h15 = datetime.strptime(f'{now.date()} 15:00:00','%Y-%m-%d %H:%M:%S')

        起时 = (起时 - h9).total_seconds() - (h13 - h11).total_seconds()
        
        geom = gfx.Geometry([[起时,起点],[起时,终点]])
        mater = gfx.LineMaterial(color="red", linewidth=2)
        line = gfx.Line(geom,mater)
        line.local.scale_x = self.axis_x.local.scale_x
        line.local.scale_y = self.axis_y.local.scale_y
        self.add(line)
        