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
        amount_end = 10
        now = datetime.now()
        h9 = datetime.strptime(f'{now.date()} 09:30:00','%Y-%m-%d %H:%M:%S')
        h11 = datetime.strptime(f'{now.date()} 11:30:00','%Y-%m-%d %H:%M:%S')
        h13 = datetime.strptime(f'{now.date()} 13:00:00','%Y-%m-%d %H:%M:%S')
        h15 = datetime.strptime(f'{now.date()} 15:00:00','%Y-%m-%d %H:%M:%S')
        time_start = 0
        time_end = ((h11 - h9).total_seconds() + (h15 - h13).total_seconds()) / 60

        # 创建Ruler 作为 x，y，z轴的可视化表示
        self.axis_x = gfx.Ruler(start_pos=(time_start,0,0), end_pos=(time_end,0,0))
        self.axis_x.local.scale_x = 1 / (time_end-time_start)
        self.add(self.axis_x)

        self.axis_y = gfx.Ruler(start_pos=(0,0,0), end_pos=(0,price_end,0),tick_format=lambda v, mi, ma: str(round(m.pow(1.01,v),2)))
        self.axis_y.local.scale_y = 1 / (price_end-price_start)
        self.add(self.axis_y)

        self.axis_z = gfx.Ruler(start_pos=(0,0,amount_start), end_pos=(0,0,amount_end))
        self.axis_z.local.x = self.axis_z.local.y = 1
        self.axis_z.local.scale_z = 1 / (amount_end-amount_start) * 0.5
        self.add(self.axis_z)

        grid_xy = gfx.Grid(
            gfx.box_geometry(),
            gfx.GridMaterial(
                major_step=(1,10 / (price_end-price_start)),
                minor_step=(1 / (time_end-time_start),2 / (price_end-price_start)),
                thickness_space="world",
                axis_thickness=0.005,
                major_thickness=0.002,
                minor_thickness=0.001,
                infinite=False
            ),
            orientation="xy",
        )
        grid_xy.local.z = -0.001
        self.add(grid_xy)

        self.process_measure = self.process_up = None
        self.line_blue = None

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

        now = datetime.now()
        if now.hour >= 15:
            date = now.date().strftime('%Y%m%d')
        else:
            date = (now.date() - timedelta(days=1)).strftime('%Y%m%d')

        def f():
            file = files("simtoy.data.stocraft") / "stocraft.py"
            self.process_up = sp.Popen(["python", file.as_posix(), 'up','--date',f'{date}','--days',f'{days}'],stdin=sp.PIPE,stdout=sp.PIPE,encoding='utf-8')
            print(' '.join(["python", file.as_posix(), 'up','--date',f'{date}','--days',f'{days}']),flush=True)

            while self.process_up.poll() is None:
                line = self.process_up.stdout.readline().strip()
                if not line: continue 
                if line != '-' and line != ">": 
                    print(line)
                    func(line)
                else:
                    self.process_up.stdin.write('exit\n')
                    self.process_up.stdin.flush()

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

            deals = []
            feats = []
            swt = 'd'
            while self.process_measure.poll() is None:
                line = self.process_measure.stdout.readline().strip()
                if not line: continue 
                print(line)
                if line in ['f','d']: swt = line
                elif line not in ['-','>']: 
                    if swt == 'd': 
                        if deals: self.add_deal(line.split(','))
                        deals.append(line)
                    if swt == 'f':
                        if feats: self.add_feature(line.split(','))
                        feats.append(line)
                else:
                    self.process_measure.stdin.write('exit\n')

        threading.Thread(target=f,daemon=True).start()

    def add_deal(self,row):
        编号,时间,价格,手数,性质,成交点 = *row,
        
        now = datetime.now()
        h9 = datetime.strptime(f'{now.date()} 09:30:00','%Y-%m-%d %H:%M:%S')
        h11 = datetime.strptime(f'{now.date()} 11:30:00','%Y-%m-%d %H:%M:%S')
        h13 = datetime.strptime(f'{now.date()} 13:00:00','%Y-%m-%d %H:%M:%S')
        h15 = datetime.strptime(f'{now.date()} 15:00:00','%Y-%m-%d %H:%M:%S')
        t = 240 - ((h15 - datetime.strptime(时间,'%Y%m%d %H:%M:%S')).total_seconds() - (h13 - h11).total_seconds()) / 60
        p = float(成交点)

        if self.line_blue:
            print([t,p])
            self.line_blue.geometry = gfx.Geometry(positions=np.vstack((self.line_blue.geometry.positions.data,[[t,p,0]]),dtype=np.float32))
            self.line_blue.geometry.update()
        else:
            geom = gfx.Geometry(positions=[[t,p,0]])
            mater = gfx.LineMaterial(thickness=2,color="blue")
            self.line_blue = gfx.Line(geom,mater)
            self.line_blue.local.scale_x = self.axis_x.local.scale_x
            self.line_blue.local.scale_y = self.axis_y.local.scale_y
            self.line_blue.local.scale_z = self.axis_z.local.scale_z
            self.line_blue.local.z = 0.001
            self.add(self.line_blue)

    def add_feature(self,row):
        编号,起时,终时,起价,终价,总价,均价,均点,起点,终点 = *row,

        now = datetime.now()
        h9 = datetime.strptime(f'{now.date()} 09:30:00','%Y-%m-%d %H:%M:%S')
        h11 = datetime.strptime(f'{now.date()} 11:30:00','%Y-%m-%d %H:%M:%S')
        h13 = datetime.strptime(f'{now.date()} 13:00:00','%Y-%m-%d %H:%M:%S')
        h15 = datetime.strptime(f'{now.date()} 15:00:00','%Y-%m-%d %H:%M:%S')
        t = 240 - ((h15 - datetime.strptime(起时,'%Y%m%d %H:%M:%S')).total_seconds() - (h13 - h11).total_seconds()) / 60
        p0,p1 = float(起点),float(终点)
        p = float(均点)

        geom = gfx.Geometry(positions=[[t,p0,0],[t,p1,0]])
        mater = gfx.LineMaterial(thickness=2,color="red" if p0 < p1 else "green")
        line = gfx.Line(geom,mater)
        line.local.scale_x = self.axis_x.local.scale_x
        line.local.scale_y = self.axis_y.local.scale_y
        line.local.z = 0.001
        
        self.add(line)

        c = float(总价) / 1e8
        
        geom = gfx.Geometry(positions=[[t,p,0],[t,p,c]])
        mater = gfx.LineMaterial(thickness=1,color="red" if p0 < p1 else "green")
        line = gfx.Line(geom,mater)
        line.local.scale_x = self.axis_x.local.scale_x
        line.local.scale_y = self.axis_y.local.scale_y
        line.local.scale_z = self.axis_z.local.scale_z
        line.local.z = 0.001

        text = gfx.Text(text=f'幅度：{p1-p0}点\n代价：{c:.2f}亿\n支撑价：{均价}',screen_space=True)
        text.local.position = [t,p,c]
        line.add(text)
        self.add(line)