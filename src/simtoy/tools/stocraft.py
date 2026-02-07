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
        self.process_measure = self.process_up = None
        self.lines_blue = []
        self.lines_red = []
        self.lines_green = []
        self.lines_feature = []
        self.axes_x = []
        self.axes_y = []
        self.axes_z = []
        self.grids = []
        self.make_box(1)
         
    def __del__(self):
        print('del')

    def make_box(self,day):
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
        axis_x = gfx.Ruler(start_pos=(time_start,0,0), end_pos=(time_end,0,0))
        axis_x.local.scale_x = 1 / (time_end-time_start)
        axis_x.local.x = day - 1
        self.add(axis_x)

        axis_y = gfx.Ruler(start_pos=(0,price_end,0), end_pos=(0,price_start,0),tick_format=lambda v, mi, ma: str(round(m.pow(1.01,v),2)))
        axis_y.local.x = day - 1
        axis_y.local.scale_y = 1 / (price_end-price_start)
        axis_y.local.y = -price_start * axis_y.local.scale_y
        axis_y.start_value = price_start
        self.add(axis_y) 

        axis_z = gfx.Ruler(start_pos=(0,0,amount_start), end_pos=(0,0,amount_end))
        axis_z.local.x = day
        axis_z.local.y = 1
        axis_z.local.scale_z = 1 / (amount_end-amount_start)
        axis_y.add(axis_z)

        grid_xy = gfx.Grid(
            gfx.box_geometry(),
            gfx.GridMaterial(
                major_step=(30,10 / (price_end-price_start)),
                minor_step=(1 / (time_end-time_start),1 / (price_end-price_start)),
                thickness_space="world",
                axis_thickness=0,
                major_thickness=0.0005,
                minor_thickness=0.00025,
                axis_color='white',major_color='white',minor_color='white',
                infinite=False
            ),
            orientation="xy",
        )

        grid_xy.local.x = day - 1
        grid_xy.local.z = -0.001
        self.add(grid_xy)

        self.axes_x.append(axis_x)
        self.axes_y.append(axis_y)
        self.axes_z.append(axis_z)
        self.grids.append(grid_xy)
        
    def step(self, dt: float,camera: gfx.Camera, canvas : RenderCanvas):
        for ob in self.children:
            if isinstance(ob, gfx.Ruler):
                ob.update(camera, canvas.get_logical_size())
        
        if not self.steps:
            return 
        
        f = self.steps.pop(0)
        if f(): self.steps.append(f)

    def cmd_up(self, days = 1, func = None):
        self.cmd_up_stop()

        now = datetime.now()
        if now.hour >= 15:
            date = now.date().strftime('%Y%m%d')
        else:
            date = (now.date() - timedelta(days=1)).strftime('%Y%m%d')

        file = files("simtoy.data.stocraft") / "stocraft.py"
        self.process_up = sp.Popen(["python", file.as_posix(),'up','--date',f'{date}','--days',f'{days}'],stdin=sp.PIPE,stdout=sp.PIPE,encoding='utf-8')    
        print(' '.join(["python", file.as_posix(), 'up','--date',f'{date}','--days',f'{days}']),flush=True)

        def f():
            while self.process_up.poll() is None:
                line = self.process_up.stdout.readline().strip()
                if not line: continue 
                print(line,flush=True)
                if line != "-" or line != ">": 
                    func(line)
                else:
                    self.process_up.stdin.write('exit\n')
                    break
            
        threading.Thread(target=f,daemon=True).start()

    def cmd_up_stop(self):
        if self.process_up:
            self.process_up.stdout.close()
            self.process_up = None

    def cmd_measure(self,code,days=1,func=None):

        if self.process_measure:
            self.process_measure.stdout.close()

        def f():
            file = files("simtoy.data.stocraft") / "stocraft.py"
            cmd = ["python", file.as_posix(), 'at','--code', code,'--days', f'{days}']
            self.process_measure = sp.Popen(cmd,stdin=sp.PIPE,stdout=sp.PIPE,encoding='utf-8')
            print(' '.join(cmd),flush=True)

            deals = []
            feats = []
            swt = None
            dates = set()
            
            for obj in self.children: 
                if obj.__class__ in [gfx.Grid,gfx.Ruler]: self.remove(obj)
            while self.process_measure.poll() is None:
                line = self.process_measure.stdout.readline().strip()
                if not line: continue 
                print(line)
                
                if line in ['f','d']: 
                    if line == 'd': self.make_box(len(dates) + 1)
                    swt = line; continue
                
                if line in ['-','>']: 
                    self.process_measure.stdin.write('exit\n')
                    break

                if swt == 'd':
                    row = line.split(',')
                    时间,买额,卖额,买总额,卖总额,成交价,成交点 = *row,

                    if deals: 
                        dates.add(line[:8])
                        day = len(dates)
                        self.add_deal(row,day)
                           
                    deals.append(row)

                    
                if swt == 'f':
                    if feats:
                        pop = False 
                        if line[:17] == feats[-1][:17]: 
                            feats.pop()
                            pop = True
    
                        day = len(dates)
                        self.add_feature(line.split(','),pop,day)

                    feats.append(line)
    
        threading.Thread(target=f,daemon=True).start()

    def add_deal(self,row,day):
        时间,买额,卖额,买总额,卖总额,成交价,成交点 = *row,
        
        for axis_y in self.axes_y:
            axis_y.start_pos[1] = m.floor(min(float(成交点),axis_y.start_pos[1]))
            axis_y.end_pos[1] = m.ceil(max(float(成交点),axis_y.end_pos[1]))
            axis_y.local.scale_y = 1 / (axis_y.end_pos[1] - axis_y.start_pos[1])
            axis_y.local.y = -axis_y.start_pos[1] * axis_y.local.scale_y
            axis_y.start_value = axis_y.start_pos[1]

        for grid in self.grids:
            lim = self.axes_y[day-1].end_pos[1] - self.axes_y[day-1].start_pos[1]
            minor_step = (1 / lim if lim > 0 else 1)
            major_step = (10 / lim if lim > 0 else 1)
            grid.material.major_step = (30 / (self.axes_x[day-1].end_pos[0] - self.axes_x[day-1].start_pos[0]),major_step)
            grid.material.minor_step = (1 / (self.axes_x[day-1].end_pos[0] - self.axes_x[day-1].start_pos[0]),minor_step)


        now = datetime.now()
        date = 时间.split(' ')[0]
        h9 = datetime.strptime(f'{date} 09:30:00','%Y%m%d %H:%M:%S')
        h11 = datetime.strptime(f'{date} 11:30:00','%Y%m%d %H:%M:%S')
        h13 = datetime.strptime(f'{date} 13:00:00','%Y%m%d %H:%M:%S')
        h15 = datetime.strptime(f'{date} 15:00:00','%Y%m%d %H:%M:%S')

        h = datetime.strptime(时间,'%Y%m%d %H:%M:%S')
        t = (h - h9).total_seconds() / 60 + (day-1) * 240
        if h > h13: t = (t - (h13 - h11).total_seconds() / 60)
        p = float(成交点)
        if day != len(self.lines_blue):
            geom = gfx.Geometry(positions=[[t,p,0]])
            mater = gfx.LineMaterial(thickness=1,color="blue")
            line_blue = gfx.Line(geom,mater)
            line_blue.local.z = 0.001
            self.add(line_blue)
            self.lines_blue.append(line_blue)

            geom = gfx.Geometry(positions=[[t,p,(int(买总额)) / 1e8]])
            mater = gfx.LineMaterial(thickness=1,color="red")
            line_red = gfx.Line(geom,mater)
            line_red.local.z = 0.001
            self.add(line_red)
            self.lines_red.append(line_red)

            geom = gfx.Geometry(positions=[[t,p,(int(卖总额)) / 1e8]])
            mater = gfx.LineMaterial(thickness=1,color="green")
            line_green = gfx.Line(geom,mater)
            line_green.local.z = 0.001
            self.add(line_green)
            self.lines_green.append(line_green)

            line_red.local.x = self.axes_x[day-1].local.x
            line_red.local.scale_x = self.axes_x[day-1].local.scale_x
            line_red.local.scale_z = self.axes_z[day-1].local.scale_z
            line_green.local.x = self.axes_x[day-1].local.x
            line_green.local.scale_x = self.axes_x[day-1].local.scale_x
            line_green.local.scale_z = self.axes_z[day-1].local.scale_z
            line_blue.local.x = self.axes_x[day-1].local.x
            line_blue.local.scale_x = self.axes_x[day-1].local.scale_x
            line_blue.local.scale_z = self.axes_z[day-1].local.scale_z
        else:
            self.lines_blue[day-1].geometry = gfx.Geometry(positions=np.vstack((self.lines_blue[day-1].geometry.positions.data,[[t,p,0]]),dtype=np.float32))
            self.lines_red[day-1].geometry = gfx.Geometry(positions=np.vstack((self.lines_red[day-1].geometry.positions.data,[[t,p,int(买总额) / 1e8]]),dtype=np.float32))
            self.lines_green[day-1].geometry = gfx.Geometry(positions=np.vstack((self.lines_green[day-1].geometry.positions.data,[[t,p,int(卖总额) / 1e8]]),dtype=np.float32))
            
            for line in self.lines_red + self.lines_green:
                line.geometry.positions.data[:,1] = self.axes_y[day-1].end_pos[1]
                line.geometry = gfx.Geometry(positions=line.geometry.positions.data)

        for line in self.lines_blue + self.lines_red + self.lines_green:
            line.local.y = self.axes_y[0].local.y
            line.local.scale_y = 1 / (self.axes_y[0].end_pos[1] - self.axes_y[0].start_pos[1])   


    def add_feature(self,row,pop,day):
        if pop:
            self.remove(self.feature_line)
            self.remove(self.feature_line1)

        起时,终时,起价,终价,总价,均价,均点,起点,终点 = *row,

        now = datetime.now()
        date = 起时.split(' ')[0]
        h9 = datetime.strptime(f'{date} 09:30:00','%Y%m%d %H:%M:%S')
        h11 = datetime.strptime(f'{date} 11:30:00','%Y%m%d %H:%M:%S')
        h13 = datetime.strptime(f'{date} 13:00:00','%Y%m%d %H:%M:%S')
        h15 = datetime.strptime(f'{date} 15:00:00','%Y%m%d %H:%M:%S')
        h = datetime.strptime(起时,'%Y%m%d %H:%M:%S')
        t = (h - h9).total_seconds() / 60 + (day-1) * 240
        if h > h13: t = (t - (h13 - h11).total_seconds() / 60)
        p0,p1 = float(起点),float(终点)
        p = float(均点)

        geom = gfx.Geometry(positions=[[t,p0,0],[t,p1,0]])
        mater = gfx.LineMaterial(thickness=4,color="red" if p0 < p1 else "green")
        line0 = gfx.Line(geom,mater)
        line0.local.z = 0.001
        self.add(line0)

        c = float(总价) / 1e8
        
        geom = gfx.Geometry(positions=[[t,p,0],[t,p,c]])
        mater = gfx.LineMaterial(thickness=1,color="red" if p0 < p1 else "green")
        line1 = gfx.Line(geom,mater)
        line1.local.z = 0.001

        if p1 - p0 >= 2:
            text = gfx.Text(family='KaiTi',text=f'幅度：{round(p1-p0,2)}点\n代价：{c:.2f}亿\n支撑价：{均价}',screen_space=True)
            text.local.position = [t,p,c]
            line1.add(text)

        self.lines_feature.append((line0,line1))
        self.add(line1)

        line1.local.x = self.axes_x[day-1].local.x 
        line1.local.y = self.axes_y[day-1].local.y 
        line1.local.z = self.axes_x[day-1].local.z
        line0.local.x = self.axes_x[day-1].local.x 
        line0.local.y = self.axes_y[day-1].local.y 
        line0.local.z = self.axes_x[day-1].local.z
        line1.local.scale_x = self.axes_x[day-1].local.scale_x
        line1.local.scale_y = self.axes_y[day-1].local.scale_y
        line1.local.scale_z = self.axes_z[day-1].local.scale_z
        line0.local.scale_x = self.axes_x[day-1].local.scale_x
        line0.local.scale_y = self.axes_y[day-1].local.scale_y
        line0.local.scale_z = self.axes_z[day-1].local.scale_z
