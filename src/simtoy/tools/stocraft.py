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

        self.up_process = None
        self.stocks = None

    def step(self, dt: float,camera: gfx.Camera, canvas : RenderCanvas):
        for ob in self.children:
            if isinstance(ob, gfx.Ruler):
                ob.update(camera, canvas.get_logical_size())
        
        if not self.steps:
            return 
        
        f = self.steps.pop(0)
        if f(): self.steps.append(f)

    def cmd_up(self, days = 7, func = None):
        if self.up_process:
            self.up_process.terminate()

        # 得到当天的时间，如果超过15点，取当天，否则取前一天
        now = datetime.now()
        if now.hour >= 15:
            date = now.date().strftime('%Y%m%d')
        else:
            date = (now.date() - timedelta(days=1)).strftime('%Y%m%d')

        file = files("simtoy.data.stocraft") / "stocraft.py"
        self.up_process = sp.Popen(["python", file.as_posix(), 'up','--date',f'{date}','--days',f'{days}'],stdin=sp.PIPE,stdout=sp.PIPE,encoding='utf-8')
        print(' '.join(["python", file.as_posix(), 'up','--date',f'{date}','--days',f'{days}']))

        def f():
            while True:
                line = self.up_process.stdout.readline().strip()
                if line: 
                    print(line,flush=True)
                    if line == '-': 
                        self.up_process.stdin.write('exit\n')
                        self.up_process.stdin.flush()

                    if self.stocks is None:
                        # 第一行是列名，用它创建DataFrame，然后填充N行数据
                        columns = line.split(',')
                        self.stocks = pd.DataFrame(columns=columns,data=np.full((10000,len(columns)),''),index=range(10000),dtype=str)
                    else:
                        # 后续行是数据，用它添加到DataFrame，第0列是索引，从1开始
                        row = line.split(',')
                        self.stocks.loc[len(self.stocks)] = row[1:]
                        func(self.stocks,row)

                if self.up_process.poll() is not None: break

        # self.steps.append(f)
        threading.Thread(target=f,daemon=True).start()

    def cmd_measure(self,code):
        if self.process:
            self.process.terminate()

        df = pd.DataFrame()

        file = files("simtoy.data.stocraft") / "stocraft.py"
        p = sp.Popen(["python", file.as_posix(), 'measure','--code', code],stdin=sp.PIPE,stdout=sp.PIPE,encoding='utf-8')

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
