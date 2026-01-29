import multiprocessing as mp
import akshare as ak
import pandas as pd
import numpy as np
import math as m
import time
import os
import sys
import argparse
import requests

from datetime import datetime,timedelta
from requests.exceptions import ReadTimeout
from queue import Queue

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer,encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer,encoding='utf-8')

import threading
import multiprocessing as mp
from multiprocessing.managers import ValueProxy,ListProxy,DictProxy

import glob

db_dir = 'db'

def feature(交易,已有特征=None,interval_seconds = 10):
    if 已有特征 is None:
        已有特征 = pd.DataFrame(columns=['起时','终时','起价','终价','总价','均价','均点','起点','终点'])

    # 把交易的价格全部换成对数，底数为1.01
    交易['成交点'] = round(np.log(交易['成交价']) / np.log(1.01))

    rows = 已有特征.values.tolist()
    i = 0
    for _,r in 交易.iterrows():
        时间,成交价,手数,买卖盘性质,成交点 = r['时间'],r['成交价'],r['手数'],r['买卖盘性质'],r['成交点']
        if 买卖盘性质 == '中性盘' or 手数 == 0:
            continue

        if i == 0:
            起时 = 终时 = 时间
            起价 = 终价 = 成交价
            总量 = 手数 * 100
            总价 = 成交价 * 总量
            均价 = 总价 / 总量
            均点 = m.log(均价,1.01)
            起点 = 终点 = round(成交点)
            当前 = 起时,终时,起价,终价,总价,均价,买卖盘性质,起点,终点
            i += 1
        else:
            _,终时,_,终价,_,均价,性质,_,终点 = 当前

            time1 = datetime.strptime(终时, "%H:%M:%S")
            time2 = datetime.strptime(时间, "%H:%M:%S")
            time_diff = time2 - time1
            if (买卖盘性质 != 性质 and 成交点 == 终点):
                买卖盘性质 = 性质

            if time_diff.total_seconds() > interval_seconds or (性质 != 买卖盘性质 and 成交点 != 终点):
                均价 = 总价 / 总量
                均点 = m.log(总价 / 总量,1.01)
            
                if len(rows):
                    近起时,近终时,近起价,近终价,近总价,近均价,近均点,近起点,近终点 = rows[-1]

                    time1 = datetime.strptime(近终时, "%H:%M:%S")
                    time2 = datetime.strptime(时间, "%H:%M:%S")
                    time_diff = time2 - time1

                    if time_diff.total_seconds() < interval_seconds or 近终点 == 成交点:
                        近起时,近终时,近起价,近终价,近总价,近均价,近均点,近起点,近终点

                        起时 = 近起时
                        终时 = 时间
                        起价 = 近起价
                        终价 = 成交价
                        近权重 = 近总价 / (近总价 + 总价)
                        权重 = 总价 / (近总价 + 总价)
                        均价 = 总价 / 总量
                        均价 = 近均价 * 近权重 + 均价 * 权重
                        总价 = 近总价 + 总价
                        均点 = 近均点 * 近权重 + 均点 * 权重
                        起点 = 近起点
                        终点 = 成交点
                        rows.pop()
                rows.append((起时,终时,起价,终价,round(总价),round(均价,2),round(均点),round(起点),round(终点)))
                总量 = 0
                总价 = 0
                起时 = 时间
                起价 = 终价
                起点 = 终点

            总价 += 成交价 * 手数 * 100
            总量 += 手数 * 100
            终时 = 时间
            终价 = 成交价
            终点 = 成交点
            当前 = 起时,终时,起价,终价,总价,均价,买卖盘性质,起点,终点
    
    df = pd.DataFrame(columns=['起时','终时','起价','终价','总价','均价','均点','起点','终点'],data=rows)
    return df


# def measure(code,date,info,freq=10):
#     df = pd.DataFrame(columns=['时间','买手','卖手','买额','卖额','涨跌','价格'])
    
#     transaction_filepath = os.path.join(db_dir,f'{code}-{date}-交易.csv')
#     if not os.path.exists(transaction_filepath):
#         return pd.DataFrame(columns=['起时','终时','起价','终价','总价','均价','涨幅'])

#     交易 = pd.read_csv(transaction_filepath)
#     基价 = info['昨收']
#     起价 = 基价

#     交易 = 交易[(交易['买卖盘性质'] != '中性盘')].copy()
    
#     if 0 == 交易.shape[0]: return df
#     交易 : pd.DataFrame

#     交易['金额'] = 交易['成交价'] * 交易['手数'] * 100
#     交易['_时间'] = pd.to_datetime(交易['时间'])
#     交易.set_index('_时间',inplace=True)
#     date = info[info['item'] == '日期'].iloc[0]['value']
#     价格 = 昨收价 = float(info[info['item'] == '昨收'].iloc[0]['value'])
#     涨跌 = 0
#     window_size = int(60 / freq * 2.5)
#     last_time = '9:25'
#     a = pd.date_range(date + ' 9:25',date + ' 15:00',freq=f'{freq}s',inclusive='right')
#     b = pd.date_range(date + ' 11:30',date + ' 13:00',freq=f'{freq}s',inclusive='right')
#     for i,cur in enumerate(a.difference(b)):
#         cur_time = cur.strftime('%H:%M:%S')
#         买卖盘 = 交易.between_time(last_time,cur_time)
#         买盘 = 买卖盘[买卖盘['买卖盘性质'] == '买盘']
#         卖盘 = 买卖盘[买卖盘['买卖盘性质'] == '卖盘']
#         买手 = 买盘['手数']
#         卖手 = 卖盘['手数']
#         买总手 = round(买手.sum(),1)
#         卖总手 = round(卖手.sum(),1)
#         买盘金额 = 买盘['金额']
#         卖盘金额 = 卖盘['金额']
#         买总额 = round(买盘金额.sum() / 1e7,1)
#         卖总额 = round(卖盘金额.sum() / 1e7,1)
#         成交价 = 买卖盘['成交价']        
        
#         if not 成交价.empty:
#             涨跌 = round((成交价.iloc[-1] / 昨收价 - 1) * 100,2)
#             价格 = 成交价.iloc[-1]
        
#         if i < window_size:
#             df.loc[i] = (date + ' ' + cur_time,买总手/window_size,卖总手/window_size,买总额/window_size,卖总额/window_size,涨跌,价格)
#         else:
#             df.loc[i] = (date + ' ' + cur_time,买总手,卖总手,买总额,卖总额,涨跌,价格)

#         if cur > 交易.index.values[-1]:
#             break

#         last_time = cur_time

#     # def gaussian_weights(n):
#     #     x = np.linspace(-1, 0, n)
#     #     sigma = 0.5
#     #     weights = np.exp(-(x ** 2) / (2 * sigma ** 2))
#     #     return weights
    
#     # weights = gaussian_weights(window_size)
#     # df[['买流量', '卖流量']] = df[['买额', '卖额']].rolling(window=window_size).apply(lambda x: np.sum(x * weights) / np.sum(weights), raw=True).round(1)
#     # df.loc[df.index[:window_size],'买流量'] = np.linspace(0,df.loc[0,'买额'],window_size)
#     # df.loc[df.index[:window_size],'卖流量'] = np.linspace(0,df.loc[0,'卖额'],window_size)
    
#     return df

def measure(交易,freq=10):
    df = pd.DataFrame(columns=['时间','买手','卖手','买额','卖额','涨跌','价格'])
    
    # 基价 = 信息['昨收']
    # 起价 = 基价

    交易 = 交易[(交易['买卖盘性质'] != '中性盘')].copy()
    
    if 0 == 交易.shape[0]: return df
    交易 : pd.DataFrame

    交易['金额'] = 交易['成交价'] * 交易['手数'] * 100
    交易['_时间'] = pd.to_datetime(交易['时间'])
    交易.set_index('_时间',inplace=True)
    # 价格 = 昨收价 = 基价
    涨跌 = 0
    window_size = int(60 / freq * 2.5)
    last_time = '9:25'
    a = pd.date_range(date + ' 9:25',date + ' 15:00',freq=f'{freq}s',inclusive='right')
    b = pd.date_range(date + ' 11:30',date + ' 13:00',freq=f'{freq}s',inclusive='right')
    for i,cur in enumerate(a.difference(b)):
        cur_time = cur.strftime('%H:%M:%S')
        买卖盘 = 交易.between_time(last_time,cur_time)
        买盘 = 买卖盘[买卖盘['买卖盘性质'] == '买盘']
        卖盘 = 买卖盘[买卖盘['买卖盘性质'] == '卖盘']
        买手 = 买盘['手数']
        卖手 = 卖盘['手数']
        买总手 = round(买手.sum(),1)
        卖总手 = round(卖手.sum(),1)
        买盘金额 = 买盘['金额']
        卖盘金额 = 卖盘['金额']
        买总额 = round(买盘金额.sum() / 1e7,1)
        卖总额 = round(卖盘金额.sum() / 1e7,1)
        成交价 = 买卖盘['成交价']        
        
        if not 成交价.empty:
            涨跌 = round((成交价.iloc[-1] / 昨收价 - 1) * 100,2)
            价格 = 成交价.iloc[-1]
        
        if i < window_size:
            df.loc[i] = (date + ' ' + cur_time,买总手/window_size,卖总手/window_size,买总额/window_size,卖总额/window_size,涨跌,价格)
        else:
            df.loc[i] = (date + ' ' + cur_time,买总手,卖总手,买总额,卖总额,涨跌,价格)

        if cur > 交易.index.values[-1]:
            break

        # last_time = cur_time

    # def gaussian_weights(n):
    #     x = np.linspace(-1, 0, n)
    #     sigma = 0.5
    #     weights = np.exp(-(x ** 2) / (2 * sigma ** 2))
    #     return weights
    
    # weights = gaussian_weights(window_size)
    # df[['买流量', '卖流量']] = df[['买额', '卖额']].rolling(window=window_size).apply(lambda x: np.sum(x * weights) / np.sum(weights), raw=True).round(1)
    # df.loc[df.index[:window_size],'买流量'] = np.linspace(0,df.loc[0,'买额'],window_size)
    # df.loc[df.index[:window_size],'卖流量'] = np.linspace(0,df.loc[0,'卖额'],window_size)
    
    return df

def evaluate(code,info,dates : list):
    # 从info中得到股票的换手率和成交量得到总量
    换手率 = info['换手率']
    成交量 = info['成交量']
    总量 = 成交量 / (换手率 / 100) if 换手率 else 0
    
    评分 = 0
    散户 = []
    游资 = []
    主力 = []
    庄家 = []
    套牢量 = 0.0
    套牢资金 = 0.0
    均价 = set()
    均点 = set()
    最新日期 = None
    最新价格 = None
    最新点 = None
    try:
        for i,date in enumerate(dates):
            filepath = os.path.join(db_dir,f'{code}-{date}-交易.csv')
            if not os.path.exists(filepath): i-=1; continue

            交易 = pd.read_csv(filepath,dtype={'代码':str})
            特征 = feature(交易)

            if not 最新价格:
                # 得到最后的价格
                最新价格 = 交易['成交价'].iloc[-1]
                最新点 = m.log(最新价格,1.01)
                最新日期 = date
            
            # 统计特征中，大单的区间，500万以下，500~1000万，1000~2000万，2000万以上的数量，分别用散户，游资，主力，庄家来表示。
            买入特征 = 特征[(特征['终点'] - 特征['起点'] > 1)]

            散户 += 买入特征[买入特征['总价'] <= 5e7].values.tolist()
            游资 += 买入特征[(买入特征['总价'] > 5e7) & (买入特征['总价'] <= 1e8)].values.tolist()
            主力特征 = 买入特征[买入特征['总价'] > 1e8]
            主力 += 主力特征.values.tolist()
            庄家特征 = 买入特征[买入特征['总价'] > 2e8]
            庄家 += 庄家特征.values.tolist()

            # 记录支撑价
            均价.update(庄家特征['均价'].values.tolist())
            均价.update(主力特征['均价'].values.tolist())
            均点.update(庄家特征['均点'].values.tolist())
            均点.update(主力特征['均点'].values.tolist())
            
            # 计算出这个价格之上有多少套牢盘
            套牢盘 = 交易[(交易['买卖盘性质'] != '中性盘') & (交易['成交价'] > 最新价格)]
            总价 = 套牢盘['成交价']  * 套牢盘['手数'] * 100
            套牢资金 += 总价.sum() / (i+1)
            套牢量 += (套牢盘['手数'].sum() * 100 / 总量) / (i+1)


    except:
        import traceback
        traceback.print_exc()
        print('评估失败',code,info['名称'],date,flush=True, file=sys.stderr)
        评分 = 0


    # 根据不同资金量的数量，给与不同的权重分数，庄家权重最大，散户最小。
    散户权重 = 0.05
    游资权重 = 0.2
    主力权重 = 0.5
    庄家权重 = 2.0
    套牢权重 = 0.05
    支撑权重 = 0.1

    # 计算最新点 在均点的
    支撑 = 0
    if 均点: 支撑 = np.mean(list(均点)) - 最新点

    if m.isnan(支撑) or 支撑 < 0: 支撑 = 0

    评分 = len(散户) * 散户权重 + len(游资) * 游资权重 + len(主力) * 主力权重 + len(庄家) * 庄家权重 + 套牢量 * 套牢权重 + 支撑 * 支撑权重
    if 评分 == 0: return None

    r = {
        '日期': 最新日期,
        '代码': info['代码'],
        '名称': info['名称'],
        '评分': float(round(评分,2)),
        '散户&游资&主力&庄家': f'[{len(散户)},{len(游资)},{len(主力)},{len(庄家)}]',
        '套牢量': f'{float(round(套牢量,2))}%',
        '套牢盘': f'{float(round(套牢资金/1e8,1))}亿',
        '均价': list(均价),
        '支撑': round(支撑,1)
    }
    return r

def up(worker_req : mp.Queue,*args):
    worker_res = shared.Queue()
    codes,date,days,cap = args

    get_stock_spot()
    stocks = pd.read_csv(os.path.join(db_dir,f'0-0-行情.csv'),dtype={'代码':str})
    stocks_seleted = stocks[stocks['代码'].isin(codes)]
    stocks = stocks[(stocks['流通市值'] >= cap[0] * 1e8) & (stocks['流通市值'] <= cap[1] * 1e8)]
    stocks = pd.concat([stocks, stocks_seleted], ignore_index=True).drop_duplicates()

    end = datetime.strptime(date,'%Y%m%d')
    start = end - timedelta(days)
    dates = [d.strftime('%Y%m%d') for d in pd.date_range(start,end,freq='1D')][1:]
    dates.reverse()
    stocks.apply(lambda r: worker_req.put((worker_res,'evaluate',r['代码'],r,dates)), axis=1)

    rows = []
    for i in range(stocks.shape[0]):
        _,_,_,_,res = worker_res.get()
        res : dict    
        if not res: continue 
        if i == 0: print(' ',','.join(res.keys()),flush=True)
        print(f'{i},{','.join(str(v) for v in res.values())}',flush=True)
        # rows.append(res)
    return pd.DataFrame(rows)
    
def play(worker_req : mp.Queue,codes,date,days):
    start = datetime.strptime(date,'%y%m%d')
    end = start - timedelta(days)
    if days < 0: start,end = end,start
    dates = [d.strftime('%Y%m%d') for d in pd.date_range(start,end,freq='1D')]

    for date in dates:
        up(worker_req,codes,date,-3,'')

def get_stock_spot():
    h15 = datetime.now().replace(hour=15, minute=0, second=0)

    stocks_file_path = os.path.join(db_dir,f'0-0-行情.csv')
    # 判断文件时间是否超过一周，超过则重新获取，否则使用现有文件
    if os.path.exists(stocks_file_path) and h15.date() - datetime.fromtimestamp(os.path.getmtime(stocks_file_path)).date() > timedelta(7):
        stocks = ak.stock_zh_a_spot_em()
        condition = ((stocks['代码'].str.startswith('00')) | (stocks['代码'].str.startswith('60'))) & \
                    (~stocks['名称'].str.startswith('PT')) & \
                    (~stocks['名称'].str.startswith('ST')) & \
                    (~stocks['名称'].str.startswith('*ST')) & \
                    (~stocks['最新价'].isnull()) & \
                    (stocks['换手率'] > 0.001)
        stocks = stocks[condition].reset_index(drop=True)
        stocks.to_csv(stocks_file_path,index=False)
    else:
        stocks = pd.read_csv(stocks_file_path,dtype={'代码':str})
        
    # try:
    #     if os.path.exists(stocks_file_path) and h15.date() == datetime.fromtimestamp(os.path.getmtime(stocks_file_path)).date():
    #         stocks = pd.read_csv(stocks_file_path,dtype={'代码':str})
    #     elif not os.path.exists(stocks_file_path) or h15.weekday() < 5 and h15 < datetime.now():
    #         stocks = ak.stock_zh_a_spot_em()
    #         condition = ((stocks['代码'].str.startswith('00')) | (stocks['代码'].str.startswith('60'))) & \
    #                     (~stocks['名称'].str.startswith('PT')) & \
    #                     (~stocks['名称'].str.startswith('ST')) & \
    #                     (~stocks['名称'].str.startswith('*ST')) & \
    #                     (~stocks['最新价'].isnull()) & \
    #                     (stocks['换手率'] > 0.001)
    #         stocks = stocks[condition].reset_index(drop=True)
    #         stocks.to_csv(stocks_file_path,index=False)
    #         spot_filepath = os.path.join(db_dir,f'0-{h15.strftime("%Y%m%d")}-行情.csv')
    #         stocks.to_csv(spot_filepath,index=False) 
    #     else:
    #         raise Exception('获取股票行情失败')
    # except:
    #     file_paths = glob.glob(os.path.join(db_dir, '0-*-行情.csv'))
    #     if not file_paths:
    #         return None
        
    #     stocks = pd.read_csv(file_paths[-1],dtype={'代码':str})

    return stocks

def sync_stock_intraday(code,date):
    filepath = os.path.join(db_dir,f'{code}-{date}-交易.csv')
    if not os.path.exists(filepath) or 0 == os.path.getsize(filepath):
        df = get_stock_intraday(code)
        df.to_csv(filepath,index=False)

def get_stock_info(code):
    i = 1
    while i:
        try: df = ak.stock_bid_ask_em(code)
        except: i += 1
        else: i = 0
    return df

def get_stock_intraday(code):
    i = 1
    while i:
        try: df = ak.stock_intraday_em(code)
        except: i += 1
        else: i = 0
    return df

def worker(id,req : mp.Queue):
    while True:
        args = req.get()
        res : mp.Queue = args[0]
        fun = args[1]
        val = eval(f'{fun}(*args[2:])')
        if res: res.put((*args[1:],val))

def data_syncing_of_stock_intraday(worker_req : mp.Queue,log : list):
    while True:
        os.makedirs(db_dir,exist_ok=True)

        now = datetime.now()
        h9 = now.replace(hour=9, minute=15, second=0)
        h11 = now.replace(hour=11, minute=30, second=0)
        h13 = now.replace(hour=13, minute=0, second=0)
        h15 = now.replace(hour=15, minute=0, second=0)
        h24 = now.replace(hour=23, minute=59, second=59)

        while datetime.now() < h15:
            time.sleep(15)
            continue
        
        stocks = get_stock_spot()
        date = now.strftime('%Y%m%d')
        start = now.replace(hour=9, minute=30, second=0).strftime('%Y-%m-%d %H:%M:%S')
        end = now.replace(hour=9, minute=40, second=0).strftime('%Y-%m-%d %H:%M:%S')
        while True:
            try:df = ak.stock_zh_a_hist_min_em(symbol="000001", start_date=start, end_date=end, period="5", adjust="")
            except:continue
            else:break

        log[:] = []
        if now.weekday() < 5 and not df.empty: 
            stocks = stocks[(stocks['流通市值'] >= 0 * 1e8) & (stocks['流通市值'] <= 60 * 1e8)].reset_index(drop=True)

            for i,r in stocks.iterrows():
                while worker_req.qsize() > os.cpu_count(): time.sleep(1)
                worker_req.put((None,'sync_stock_intraday',r['代码'],date))
                log.append(f'{int(i)+1}/{stocks.shape[0]} {r["代码"]}-{r["名称"]}')

        while datetime.now() < h24:
            time.sleep(60)
            continue

if __name__ == '__main__':
    shared = mp.Manager()
    worker_req = shared.Queue()
    log = shared.list()
    pi = 0
    
    while True:
        if len(sys.argv) > 1:
            cmd = sys.argv[1:]
            sys.argv = sys.argv[0:1]
        else:
            try:
                cmd = input('> ').strip().split(' ')
            except (EOFError,KeyboardInterrupt):
                break

        parser = argparse.ArgumentParser()
        parser.add_argument('mode', type=str, help='The mode to run the program')
        parser.add_argument('--name',type=str,help='股票名称')
        parser.add_argument('--code',type=str,help='股票代码',default='')
        parser.add_argument('--date',type=str,default=datetime.now().strftime('%Y%m%d'))
        parser.add_argument('--days',type=int,default=1)
        parser.add_argument('--cap',nargs=2,type=float,default=[0,60],help='流通市值范围，单位：亿')
        args = parser.parse_args(cmd)
        
        if args.mode == 'sync':
            for i in range(pi,os.cpu_count()):
                process = mp.Process(target=worker,name=f'牛马-{i}',args=[i,worker_req],daemon=True)
                process.start()
                pi += 1
            threading.Thread(target=data_syncing_of_stock_intraday,args=[worker_req,log],name='股票数据同步',daemon=True).start()
        elif args.mode == 'up':
            for i in range(pi,os.cpu_count()):
                process = mp.Process(target=worker,name=f'牛马-{i}',args=[i,worker_req],daemon=True)
                process.start()
                pi += 1
            codes = args.code.split(',')
            up(worker_req,codes,args.date,args.days,args.cap)
        elif args.mode == 'evaluate':
            stocks = pd.read_csv(os.path.join(db_dir,f'0-{args.date}-行情.csv'),dtype={'代码':str})
            stocks_seleted = stocks[stocks['代码'] == args.code]
            info = stocks_seleted.iloc[0]
            print(evaluate(args.code,info,[args.date]))
        elif args.mode == 'measure':
            已有交易 = None
            已有特征 = None
            
            h9 = datetime.now().replace(hour=9, minute=15, second=0)
            h15 = datetime.now().replace(hour=15, minute=0, second=0)
            
            if datetime.now() < h9: continue

            交易 = get_stock_intraday(args.code)
            # 流量 = measure(交易)

            if 已有交易 is None:
                新增交易 = 交易
                print(','.join(交易.columns))
            else:
                index_diff = 交易.index.difference(已有交易.index)
                新增交易 = 交易.loc[index_diff]

            if not 新增交易.empty: print(新增交易.to_csv(header=False))
            print('-')
            特征 = feature(新增交易,已有特征)

            if 已有特征 is None:
                新增特征 = 特征
                print(','.join(特征.columns))
            else:
                index_diff = 特征[特征.index.difference(已有特征.index)]
                新增特征 = 特征.loc[index_diff]
            
            if not 新增特征.empty: print(新增特征.to_csv(header=False))
            
            已有交易 = 交易
            已有特征 = 特征
        elif args.mode == 'test':
            pass
        elif args.mode == 'exit':
            break
        else:
            while len(log):
                print(log.pop(0))
        print('-')
