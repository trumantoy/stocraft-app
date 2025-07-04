import multiprocessing as mp
import akshare as ak
import matplotlib,matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math as m
import traceback as tb
import time
import os
import sys
import argparse
import shutil
import requests

from datetime import datetime,timedelta
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.dates import DateFormatter,HourLocator,SecondLocator,date2num
from matplotlib.ticker import MultipleLocator
from requests.exceptions import ReadTimeout

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer,encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer,encoding='utf-8')

db_dir = 'db'

def sync():
    os.makedirs(db_dir,exist_ok=True)
    now = datetime.now(); date = now.strftime('%Y%m%d')
    
    h9 = datetime.now().replace(hour=9, minute=16, second=0)
    if datetime.now() < h9:
        return
        
    stocks = ak.stock_zh_a_spot_em()
    condition = ((stocks['代码'].str.startswith('00')) | (stocks['代码'].str.startswith('60'))) & \
                (~stocks['名称'].str.startswith('PT')) & \
                (~stocks['名称'].str.startswith('ST')) & \
                (~stocks['名称'].str.startswith('*ST')) & \
                (~stocks['最新价'].isnull())
    part_stocks = stocks = stocks[condition].reset_index(drop=True)
    err_part_stocks = pd.DataFrame(columns=part_stocks.columns)
    stocks.to_csv(os.path.join(db_dir,f'0-{date}-行情.csv'),index=False)

    while part_stocks.shape[0]:
        for i,r in part_stocks.iterrows():
            try:
                code = r['代码']
                name = r['名称']

                filepath = os.path.join(db_dir,f'{code}-{date}-信息.csv')
                if not os.path.exists(filepath):
                    info = ak.stock_individual_info_em(code,10)
                    if info[info['item'] == '总市值'].iloc[0]['value'] == '-':
                        continue

                    if info[info['item'] == '股票代码'].iloc[0]['value'] != code:
                        raise Exception(code,info)
                    
                    bid = ak.stock_bid_ask_em(symbol=code)
                    info.loc[len(info.index)] = ['市盈率-动态',r['市盈率-动态']]
                    info.loc[len(info.index)] = ['昨收',bid[bid['item'] == '昨收'].iloc[0]['value']]
                    info.loc[len(info.index)] = ['今开',bid[bid['item'] == '今开'].iloc[0]['value']]
                    info.to_csv(filepath,index=False)

                filepath = os.path.join(db_dir,f'{code}-{date}-交易.csv')
                if not os.path.exists(filepath):
                    deals = ak.stock_intraday_em(symbol=code)
                    deals.to_csv(filepath,index=False)

                filepath = os.path.join(db_dir,f'{code}-{date}-人气.csv')
                if not os.path.exists(filepath):
                    prefix = 'SH' if code[0:2] == '60' else 'SZ' if code[0:2] == '00' else ''
                    ranks = ak.stock_hot_rank_detail_realtime_em(prefix + code)
                    ranks = ranks[50:].reset_index(drop=True)
                    ranks['时间'] = ranks['时间'].str[11:]
                    ranks.to_csv(filepath,index=False)

                print(i+1,'/',part_stocks.shape[0],code,name,flush=True)
            except (ConnectionError, ReadTimeout, ValueError, ConnectionResetError,requests.exceptions.ChunkedEncodingError,requests.exceptions.ConnectionError):
                err_part_stocks.loc[err_part_stocks.shape[0]] = r
            except (BrokenPipeError, KeyboardInterrupt):
                return
            except:
                tb.print_exc()
                return

        part_stocks = err_part_stocks
        err_part_stocks = pd.DataFrame(columns=part_stocks.columns)
        if len(part_stocks): print('错误集',part_stocks,len(part_stocks),file=sys.stderr)
    pass
    
if __name__ == '__main__':
    works = list()
    plays = dict()

    date = datetime.now().strftime('%y-%m-%d')
    days = 0
    cap = (0,0)
    pe = (0,0)

    while True:
        if len(sys.argv) > 1:
            cmd = sys.argv[1:]
            sys.argv = sys.argv[0:1]
        else:
            try:
                cmd = input('>\n').strip().split(' ')
            except (EOFError,KeyboardInterrupt):
                break

        parser = argparse.ArgumentParser()
        parser.add_argument('mode', type=str, help='The mode to run the program')
        parser.add_argument('--name',type=str,default='')
        parser.add_argument('--code',type=str,default='')
        parser.add_argument('--date',type=str,default=datetime.now().strftime('%y%m%d'))
        parser.add_argument('--days',type=int,default=0)
        
        args = parser.parse_args(cmd)
        if args.date: date = args.date
        if args.days: days = args.days
        
        if args.mode == 'sync':
            sync()
            break
        else:
            pass