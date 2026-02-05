from datetime import datetime, timedelta

now = datetime.now()
h9 = datetime.strptime(f'{now.date()} 09:30:00','%Y-%m-%d %H:%M:%S')
h11 = datetime.strptime(f'{now.date()} 11:30:00','%Y-%m-%d %H:%M:%S')
h13 = datetime.strptime(f'{now.date()} 13:00:00','%Y-%m-%d %H:%M:%S')
h15 = datetime.strptime(f'{now.date()} 15:00:00','%Y-%m-%d %H:%M:%S')

a = (h11 - h9).total_seconds() + (h15 - h13).total_seconds()


print(a)
