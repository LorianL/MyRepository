import os
import wget
from datetime import date, timedelta

TIMEFRAME = "1m"
PAIRNAME = "ETHBTC"
FOLDERNAME = "ETH-BTC_daily"



path = f"c:/Users/Lorian/Desktop/Trading_AI/Data/{FOLDERNAME}_zipped/"


if not os.path.exists(path):
    os.makedirs(path)

start_date = date(2021, 3, 1)
delta = timedelta(days=1)
date_list = []

while start_date <= date(2023, 4, 5):
    date_list.append(start_date.strftime("%Y-%m-%d"))
    start_date += delta

for i,date in enumerate(date_list):
    print(f"\n {int(i/len(date_list)*100)}% \n")
    wget.download(f"https://data.binance.vision/data/spot/daily/klines/{PAIRNAME}/{TIMEFRAME}/{PAIRNAME}-{TIMEFRAME}-{date}.zip", out=path)
