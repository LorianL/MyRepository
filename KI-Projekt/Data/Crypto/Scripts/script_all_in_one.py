import os
import wget
from datetime import date, timedelta
import time
import zipfile
import shutil
import pandas as pd


TIMEFRAME = "1m"
PAIRNAME = "DOGEUSDT"
FILENAME = "DOGE-USD"

#TARGET_FOLDERNAME = "ETH-BTC_zipped"
#END_FOLDERNAME = "ETH-BTC"

path_download = f"c:/Users/Lorian/Desktop/Trading_AI/Data/{FILENAME}_zipped/"


if not os.path.exists(path_download):
    os.makedirs(path_download)

start_date = date(2021, 3, 1)
delta = timedelta(days=1)
date_list = []

while start_date <= date(2023, 4, 5):
    date_list.append(start_date.strftime("%Y-%m-%d"))
    start_date += delta

for i,date in enumerate(date_list):
    print(f"\n {round(i/len(date_list)*100,1)}%")
    wget.download(f"https://data.binance.vision/data/spot/daily/klines/{PAIRNAME}/{TIMEFRAME}/{PAIRNAME}-{TIMEFRAME}-{date}.zip", out=path_download)
   
print(f"\n\nDownload Finished \n")
#------------------ 
time.sleep(5)
#------------------


path_extract = f"C:/Users/Lorian/Desktop/Trading_AI/Data/{FILENAME}_data/"

if not os.path.exists(path_extract):
    os.makedirs(path_extract)
if not os.path.exists(path_download):
    print("\nERROR download Folder not found\n")


paths_list = os.listdir(path_download)
# Öffnet das zip-File und extrahiert alle enthaltenen Dateien in den Zielordner
for path in paths_list:
    if path.endswith('.zip'):
        with zipfile.ZipFile(f"C:/Users/Lorian/Desktop/Trading_AI/Data/{FILENAME}_zipped/{path}", 'r') as zip_ref:
            zip_ref.extractall(f"C:/Users/Lorian/Desktop/Trading_AI/Data/{FILENAME}_data/")

print("Extraction Finished\n")
#------------------
time.sleep(5)
#------------------

shutil.rmtree(path_download)


main_df = pd.DataFrame(columns = ["time", "open", "high","low", "close", "volume", "closetime", "qav","num of trades", "tbbav", "tbqav", "ignore"])
main_df.set_index("time",inplace=True)


def file_to_df(file_name):
    df = pd.read_csv(f"{path_extract}/{file_name}", names=["time", "open", "high","low", "close", "volume", "closetime", "qav","num of trades", "tbbav", "tbqav", "ignore"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.set_index("time")
    return df

files_list = os.listdir(path_extract)
for file in files_list:
    
    if file.endswith(".csv"):
        df = file_to_df(file)
        main_df = pd.concat([main_df, df], ignore_index=False)

main_df = main_df.drop(columns=["open", "high","low","closetime", "qav","num of trades", "tbbav", "tbqav", "ignore"], axis=1)
main_df.to_csv(f"C:/Users/Lorian/Desktop/Trading_AI/Data/{FILENAME}.csv")
print("File successfully saved")