import zipfile
import os
from datetime import date, timedelta



TIMEFRAME = "1m"
PAIRNAME = "ETHBTC"
TARGET_FOLDERNAME = "ETH-BTC_daily_zipped"
END_FOLDERNAME = "ETH-BTC_daily"



start_date = date(2021, 3, 1)
delta = timedelta(days=1)
date_list = []

while start_date <= date(2023, 4, 5):
    date_list.append(start_date.strftime("%Y-%m-%d"))
    start_date += delta
target_path = f"C:/Users/Lorian/Desktop/Trading_AI/Data/{TARGET_FOLDERNAME}"
end_path = f"C:/Users/Lorian/Desktop/Trading_AI/Data/{END_FOLDERNAME}/"

if not os.path.exists(end_path):
    os.makedirs(end_path)
if not os.path.exists(target_path):
    print("\nInvalid Target Folder\n")

# Öffnet das zip-File und extrahiert alle enthaltenen Dateien in den Zielordner
for date in date_list:
    with zipfile.ZipFile(f"C:/Users/Lorian/Desktop/Trading_AI/Data/{TARGET_FOLDERNAME}/{PAIRNAME}-{TIMEFRAME}-{date}.zip", 'r') as zip_ref:
        zip_ref.extractall(f"C:/Users/Lorian/Desktop/Trading_AI/Data/{END_FOLDERNAME}/")
