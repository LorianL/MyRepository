import os
import pandas as pd

def create_datelist():
    start_date = date(2021, 3, 1)

    delta = timedelta(days=1)

    date_list = []

    while start_date <= date(2023, 4, 5):
        date_list.append(start_date.strftime("%Y-%m-%d"))
        start_date += delta
    return date_list

def file_to_df(file_name):
    df = pd.read_csv(f"C:/Users/Lorian/Desktop/Trading_AI/Data_BTC/{file_name}", names=["time", "open", "high","low", "close", "volume", "closetime", "qav","num of trades", "tbbav", "tbqav", "ignore"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.set_index("time")
    return df

main_df = pd.DataFrame(columns = ["time", "open", "high","low", "close", "volume", "closetime", "qav","num of trades", "tbbav", "tbqav", "ignore"])
main_df.set_index("time",inplace=True)

counter = 0
files_list = os.listdir("C:/Users/Lorian/Desktop/Trading_AI/Data_BTC")
for file in files_list:
    if (counter % 20) == 0:
        print(counter,len(files_list))
    if file.endswith(".csv"):
        #main_df = main_df.append(pd.read_csv(f"C:/Users/Lorian/Desktop/Trading_AI/Data_BTC/{file}"))
        df = file_to_df(file)
        main_df = pd.concat([main_df, df], ignore_index=False)
    counter += 1


main_df.to_csv("Full_BTC_Data.csv")
print(main_df)
