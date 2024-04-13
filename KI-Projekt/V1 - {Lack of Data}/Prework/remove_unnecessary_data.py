import os
import pandas as pd

def file_to_df(file_name):
    df = pd.read_csv(file_name)
    #df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.set_index("time")
    df = df.drop(columns=["open", "high","low","closetime", "qav","num of trades", "tbbav", "tbqav", "ignore"], axis=1)
    return df

file_name = "C:/Users/Lorian/Desktop/Trading_AI/Prework/Full_BTC_Data.csv"
df = file_to_df(file_name)

output_path = "C:/Users/Lorian/Desktop/Trading_AI/BTC_Data.csv"
df.to_csv(output_path)

print("Dataframe wurde erfolgreich als CSV-Datei gespeichert!")