import os
import random

import pandas as pd
import numpy as np
from collections import deque

import torch
import torch.nn as nn
from torch.nn import Embedding
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

import matplotlib.pyplot as plt
import copy

print()
if torch.cuda.is_available():
    print("GPU available")
    device = torch.device("cuda")
else:
    print("GPU NOT available")
    device = torch.device("cpu")

script_path = os.path.abspath(__file__)
script_folder = os.path.dirname(script_path)
print(script_folder)
#--------------------------Hyperparameters-------------------------

SEQ_LEN = 120
FUTURE_PERIOD_PREDICT = 30
BATCH_SIZE = 512
NUM_EPOCHS = 50

ENCODING_LENGTH = 20
TOKENIZING_LENGTH = 40

###------------------Additional Helper Functions------------------------------

def categorize(df, name:str):
    
    print(df.isna().sum())
    
    bin_amount = ENCODING_LENGTH
    #Werte sortieren
    value_list = df[name].tolist()
    value_list.sort()
    #Bins erstellen, in jedem bin gleichviele Werte
    bin_list  = [-float("inf")]
    
    for i in range(0,bin_amount-1):
        val = value_list[int(len(value_list) * i * 1/bin_amount)]
        bin_list.append(val)

    bin_list.append(float("inf"))
    print(bin_list)
    labels = [i for i in range(0, len(bin_list)-1)]

    df[name] = pd.cut(df[name], bins=bin_list, labels=labels, include_lowest=True)

    return df

def classify(current, future):
        if float(future) > float(current):
            return 1
        else:
            return 0

def min_max_normalize(df, name: str):
        min_value = df[name].min()
        max_value = df[name].max()
        
        df[name] = (df[name] - min_value) / (max_value - min_value)
        
        return df

def normalize(df, name: str):
    # Berechne den 1000-Tage-SMA
    sma_1000 = df[name].rolling(window=1000).mean()

    # Berechne die Differenz zwischen dem Wert und dem 1000-Tage-SMA
    df[name] = df[name] - sma_1000

    return df

#-------------------------Data Collection-------------------------------------
def get_data_file(filename):

    df = pd.read_csv(f"{script_folder}\{filename}")
    df = df.set_index("time")
    df = df.drop(columns=["open","high","low","volume"])
    
    
    
    #Füge das Trainingsziel hinzu

    df["future"] = df["close"].shift(-FUTURE_PERIOD_PREDICT)
    df["target"] = list(map(classify, df["close"], df["future"]))
    df = df.drop(columns=["future"])
    
    #Rate of Change erstellen
    
    df["ROC"] = df["close"].pct_change()
    df = normalize(df, "close")
    #Bei der erstellung des ROC entsteht ein NaN wert
    df = df.dropna()
    df = min_max_normalize(df, "ROC")
    df = min_max_normalize(df, "close")
    
    

    
    
    #Werte Kategoriesiern
    df = categorize(df, "ROC")
    df = categorize(df, "close")
    
    #Umsortierung der Spalten
    new_order = ["close", "ROC", "target"]
    df = df[new_order]
    
    #Entferne alle NaN Spalten
    df = df.dropna()
    #df = df[:20]
    print(df)
    
    return df

#----------------Aufteilung in train/valid Sets-------------------
def divide_df(df):
    times = sorted(df.index.values)
    last_5pct = times[- int(0.05*len(times))]

    train_df = df[(df.index < last_5pct)]
    valid_df = df[(df.index >= last_5pct)]
    
    return train_df, valid_df
#--------------Sequentialsierung und Balacierung der Daten---------------------
def preprocessing(df):
    #Unterteil den Trainingsdaten satz in Trainingsfenster(sequenzen)
    def sequentialise(df):
        sequential_data = []
        prev_days = deque(maxlen=SEQ_LEN)

        for v in df.values:
            prev_days.append(np.array(v[:-1]))
            if len(prev_days) == SEQ_LEN:
                sequential_data.append([np.array(prev_days, dtype=np.float32),v[-1]])
        random.shuffle(sequential_data)
        return sequential_data
    
    #Balanziert die Daten so, dass genau soviele Positive als auch negative Beispiele vorhanden sind
    def balancing(sequential_data):
        buys = []
        sells = []

        for seq, target in sequential_data:
            if target == 0:
                sells.append([seq,target])
            elif target == 1:
                buys.append([seq,target])
        random.shuffle(buys)
        random.shuffle(sells)

        lower = min(len(buys),len(sells))

        buys = buys[:lower]
        sells = sells[:lower]
        
        if len(buys) != len(buys):
            print("Achtung, nicht die gleiche Länge:",len(buys),len(buys))

        sequential_data = buys + sells

        random.shuffle(sequential_data)
        return sequential_data
    
    #print(df.values[0])
    seq_data = sequentialise(df)
    seq_data = balancing(seq_data)
    return seq_data

#-----------------Dataset Class----------------------
class CustomDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            array, target = self.data[index]
            data_float_tensor = torch.from_numpy(array)
            data_int_tensor = data_float_tensor.to(torch.int)
            return data_int_tensor, torch.tensor(target, dtype=torch.float32)

#-----------------Preprocessing Managment-----------------------
if True:
    df = get_data_file("BTC-USD.csv") #"TestFile.csv" ,"BTC-USD.csv"
    train_df, valid_df = divide_df(df)
    train_data = preprocessing(train_df)
    valid_data = preprocessing(valid_df)
    print(f"train_data: {len(train_data)} validation:{len(valid_data)} ")
    
    train_dataset = CustomDataset(train_data)
    valid_dataset = CustomDataset(valid_data)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

    x,y = next(iter(train_loader))
    print(x.shape, y.shape)
    print(x)
    