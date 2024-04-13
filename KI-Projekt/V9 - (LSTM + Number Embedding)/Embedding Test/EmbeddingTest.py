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



#--------------------------Hyperparameters-------------------------
SEQ_LEN = 3
FUTURE_PERIOD_PREDICT = 30
BATCH_SIZE = 2

ENCODING_LENGTH = 5
TOKENIZING_LENGTH = 0


###------------------Additional Helper Functions------------------------------

def create_bin():
    amount = ENCODING_LENGTH
    size = 2/amount
    startvalue = -1
    bin_list  = [-1]
    for i in range(0,amount):
        startvalue = round((startvalue + size), 3)
        bin_list.append(startvalue)
    return bin_list

def classify(current, future):
        if float(future) > float(current):
            return 1
        else:
            return 0


#-------------------------Data Collection-------------------------------------
def get_data_file(filename):
    script_path = os.path.abspath(__file__)
    script_folder = os.path.dirname(script_path)
    print(script_folder)
    
    df = pd.read_csv(f"{script_folder}\{filename}")
    df = df.set_index("time")
    df = df.drop(columns=["open","high","low","volume"])
    
    
    
    #Füge das Trainingsziel hinzu

    df["future"] = df["close"].shift(-FUTURE_PERIOD_PREDICT)
    df["target"] = list(map(classify, df["close"], df["future"]))
    df = df.drop(columns=["future"])
    
    #Normalisierung der Werte
    df["ROC"] = df["close"].pct_change()
    
    min_value = df['ROC'].min()
    max_value = df['ROC'].max()

    df['ROC'] = 2 * (df['ROC'] - min_value) / (max_value - min_value) - 1
    
    #Werte Kategoriesiern
    
    bins = create_bin()
    print(bins)
    labels = [i for i in range(0, len(bins)-1)]
    df['OneHotEncoded'] = pd.cut(df['ROC'], bins=bins, labels=labels, include_lowest=True)
    
    #Umsortierung der Spalten
    new_order = ["OneHotEncoded", "close", "ROC", "target"]
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
            prev_days.append(np.array(v[0]))
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
    
    print(df.values[0])
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
            return torch.from_numpy(array), torch.tensor(target, dtype=torch.float32)

#-----------------Preprocessing Managment-----------------------
if True:
    df = get_data_file("TestFile.csv") #"TestFile.csv" ,"BTC-USD.csv"
    train_df, valid_df = divide_df(df)
    train_data = preprocessing(train_df)
    valid_data = preprocessing(valid_df)
    print(f"train_data: {len(train_data)} validation:{len(valid_data)} ")
    
    train_dataset = CustomDataset(train_data)
    valid_dataset = CustomDataset(valid_data)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    #valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

    x,y = next(iter(train_loader))
    print(x,y)


#index_tensor = torch.tensor(index_list)
#print(index_list)

#----------------Embedding Tests--------------------------

"""n_embeddings, dim = ENCODING_LENGTH,4

emb = Embedding(n_embeddings, dim)

print(emb(index_tensor))
"""

