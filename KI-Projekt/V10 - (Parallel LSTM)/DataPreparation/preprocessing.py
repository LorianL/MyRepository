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

def plot_list(list):
    bin_width = 0.0001
    num_bins = int(0.02 / bin_width)
        
    counts, bins, _ = plt.hist(list, bins=np.linspace(-0.01, 0.01, num=num_bins + 1), rwidth=0.8)
        
    plt.bar(bins[:-1], counts, width=bin_width, align='edge', edgecolor='black')
    plt.ylabel('Anzahl der Elemente')
        
    plt.show()



SEQ_LEN = 4
FUTURE_PERIOD_PREDICT = 30
BATCH_SIZE = 1
NUM_EPOCHS = 10

ENCODING_LENGTH = 100
TOKENIZING_LENGTH = 10

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
    def normalise(df, key_name):
        df[key_name] = df[key_name].round(decimals=3)
        #min max normalisieren
        min_value = df[key_name].min()
        max_value = df[key_name].max()

        df[key_name] = 2 * (df[key_name] - min_value) / (max_value - min_value) - 1
        
        #Werte Kategoriesiern
    
        bins = create_bin()

        labels = [i for i in range(0, len(bins)-1)]
        df[key_name] = pd.cut(df[key_name], bins=bins, labels=labels, include_lowest=True)
        
        return df
    
    
    df = pd.read_csv(f"{script_folder}\{filename}")
    df = df.set_index("time")
    df = df.drop(columns=["open","high","low"])
    
    
    
    #Füge das Trainingsziel hinzu

    df["future"] = df["close"].shift(-FUTURE_PERIOD_PREDICT)
    df["target"] = list(map(classify, df["close"], df["future"]))
    df = df.drop(columns=["future"])
    
    #RateOfChange berechnen
    
    df["ROC"] = df["close"].pct_change()
    roc_list = df["ROC"].tolist()
    roc_list.sort()
    #print(roc_list)
    #print(roc_list[int(len(roc_list)*0.02)],roc_list[int(len(roc_list)*0.99)])
    min_cut_value = roc_list[int(len(roc_list)*0.05)]
    max_cut_value = roc_list[int(len(roc_list)*0.95)]
    
    
    df.loc[df['ROC'] < min_cut_value, 'ROC'] = min_cut_value
    df.loc[df['ROC'] > max_cut_value, 'ROC'] = max_cut_value
    
    #Volume daten abschneiden
    
    volume_list = df["volume"].tolist()
    volume_list.sort()
    
    cut_value = volume_list[int(len(volume_list)*0.95)]
    
    df.loc[df['volume'] > cut_value, 'volume'] = cut_value
    
    #Werte Normalisieren und OneHotEncoden
    df = normalise(df, "ROC")
    df = normalise(df, "volume")
    df = df.drop(columns=["close"])
    
    #Umsortierung der Spalten
    new_order = ["ROC","volume", "target"]
    df = df[new_order]
    
    #Entferne alle NaN Spalten
    df = df.dropna()
    #df = df[:20]
    #print(df)
    
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
            data_float_tensor = torch.from_numpy(array)
            data_int_tensor = data_float_tensor.to(torch.int)
            return data_int_tensor, torch.tensor(target, dtype=torch.float32)
def initialize_weights(layer):

    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        # Optional: Glorot/Xavier-Initialisierung für die Bias-Terme
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
def clip_weights(model, clip_value=0.5):
    for param in model.parameters():
        #if param.ndim > 1:  # Nur Gewichte (keine Biases)
        param.data.clamp_(-clip_value, clip_value)

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
    print(x.shape,y)


"""
emb_close = Embedding(ENCODING_LENGTH, TOKENIZING_LENGTH)
emb_volume = Embedding(ENCODING_LENGTH, TOKENIZING_LENGTH)


close = emb_close(x[:,:,0])
volume = emb_volume(x[:,:,1])

print(close.shape)
print(volume.shape)

combined = torch.cat((close, volume), dim=2)

print(combined.shape)"""