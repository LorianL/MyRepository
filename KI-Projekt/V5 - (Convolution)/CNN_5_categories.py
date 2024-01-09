#from torchsummary import summary
import pandas as pd
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
#from collections import deque
import numpy as np
import random
#import time
#import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision
#import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

print()
if torch.cuda.is_available():
    print("GPU available")
    device = torch.device("cuda")
else:
    print("GPU NOT available")
    device = torch.device("cpu")




FUTURE_PERIOD_PREDICT = 120
EPOCHS = 10
BATCH_SIZE = 256
NUM_OF_PERIODS = 10
NUM_OF_CATEGORIES = 5
learning_rate = 0.01
num_epoch = 10

global classes_categories
classes_categories = []

def get_data():
 #-------------------------------------------------------   
    def none(df):
        return df

    def z_score_normalize(df_series):
        # Berechne den Mittelwert
        mean = df_series.mean()
        
        # Berechne die Standardabweichung
        std_dev = df_series.std()
        
        # Wende die Z-Score-Normalisierung auf die Daten an
        normalized_data = (df_series - mean) / std_dev
        
        return pd.Series(normalized_data)

#------------------------------------------------

#none        
    def add_rsi(df, period=14, norm_fn = z_score_normalize):
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 1 - (1 / (1 + rs))
        rsi = norm_fn(rsi)
        #rsi = min_max_normalize(rsi)
        df[f'rsi_{period}'] = rsi
        return df
#winsorize
    def add_atr(df, period=14, norm_fn = z_score_normalize):
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        atr = norm_fn(atr)
        #atr = min_max_normalize(atr)
        df[f'atr_{period}'] = atr
        return df
#none
    def add_mfi(df, period=14, norm_fn = z_score_normalize):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        positive_flow = raw_money_flow.where(df['close'] > df['close'].shift(), 0)
        negative_flow = raw_money_flow.where(df['close'] < df['close'].shift(), 0)
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        money_ratio = positive_mf / negative_mf
        mfi = 1 - (1 / (1 + money_ratio))
        mfi = norm_fn(mfi)
        #mfi = min_max_normalize(mfi)
        df[f'mfi_{period}'] = mfi
        return df
#winsorize
    def add_cmf(df, period=20, norm_fn = z_score_normalize):
        mfm = ((2 * df['close'] - df['low'] - df['high']) / (df['high'] - df['low'])) * df['volume']
        cmf = mfm.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        cmf = norm_fn(cmf)
        #cmf = min_max_normalize(cmf)
        df[f"cmf_{period}"] = cmf
        return df
#min_max_normalize
    def add_roc(df, period=12, norm_fn = z_score_normalize):
        roc = (df['close'] - df['close'].shift(periods=period)) / df['close'].shift(periods=period)
        roc = norm_fn(roc)
        #roc = min_max_normalize(roc)
        df[f'roc_{period}'] = roc
        return df
#none
    def add_vo(df, short_period=14, long_period=28, norm_fn = z_score_normalize):
        short_ma = df['volume'].rolling(window=short_period).mean()
        long_ma = df['volume'].rolling(window=long_period).mean()
        vo = (short_ma - long_ma) / long_ma
        vo = norm_fn(vo)
        #vo = min_max_normalize(vo)
        df[f'vo_{short_period}_{long_period}'] = vo
        return df
#z_score_normalize
    def add_cci(df, period=20, norm_fn = z_score_normalize):
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = np.abs(tp - sma).rolling(window=period).mean()
        cci = (tp - sma) / (0.015 * mad)
        cci = norm_fn(cci)
        #cci = min_max_normalize(cci)
        df[f'cci_{period}'] = cci
        return df
#none
    def add_cmo(df, period=14, norm_fn = z_score_normalize):
        diff = df['close'].diff()
        pos_sum = diff.where(diff > 0, 0).rolling(window=period).sum()
        neg_sum = np.abs(diff.where(diff < 0, 0)).rolling(window=period).sum()
        cmo = ((pos_sum - neg_sum) / (pos_sum + neg_sum))
        cmo = norm_fn(cmo)
        #cmo = min_max_normalize(cmo)
        df[f'cmo_{period}'] = cmo
        return df
#Sma + winsorize
    def add_eom(df, period=14, norm_fn = z_score_normalize):
        high = df['high']
        low = df['low']
        vol = df['volume']
        mid = (high + low) / 2
        box_ratio = (high - low) / vol
        one_period_eom = (mid.diff(1) / box_ratio)
        eom = one_period_eom.rolling(window=period).mean()
        #eom = sma_normalize(eom)
        eom = norm_fn(eom)
        #eom = min_max_normalize(eom)
        df[f'eom_{period}'] = eom
        return df
#min_max_normalize
    def add_bb_bandwidth(df, window=20, std=2 , norm_fn = z_score_normalize):
        df['bb_upper'] = df['close'].rolling(window=window).mean() + std * df['close'].rolling(window=window).std()
        df['bb_lower'] = df['close'].rolling(window=window).mean() - std * df['close'].rolling(window=window).std()
        bb_bandwidth = (df['bb_upper'] - df['bb_lower']) / df['close'].rolling(window=window).mean()
        bb_bandwidth = norm_fn(bb_bandwidth)
        #bb_bandwidth = min_max_normalize(bb_bandwidth)
        df[f'bb_bandwidth_{window}'] = bb_bandwidth
        df.drop(['bb_upper', 'bb_lower'], axis=1, inplace=True)
        return df
#sma + z_score_normalize
    def add_ema(df, span=20, norm_fn = z_score_normalize):
        ema = df['close'].ewm(span=span).mean()
        #ema = sma_normalize(ema)
        ema = norm_fn(ema)
        #ema = min_max_normalize(ema)
        col_name = f'ema_{span}'
        df[col_name] = ema
        
        return df

    def add_target(df, lim_list):
        def classify_diff(difference):
            for i, (lower_lim,higher_lim) in enumerate(lim_list):
                if lower_lim <= difference and difference <= higher_lim:
                    return i
            print("No matching categorie")
        future_df = pd.DataFrame()
        future_df["future"] = df["close"].shift(-FUTURE_PERIOD_PREDICT)
        df["target"] = list(map(classify_diff, df["diff"]))
        return df
        
    def add_diff(df):
        def difference_target(current, future):
            return (future / current) - 1
        temp_df = pd.DataFrame()
        temp_df["future"] = df["close"].shift(-FUTURE_PERIOD_PREDICT)
        df["diff"] = list(map(difference_target, df["close"], temp_df["future"]))
        return df

    df = pd.read_csv("C:/Users/Lorian/Desktop/Trading_AI/V5 - (Convolution)/BTC-USD.csv")
    df = df.set_index("time")

    indicator_fn = [add_rsi,add_atr,add_cmf,add_roc,add_cci,add_cmo,add_bb_bandwidth,add_mfi,add_ema] #add_eom ,add_vo


    for ind_fn in indicator_fn:
        for i in range(1,NUM_OF_PERIODS+1):
            df = ind_fn(df,i*7)
        df = df.copy()
    
    for i in range(1,NUM_OF_PERIODS+1):
        df = add_vo(df, i*7, (i+1)*7)

    
    df = add_diff(df)
    df = df.dropna()

    #TARGET CALCULATION
    data = df['diff'].tolist()
    lim_list = []
    data.sort()
    for k in range(0,NUM_OF_CATEGORIES):
        lower_lim = data[int((1/NUM_OF_CATEGORIES)*k*len(data))]
        higher_lim = data[int((1/NUM_OF_CATEGORIES)*(k+1)*len(data)-1)]
        lim_list.append((lower_lim,higher_lim))
        classes_categories.append((lower_lim,higher_lim))
        print(f"Kategorie {k+1}: {(lower_lim*100):.4f} - {(higher_lim*100):.4f}")
    
    df = add_target(df,lim_list)

    df = df.dropna()

    df = df.drop(columns=["close","volume","open","high","low"])
    #print(df)
    #print(df.size)

    return df

def divide_df(df):
    times = sorted(df.index.values)
    last_5pct = times[- int(0.05*len(times))]

    train_df = df[(df.index < last_5pct)]
    valid_df = df[(df.index >= last_5pct)]
    
    return train_df, valid_df

def formate(df):
        indicators = list(df.columns)
        matrix_data = []

        for v in df.values:
            list_matrix = []
            for i in range(0,int((len(indicators)-1)/NUM_OF_PERIODS)):
                list_matrix.append(v[i*NUM_OF_PERIODS:(i+1)*NUM_OF_PERIODS])
            matrix = np.matrix(list_matrix, dtype= np.float32)
            matrix = np.expand_dims(matrix, axis=0)
            matrix_data.append([matrix,v[-1]])
        random.shuffle(matrix_data)
        return matrix_data

df = get_data()
train_df,test_df = divide_df(df)
train_data = formate(train_df)
test_data = formate(test_df)


print(f"train_data: {len(train_data)} validation:{len(test_data)} ")

#print(train_data[-100])



class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        matrix, target = self.data[index]
        return torch.from_numpy(matrix), torch.tensor(target, dtype=torch.float32)



train_dataset = CustomDataset(train_data)
test_dataset = CustomDataset(test_data)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(next(iter(train_dataset))[0].shape)

print(3*"\n")




#implement convolutional Net
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 2)
        self.fc1 = nn.Linear(16*7*7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,NUM_OF_CATEGORIES)
    
    def forward(self,x):
        #First Conv Layer
        x = self.conv1(x)
        x = F.relu(x)
        #x = self.pool(x)
        #Second Conv Layer        
        x = self.conv2(x)
        x = F.relu(x)
        #print(x.shape)
        #Flatten the Tensor
        x = x.view(-1, 16*7*7)
        #First Fully Connected Layer
        x = self.fc1(x)
        x = F.relu(x)
        #Second FC Layer
        x = self.fc2(x)
        x = F.relu(x)
        #Last Layer
        x = self.fc3(x)
        #No activation Function
        return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def print_information(model):
    # Zähle die Gesamtzahl der Parameter im Modell
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Gesamtzahl der Parameter: {total_params}")

    """# Iteriere über die einzelnen Schichten des Modells und gib die Größe der Parameter aus
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Schichtname: {name}, Größe der Parameter: {param.size()}")"""


def train(epoch):
    model.train()
    
    n_total_steps = len(train_loader)
    total_loss = 0
    
    for i, (images, labels) in enumerate(train_loader):
        #origin shape: [4,3,32,32] = 4,3,1024
        #input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        #labels = labels.view(-1,1)
        labels = labels.long()
        #labels = labels.squeeze()
        labels = labels.to(device)
        
        #Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        #Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 1000 == 0:
            print(f"Epoch[{epoch+1}/{num_epoch}], Step [{i+1}/{n_total_steps}], Loss: {(total_loss/i):.4f}")

def test():
    model.eval()
    
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(10)]
        n_class_samples = [1 for i in range(10)]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.long()
            labels = labels.to(device)
            outputs = model(images)
            
            #max returns (value, index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            
            for i in range(len(labels)):
                #print(labels)
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[pred] += 1
                n_class_samples[pred] += 1
        
        acc = 100 * n_correct / n_samples
        print(f"Accuracy of the network: {acc:.4f}%")
        
        for i in range(len(classes_categories)):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f"Accuracy of <<{classes_categories[i][0]*100:.3f}>> - <<{classes_categories[i][1]*100:.3f}>>: {acc:.4f}%\t ({n_class_correct[i]} / {n_class_samples[i]})")
        
    

test()
if __name__ == "__main__":
    print_information(model)
    for epoch in range(num_epoch):
        print(20*"-")
        train(epoch)
        test()
        print(20*"-")
