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




FUTURE_PERIOD_PREDICT = 30
EPOCHS = 10
BATCH_SIZE = 512
NUM_OF_PERIODS = 10
LEARNING_RATE = 0.01
weight_decay_value = 0.001
#NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"



def get_data(file_name):
 #-------------------------------------------------------   

    def none(df_series):
        return df_series

    def z_score_normalize(df_series):
        # Berechne den Mittelwert
        mean = df_series.mean()
        
        # Berechne die Standardabweichung
        std_dev = df_series.std()
        
        # Wende die Z-Score-Normalisierung auf die Daten an
        normalized_data = (df_series - mean) / std_dev
        
        return pd.Series(normalized_data)

    def classify(current, future):
        if float(future) > float(current):
            return 1
        else:
            return 0
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

    def add_target(df):
        future_df = pd.DataFrame()
        future_df["future"] = df["close"].shift(-FUTURE_PERIOD_PREDICT)
        df["target"] = list(map(classify, df["close"], future_df["future"]))
        return df

    def add_diff(df):
        def difference_target(current, future):
            return (future / current) - 1
        temp_df = pd.DataFrame()
        temp_df["future"] = df["close"].shift(-FUTURE_PERIOD_PREDICT)
        df["diff"] = list(map(difference_target, df["close"], temp_df["future"]))
        return df

    path = f"C:/Users/Lorian/Desktop/Trading_AI/V5 - (Convolution)/{file_name}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        print(f"Invalid pairname: {file_name}")
    df = df.set_index("time")

    indicator_fn = [add_rsi,add_atr,add_cmf,add_roc,add_cci,add_cmo,add_bb_bandwidth,add_mfi,add_ema] #add_eom ,add_vo

    for ind_fn in indicator_fn:
        for i in range(1,NUM_OF_PERIODS+1):
            df = ind_fn(df,i*7)
        df = df.copy() 
    for i in range(1,NUM_OF_PERIODS+1):
        df = add_vo(df, i*7, (i+1)*7)

    shuffled_columns = np.random.permutation(df.columns)
    df = df[shuffled_columns]
    
    df = add_diff(df)
    df = add_target(df)
    
    df = df.dropna()

    df = df.drop(columns=["close","volume","open","high","low"])
    
    print(df)
    return df

def divide_df(df):
    times = sorted(df.index.values)
    last_5pct = times[- int(0.05*len(times))]

    train_df = df[(df.index < last_5pct)]
    valid_df = df[(df.index >= last_5pct)]
    
    return train_df, valid_df

def preprocessing(df):
    
    def formate(df):
        indicators = list(df.columns)
        matrix_data = []

        for v in df.values:
            inds = v[:-2]
            target = random.randint(0,1) 
            difference = v[-2]
            list_matrix = []
            for i in range(0,int((len(indicators)-1)/NUM_OF_PERIODS)):
                list_matrix.append(inds[i*NUM_OF_PERIODS:(i+1)*NUM_OF_PERIODS])
            #wandele die Liste in ein Numpy Array
            matrix = np.array(list_matrix, dtype= np.float32)
            # Füge ein einzelnen Farbchannel hinzu
            matrix = np.expand_dims(matrix, axis=0)
            
            matrix_data.append([matrix,(np.array(target, dtype=np.float32), difference)])
        return matrix_data

    def balancing(data):
        buys = []
        sells = []

        for seq, (target,diff) in data:
            if target == 0:
                sells.append([seq,(target,diff)])
            elif target == 1:
                buys.append([seq,(target,diff)])
        random.shuffle(buys)
        random.shuffle(sells)

        lower = min(len(buys),len(sells))

        buys = buys[:lower]
        sells = sells[:lower]

        data = buys + sells

        random.shuffle(data)
        return data
    
    
    data = formate(df)

    data = balancing(data)
    
    return data


def collect_data():
    main_pair_name = "BTC-USD"
    other_pair_names = [] #"ETH-USD", "BCH-USD", "BNB-USD", "ADA-USD"
    #Get Data of main_pair_name and isolate Validation Data
    main_df = get_data(main_pair_name)
    train_df,valid_df = divide_df(main_df)
    
    train_data: list = preprocessing(train_df)
    valid_data: list = preprocessing(valid_df)
    for pair in other_pair_names:
        next_df = get_data(pair)
        next_train_data = preprocessing(next_df)
        train_data.extend(next_train_data)
    
    random.shuffle(train_data)
    random.shuffle(valid_data)
    
    return train_data, valid_data


train_data, valid_data = collect_data()
print(f"train_data: {len(train_data)} validation:{len(valid_data)} ")
#print(train_data[-1])
#print(torch.from_numpy(train_data[-1][1]))


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        matrix, target = self.data[index]
        tensor_matrix = torch.from_numpy(matrix)
        tensor_target = torch.from_numpy(target)
        return tensor_matrix, tensor_target
    
train_dataset = CustomDataset(train_data)
valid_dataset = CustomDataset(valid_data)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)

#image, target = next(iter(train_loader))

#print(image, target)



#implement convolutional Net
class ConvNet_V1(nn.Module):
    def __init__(self):
        super(ConvNet_V1, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        #self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        #self.fc1_output_size = 200
        self.fc1 = nn.Linear(576, 260)
        self.fc2 = nn.Linear(260, 120)
        self.fc3 = nn.Linear(120,40)
        self.fc4 = nn.Linear(40,1)
    
    def forward(self,x):
        #First Conv Layer
        x = self.conv1(x)
        x = F.relu(x)
        #Second Conv Layer        
        x = self.conv2(x)
        x = F.relu(x)
        #Berechne den Output des ConvLayers
        conv_out = torch.prod(torch.tensor(x.shape[1:])).item()
        #print(conv_out)
        #Flatten the Tensor
        x = x.view(-1, conv_out)
        #First Fully Connected Layer
        x = self.fc1(x)
        x = F.relu(x)
        #Second FC Layer
        x = self.fc2(x)
        x = F.relu(x)
        #3. Layer
        x = self.fc3(x)
        x = F.relu(x)
        #5. Layer
        x = self.fc4(x)
        x = F.sigmoid(x)
        return x

model = ConvNet_V1().to(device)

# criterion
criterion = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay_value)
#optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay_value)
#learning rate scheduler
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.33, patience=20, verbose=True)


#decay Scheduler
def weight_decay(loss_diff):
    global weight_decay_value
    threshold = 0.01
    factor = 2
    if loss_diff < (-1*threshold):
        weight_decay_value = weight_decay_value * factor  # Verdopple das Weight Decay
        
        # Aktualisiere den Optimizer mit dem neuen Weight Decay
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = weight_decay_value
        print(f"New increased Decay-Value: {weight_decay_value:.7f}, Difference was at {loss_diff:.4f}\n")
    elif loss_diff > threshold:
        weight_decay_value = weight_decay_value * (1/factor)  # Verdopple das Weight Decay
        
        # Aktualisiere den Optimizer mit dem neuen Weight Decay
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = weight_decay_value
        print(f"New reduced Decay-Value: {weight_decay_value:.7f}, Difference was at {loss_diff:.4f}\n")




def train(epoch):
    n_total_steps = len(train_loader)
    total_loss = 0.0
    for i, (images, target) in enumerate(train_loader):
        #origin shape [batch_size,1,10,10]
        #target hat folgende Form: [labels, difference]
        labels = target[0]
        labels = labels.view(-1,1)
        images = images.to(device)
        labels = labels.to(device)
        
        #Forward pass
        outputs = model(images)
        #print(outputs.shape)
        loss = criterion(outputs, labels)
        
        #Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        total_loss += loss.item()
        if (i+1) % 1000 == 0:
            print(f"Epoch[{epoch+1}/{EPOCHS}], Step [{i+1}/{n_total_steps}], Loss: {total_loss / (i+1):.4f}")
    avg_loss = total_loss / n_total_steps
    return avg_loss
            
def valid(epoch):
    with torch.no_grad():
        n_correct = 0
        n_samples = len(valid_dataset)
        n_total_steps = len(valid_loader)
        total_loss = 0.0
        for images, target in valid_loader:
            labels = target[0]
            labels = labels.view(-1,1)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = torch.round(outputs)
            n_correct += (predicted == labels).sum().item()
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
        acc = 100 * n_correct / n_samples
        avg_loss = total_loss / n_total_steps
        print(f"\n\nAccuracy in Epoch {epoch+1}: {acc:.4f}%     Loss: {avg_loss:.4f}\n\n")
        return acc, avg_loss

def performance(startcapital=50):
    
    with torch.no_grad():
        sum_profit = 1.0
        mul_profit = 1.0
        n_correct = 0.0
        n_samples = len(valid_dataset)
        count_True = 0
        count_False = 0
        for images, target in valid_loader:
            #Get necessary data
            labels = target[0]
            differences = target[1]
            #Get Outputs
            labels = labels.view(-1,1)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = torch.round(outputs)
            n_correct += (predicted == labels).sum().item()
            #print("---------------------")
            #print(labels.view(-1,BATCH_SIZE))
            #print(predicted.view(-1,BATCH_SIZE))
            #print((predicted == labels).view(-1,BATCH_SIZE))
            corrects = (predicted == labels)
            corrects = corrects.view(-1)
            corrects = corrects.tolist()
            #print(corrects)

            for i,Bool in enumerate(corrects):
                if Bool:
                    profit = abs(differences[i].item())
                    sum_profit = sum_profit + profit
                    mul_profit = mul_profit*(1+profit)
                    count_True += 1
                else:
                    loss = abs(differences[i].item())
                    sum_profit = sum_profit - loss
                    mul_profit = mul_profit*(1-loss)
                    count_False += 1
            
        acc = 100 * n_correct / n_samples
        print(50*"-")
        print(f"Profitability Analysis with a startcapital of {startcapital}$:")
        print(50*"-")
        print(f"\nAccuracy of the model: {acc}%\n")
        print(f"Profitability: {(mul_profit*100-100):.2f}%")
        print(f"Average Profit per Trade: {((sum_profit*100)/n_samples):.4f}%   Sum of all Profits: {(sum_profit*100):.4f}%")
        print(f"Total Profit: {(startcapital*mul_profit-startcapital):.2f}$")
        print(f"Final Capital: {(startcapital*mul_profit):.2f}\n")
            
            
        print(f"Winning Trades: {count_True}, Losing Trades: {count_False}")
                        
def performance_best(startcapital=50, threshold = 0.6):
    n_total_steps = len(valid_loader)
    n_total_samples = len(valid_dataset)
    sum_profit = 1.0
    mul_profit = 1.0
    n_correct = 0.0
    n_samples = 1 #Samples wird auf 1 gesetz um division by 0 zu vermeiden
    total_loss = 0.0
    count_True = 0
    count_False = 0
    all_outputs = []
    with torch.no_grad():
            for images, target in valid_loader:
                #Get necessary data
                labels = target[0]
                differences = target[1]
                #Get Outputs
                labels = labels.view(-1,1)
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                predicted = torch.round(outputs)
                n_correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                
                
                outputs = outputs.view(-1)
                outputs = outputs.tolist()
                all_outputs.extend(outputs)
                
                for i,out in enumerate(outputs):
                    if out >= threshold:
                        out_diff = differences[i]
                        print(out_diff)
                        n_samples += 1
                        mul_profit = mul_profit*(1+out_diff)
                        sum_profit += out_diff
                        if out_diff >= 0:
                            count_True += 1
                        else:
                            count_False += 1
            
            acc = 100 * n_correct / n_total_samples
            avg_loss = total_loss / n_total_steps
            print(50*"-")
            print(f"Profitability Analysis (Only trades with a confidence of {threshold}) with a startcapital of {startcapital}$:")
            print(50*"-")
            print(f"\nAccuracy of the model: {acc:.4f}%, Average Loss: {avg_loss:.4f}\n")
            print(f"Profitability: {(mul_profit*100-100):.2f}%")
            print(f"Average Profit per Trade: {((sum_profit*100)/n_samples):.4f}%   Sum of all Profits: {(sum_profit*100):.4f}%")
            print(f"Total Profit: {(startcapital*mul_profit-startcapital):.2f}$")
            print(f"Final Capital: {(startcapital*mul_profit):.2f}\n")   
            
            print(f"Winning Trades: {count_True}, Losing Trades: {count_False}\n\n")  
            
            return acc, avg_loss, ((sum_profit*100)/n_samples)

def plot_outputs():
    with torch.no_grad():
        all_outputs = []
        n_correct = 0.0
        total_loss = 0.0
        for images, target in valid_loader:
            #Get necessary data
            labels = target[0]
            differences = target[1]
            #Get Outputs
            labels = labels.view(-1,1)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
                
            predicted = torch.round(outputs)
            n_correct += (predicted == labels).sum().item()
                 
            loss = criterion(outputs, labels)
            total_loss += loss.item()
                
            outputs = outputs.view(-1)
            outputs = outputs.tolist()
            all_outputs.extend(outputs)
    bin_width = 0.05
    num_bins = int(1 / bin_width)
    
    counts, bins, _ = plt.hist(all_outputs, bins=np.linspace(0, 1, num=num_bins + 1), rwidth=0.8)
    plt.bar(bins[:-1], counts, width=bin_width, align='edge', edgecolor='black')
    plt.ylabel('Anzahl der Elemente')
    plt.show()

#train(1)
#performance_best()
#plot_outputs()

if __name__ == "__main__":
    acc_history = []
    train_loss_history = []
    valid_loss_history = []
    for epoch in range(EPOCHS):
        #Train und Validation Durchlauf
        train_loss = train(epoch)
        acc, valid_loss = valid(epoch)
        #
        loss_diff = train_loss - valid_loss
        # Aktualisiere den Learning Rate Scheduler
        lr_scheduler.step(train_loss)
        #weight_decay(loss_diff)
        
        #Speicherung aller Relevanten Daten
        acc_history.append(acc)
        train_loss_history.append(train_loss*100)
        valid_loss_history.append(valid_loss*100)
        
    plot_outputs()    
    

    
    #Den Wahrscheinlichkeits Verlauf plotten
    
    x = range(1,len(acc_history)+1)
    # Plot erstellen
    #plt.plot(x, acc_history, color="blue")
    plt.plot(x, valid_loss_history, color='#FF9999') #Hellrot
    plt.plot(x, train_loss_history, color='#CC0000') #Dunkelrot
    # Diagramm anzeigen
    plt.show()
    