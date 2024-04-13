import pandas as pd
import os
from sklearn import preprocessing
from collections import deque
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Compose
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
    
    
    
def empty_folder(folder_path):
    try:
        # Überprüfen, ob der angegebene Pfad ein Verzeichnis ist
        if os.path.isdir(folder_path):
            # Durchlaufen Sie alle Dateien und Unterverzeichnisse im Ordner
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                
                # Prüfen, ob es sich um eine Datei handelt und keine Unterverzeichnisse
                if os.path.isfile(file_path):
                    # Löschen Sie die Datei
                    os.remove(file_path)
            print(f"Der Ordner '{folder_path}' wurde erfolgreich geleert.")
        else:
            print(f"'{folder_path}' ist kein gültiger Ordner.")
    except Exception as e:
        print(f"Fehler beim Leeren des Ordners '{folder_path}': {str(e)}")

# Beispielaufruf: Leeren Sie den Ordner 'beispielordner'
empty_folder('V8 - (LSTM)/Graphs')


SEQ_LEN = 30 #30
FUTURE_PERIOD_PREDICT = 20 #20
BATCH_SIZE = 512
NUM_OF_PERIODS = 10
num_epoch = 1000



def get_data():
 #-------------------------------------------------------   


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

    def add_cmf(df, period=20, norm_fn = z_score_normalize):
        mfm = ((2 * df['close'] - df['low'] - df['high']) / (df['high'] - df['low'])) * df['volume']
        cmf = mfm.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        cmf = norm_fn(cmf)
        #cmf = min_max_normalize(cmf)
        df[f"cmf_{period}"] = cmf
        return df

    def add_roc(df, period=12, norm_fn = z_score_normalize):
        roc = (df['close'] - df['close'].shift(periods=period)) / df['close'].shift(periods=period)
        roc = norm_fn(roc)
        #roc = min_max_normalize(roc)
        df[f'roc_{period}'] = roc
        return df

    def add_vo(df, short_period=14, long_period=28, norm_fn = z_score_normalize):
        short_ma = df['volume'].rolling(window=short_period).mean()
        long_ma = df['volume'].rolling(window=long_period).mean()
        vo = (short_ma - long_ma) / long_ma
        vo = norm_fn(vo)
        #vo = min_max_normalize(vo)
        df[f'vo_{short_period}_{long_period}'] = vo
        return df

    def add_cci(df, period=20, norm_fn = z_score_normalize):
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = np.abs(tp - sma).rolling(window=period).mean()
        cci = (tp - sma) / (0.015 * mad)
        cci = norm_fn(cci)
        #cci = min_max_normalize(cci)
        df[f'cci_{period}'] = cci
        return df

    def add_cmo(df, period=14, norm_fn = z_score_normalize):
        diff = df['close'].diff()
        pos_sum = diff.where(diff > 0, 0).rolling(window=period).sum()
        neg_sum = np.abs(diff.where(diff < 0, 0)).rolling(window=period).sum()
        cmo = ((pos_sum - neg_sum) / (pos_sum + neg_sum))
        cmo = norm_fn(cmo)
        #cmo = min_max_normalize(cmo)
        df[f'cmo_{period}'] = cmo
        return df

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

    def add_bb_bandwidth(df, window=20, std=2 , norm_fn = z_score_normalize):
        df['bb_upper'] = df['close'].rolling(window=window).mean() + std * df['close'].rolling(window=window).std()
        df['bb_lower'] = df['close'].rolling(window=window).mean() - std * df['close'].rolling(window=window).std()
        bb_bandwidth = (df['bb_upper'] - df['bb_lower']) / df['close'].rolling(window=window).mean()
        bb_bandwidth = norm_fn(bb_bandwidth)
        #bb_bandwidth = min_max_normalize(bb_bandwidth)
        df[f'bb_bandwidth_{window}'] = bb_bandwidth
        df.drop(['bb_upper', 'bb_lower'], axis=1, inplace=True)
        return df

    def add_ema(df, span=20, norm_fn = z_score_normalize):
        ema = df['close'].ewm(span=span).mean()
        #ema = sma_normalize(ema)
        ema = norm_fn(ema)
        #ema = min_max_normalize(ema)
        col_name = f'ema_{span}'
        df[col_name] = ema
        
        return df

    def add_new_target(df):
        def difference_target(current, future):
            return (future / current)*100 - 100
        def classify_mod(difference,lim):
            if float(difference) > float(lim):
                return 1
            else:
                return 0
        temp_df = pd.DataFrame()
        temp_df["future"] = df["close"].shift(-FUTURE_PERIOD_PREDICT)
        temp_df["diff"] = list(map(difference_target, df["close"], temp_df["future"]))
        #Berechnung des Grenzwertes:
        data = temp_df['diff'].tolist()
        data.sort()
        lim = data[int(0.50*len(data))]
        print(f"lim: {round(lim,4)}")
        #---
        df["target"] = list(map(lambda x: classify_mod(x, lim), temp_df["diff"]))
        return df
        

    df = pd.read_csv("C:/Users/Lorian/Desktop/Trading_AI/V8 - (LSTM)/BTC-USD.csv")
    df = df.set_index("time")
    df = add_rsi(df)
    df = add_mfi(df)
    df = add_atr(df)
    df = add_cmf(df)
    df = add_roc(df)
    df = add_vo(df)
    df = add_cci(df)
    df = add_cmo(df)
    df = add_eom(df)
    df = add_bb_bandwidth(df)
    df = add_ema(df)
    df = add_new_target(df)

    df["close"] = z_score_normalize(df["close"])
    df["volume"] = z_score_normalize(df["volume"])


    df = df.drop(columns=["open","high","low"])
    df = df.dropna()

    print(df.head(5))
    return df

def divide_df(df):
    times = sorted(df.index.values)
    last_5pct = times[- int(0.05*len(times))]

    train_df = df[(df.index < last_5pct)]
    valid_df = df[(df.index >= last_5pct)]
    
    return train_df, valid_df

def preprocessing(df):
    
    def sequentialise(df):
        indicators = list(df.columns)
        
        sequential_data = []
        prev_days = deque(maxlen=SEQ_LEN)

        for i in df.values:
            prev_days.append(np.array(i[:-1]))
            if len(prev_days) == SEQ_LEN:
                sequential_data.append([np.array(prev_days, dtype=np.float32),i[-1]])
        random.shuffle(sequential_data)
        return sequential_data,indicators

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

    
    sequential_data,indicators = sequentialise(df)
    print(indicators)
    sequential_data = balancing(sequential_data)
    #print(sequential_data[0])
    
    return np.array(sequential_data)

class CustomDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            array, target = self.data[index]
            return torch.from_numpy(array), torch.tensor(target, dtype=torch.float32)
#------------------------------------------------------------------------------
if True:
    df = get_data()
    train_df,valid_df = divide_df(df)
    train_data = preprocessing(train_df)
    valid_data = preprocessing(valid_df)
    print(f"train_data: {len(train_data)} validation:{len(valid_data)} ")



    #transform = Compose([ToTensor()])

    train_dataset = CustomDataset(train_data)
    test_dataset = CustomDataset(valid_data)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    x,y = next(iter(train_loader))
#print(x, y)

#Model
#------------------------------------------------------------------

def dropout_features(input_tensor):
    if train_mode:
        batchsize = input_tensor.shape[0]
        sequence_length = input_tensor.shape[1]
        features = input_tensor.shape[2]
        
        t0 = torch.zeros(batchsize,sequence_length)
        #Liste mit allen Möglichen Feature indexen
        all_features = list(range(features))
        #Entnimmt zufällig die features
        selected_features = random.sample(all_features, int(features*dropout_features_prob))
        
        for f in selected_features:
            input_tensor[:,:,f] = t0

        return input_tensor
    else:
        return input_tensor


global dropout_features_prob
dropout_features_prob = 0.9

global train_mode
train_mode = True


#---------------------------------------------------------------
input_size = x.shape[-1]
hidden_size = 100
num_layers = 3
output_size = 1


lstm_dropout_prob = 0.3
dropout_prob = 0.2

l2_lambda = 0.0001
print(x.size(0))
#implement rnn Net
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Initialisiere den versteckten Zustand
        self.h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        self.c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        #Layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout_prob)
        
        self.fc1 = nn.Linear(hidden_size,int(hidden_size/4))
        self.fc2 = nn.Linear(int(hidden_size/4), int(hidden_size/8))
        self.fc3 = nn.Linear(int(hidden_size/8), output_size)
        
        #Dropout und Aktivierungsfunktionen
        
        self.bn1 = nn.BatchNorm1d(int(hidden_size/4))
        self.bn2 = nn.BatchNorm1d(int(hidden_size/8))
        #self.dropout_features = nn.Dropout(p=features_dropout_prob)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        
        x = dropout_features(x)
        
        # Führe die Vorwärtsberechnung des RNN durch
        x, (h_n, c_n) = self.lstm(x, (self.h0, self.c0))
        
        
        #x = x.mean(dim=1) #x[:, -1, :]
        x = x[:, -1, :]

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Verwende den letzten Ausgabe-Zeitschritt des RNN
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x



classes_categories = ["sell", "buy"]

model = LSTM(input_size, hidden_size, num_layers, output_size,dropout_prob).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=l2_lambda) #, weight_decay=l2_lambda

lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.33, patience=20, verbose=True)



def train(epoch):
    model.train()
    train_mode = True
    
    n_total_steps = len(train_loader)
    total_loss = 0
    
    for i, (seq, labels) in enumerate(train_loader):
        #Input shape [Sequenzlänge, Batchsize, Features Anzahl]
        if seq.shape[0] != BATCH_SIZE:
            break
        seq = seq.to(device)


        #labels = labels.view(-1,1)
        labels = labels.to(device)
        
        #Forward pass
        outputs = model(seq)

        outputs = outputs.view(-1)
        #print(outputs,labels)
        #exit()
        loss = criterion(outputs, labels)

        #Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        optimizer.zero_grad()
        
        
        
        if (i+1) % 1000 == 0:
            print(f"Epoch[{epoch+1}/{num_epoch}], Step [{i+1}/{n_total_steps}], Loss: {(total_loss/(i+1)):.4f}")

def test(epoch, highest_acc):
    model.eval()
    train_mode = False
    
    
    with torch.no_grad():
        n_correct = 0
        n_false = 0
        n_samples = len(test_dataset)
        n_total_steps = len(test_loader)
        total_loss = 0 
        print(n_samples)
        for seq, labels in test_loader:
        #Input shape [Sequenzlänge, Batchsize, Features Anzahl]
            if seq.shape[0] != BATCH_SIZE:
                break
            seq = seq.to(device)


            labels = labels.to(device)
            
            #Forward pass
            outputs = model(seq)
            outputs = outputs.view(-1)
            
            #loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            
            #max returns (value, index)
            predicted = torch.round(outputs)
            for p,l in zip(predicted,labels):
                if p == l:
                    n_correct +=1
                else:
                    n_false += 1
                
            #n_correct += (predicted == labels).sum().item()
            
            
            
        acc = 100 * n_correct / n_samples
        avg_loss = total_loss / n_total_steps
        print(f"\n\nAccuracy in Epoch {epoch+1}: {acc:.4f}%     Loss: {avg_loss:.4f}\t\t {n_correct}+{n_false} \nHighest_Acc was: {highest_acc:.4f}%\n\n")
        
        weights_tensor_copy = copy.deepcopy(model.state_dict()[f"fc1.weight"]).to("cpu")
        weights = weights_tensor_copy.numpy()
        print(f"Max Weight: {weights.max()} \t Min Weight: {weights.min()}")
        return acc, avg_loss


#Plotting Functions

def plot_outputs(epoch=-1, showing = False, loader_name = test_loader):
    with torch.no_grad():
        all_outputs = []
        n_correct = 0.0
        total_loss = 0.0
        for seq, labels in loader_name:
            if seq.shape[0] != BATCH_SIZE:
                break
            seq = seq.to(device)

            #labels = labels.long()
            labels = labels.to(device)
        
            outputs = model(seq)
            outputs = outputs.view(-1)
            #max returns (value, index)
            predicted = torch.round(outputs)
            n_correct += (predicted == labels).sum().item()
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
                
            outputs = outputs.view(-1)
            outputs = outputs.tolist()
            all_outputs.extend(outputs)
        bin_width = 0.01
        num_bins = int(1 / bin_width)
        
        counts, bins, _ = plt.hist(all_outputs, bins=np.linspace(0, 1, num=num_bins + 1), rwidth=0.8)
        
        plt.bar(bins[:-1], counts, width=bin_width, align='edge', edgecolor='black')
        plt.ylabel('Anzahl der Elemente')
        plt.savefig(f"V8 - (LSTM)/Graphs/LossDistribution-Epoch{epoch+1}.png")
        if showing:
            plt.show()

        plt.clf()

def plot_weights():
    all_weights = model.state_dict()[f"fc1.weight"].view(-1).tolist()

    bin_width = 0.01
    num_bins = int(1 / bin_width)
        
    counts, bins, _ = plt.hist(all_weights, bins=np.linspace(0, 1, num=num_bins + 1), rwidth=0.8)
        
    plt.bar(bins[:-1], counts, width=bin_width, align='edge', edgecolor='black')
    plt.ylabel('Anzahl der Elemente')
    #plt.savefig(f"V8 - (LSTM)/Graphs/LossDistribution-Epoch{epoch+1}.png")
    plt.show()

    #plt.clf()

def print_information(model):
    # Zähle die Gesamtzahl der Parameter im Modell
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Gesamtzahl der Parameter: {total_params}")

    """# Iteriere über die einzelnen Schichten des Modells und gib die Größe der Parameter aus
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Schichtname: {name}, Größe der Parameter: {param.size()}")"""

def print_weights_state(layer_name = "fc1"):
    # Nehmen wir an, 'weights' ist ein NumPy-Array mit den Gewichtswerten eines bestimmten Layers
    weights_tensor_copy = copy.deepcopy(model.state_dict()[f"{layer_name}.weight"]).to("cpu")
    weights = weights_tensor_copy.numpy()

    # Normalisieren Sie die Gewichtswerte, um sie auf den Bereich [0, 1] abzubilden
    normalized_weights = (weights - weights.min()) / (weights.max() - weights.min())
    print(f"Max Weight: {weights.max()} \t Min Weight: {weights.min()}")

    # Erstellen Sie ein Bild (Matrix) entsprechend den Gewichtswerten
    # Hier verwenden wir ein viridis-Farbmap, aber Sie können andere Colormaps auswählen
    # Je niedriger der Wert, desto dunkler Blau; je höher der Wert, desto röter
    image = plt.get_cmap('viridis')(normalized_weights)

    # Zeigen Sie das Bild an
    plt.imshow(image)
    plt.colorbar()  # Fügen Sie eine Farbskala hinzu
    plt.title('Gewichtsvisualisierung des Layers')
    plt.show()

plot_outputs(showing=False)
#print_weights_state()
if __name__ == "__main__":
    print_information(model)
    highest_acc = 0
    best_model = None
    weight_decay_values = [] #,0.55,0.3     0.9,0.9,0.9,0.9,0.9,0.9
    print_weights_state()
    for epoch in range(num_epoch):
        
        if epoch % 5 == 0:
            plot_weights()
            if len(weight_decay_values) != 0:
                dropout_features_prob = weight_decay_values.pop(0)
                print(f"New Feature Dropout Probability = {dropout_features_prob}")
            else:
                dropout_features_prob = 0
                print(f"Feature Dropout Probability = {dropout_features_prob}")
            #print_weights_state()
            pass
    
    
        print(20*"-")
        
        
        train(epoch)
        acc,loss = test(epoch, highest_acc)
        
    
        if acc > highest_acc:
            highest_acc = acc
            best_model = copy.deepcopy(model)
            torch.save(best_model, f"V8 - (LSTM)/Models/Acc-{acc:.2f}.pt")
            
        
        lr_scheduler.step(loss)
        
        print(20*"-")
        plot_outputs(epoch)
    plot_outputs(showing=True)














