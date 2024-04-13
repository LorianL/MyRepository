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
NUM_EPOCHS = 200

ENCODING_LENGTH = 150
TOKENIZING_LENGTH = 40

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
    
#------------------------------Model Hyperparameter---------------------------------
num_layers = 3
forward_expansion = 2
heads = 4

dropout_prob = 0.1

learning_rate = 0.001 #0.00005 bei Batch 64
l2_lambda = 0.000001#0.000001
#----------------------------Transformer------------------------------------------------------

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"
        
        self.values_linear = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys_linear = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.queries_linear = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)
        
    def forward(self, values, keys, query):
        N = query.shape[0] #How many examples we send in at the same time aka Batchsize
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1] #the lens may vary
        
        #split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)
        
        #Addatpation layer
        
        values = self.values_linear(values)
        keys = self.keys_linear(keys)
        query = self.queries_linear(query)
        
        #Multiplie querys with the keys
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, keys])
        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # energy shape: (N, heads, query_len, key_len)
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention,values]).reshape(N, query_len, self.heads * self.head_dim)
        #attention shape: (N, heads, query_len, key_len)
        #values shape: (N, value_len, heads, heads_dim)
        #after einsum (N, query_len, heads, head_dim), flatten last two dims
        
        out = self.fc_out(out)
        return out




class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query):
        
        #Attention Part
        attention = self.attention(value, key, query)
        out = attention + query
        out = self.norm1(out)
        out = self.dropout(out)
        
        #Feed-Forward part
        
        forward = self.feed_forward(out)
        out = forward + out
        out = self.norm2(out)
        out = self.dropout(out)
        
        return out



class Encoder(nn.Module):
    def __init__(self, encoding_length, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        
        self.number_embedding = nn.Embedding(encoding_length, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size, 
                    heads, 
                    dropout=dropout, 
                    forward_expansion=forward_expansion)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        
        self.fcOut = nn.Linear(embed_size*SEQ_LEN,1)
    
    def forward(self, x):
        N, seq_length = x.shape
        positions = torch.arange(0,seq_length).expand(N,seq_length).to(self.device)
        
        x = self.number_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, x, x)
        

        x = x.view(N,-1)

        x = self.fcOut(x)

        x = F.sigmoid(x)
        
        return x



classes_categories = ["sell", "buy"]

model = Encoder(ENCODING_LENGTH, TOKENIZING_LENGTH, num_layers, heads, device, forward_expansion, dropout_prob, SEQ_LEN).to(device)
#model = model.apply(initialize_weights)


criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda) #, weight_decay=l2_lambda
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
#-----------------------------Managing Part-----------------------------

def train(epoch):
    model.train()
    
    n_total_steps = len(train_loader)
    total_loss = 0
    n_correct = 0
    
    
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
        
        #max returns (value, index)
        predicted = torch.round(outputs)
                
        n_correct += (predicted == labels).sum().item()
            
        if (i+1) % 1000 == 0:
            acc = 100 * n_correct / (i*BATCH_SIZE)
            print(f"Epoch[{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{n_total_steps}], Loss: {(total_loss/(i+1)):.4f}, Acc: {acc:.2f}%   \t\t\tLoss: {(total_loss/(i+1)):.6f}")
        
        optimizer.zero_grad()

def valid(epoch, highest_acc):
    model.eval()    
    
    with torch.no_grad():
        n_correct = 0
        n_false = 0
        n_samples = len(valid_dataset)
        n_total_steps = len(valid_loader)
        total_loss = 0 
        print(n_samples)
        for seq, labels in valid_loader:
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
        print(f"\n\nAccuracy in Epoch {epoch+1}: {acc:.4f}%     Loss: {avg_loss:.4f}\t\t {n_correct}+{n_false} \nHighest_Acc was: {highest_acc:.6f}%\n\n")
        
        weights_tensor_list = copy.deepcopy(list(model.state_dict().values()))

        min_weight = min([torch.min(tensor).item() for tensor in weights_tensor_list])
        max_weight = max([torch.max(tensor).item() for tensor in weights_tensor_list])

        print(f"Min Weight: {min_weight} \t Max Weight: {max_weight}")
        return acc, avg_loss


#Plotting Functions

def plot_outputs(epoch=-1, showing = False, loader_name = valid_loader):
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
        plt.savefig(f"{script_folder}/Graphs/LossDistribution-Epoch{epoch+1}.png")
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
    return total_params
    

def print_weights_state():
    for layer_name in model.state_dict().keys():
        if "weight" in layer_name:
            # Nehmen wir an, 'weights' ist ein NumPy-Array mit den Gewichtswerten eines bestimmten Layers
            weights_tensor_copy = copy.deepcopy(model.state_dict()[layer_name]).to("cpu")
            weights = weights_tensor_copy.numpy()

            # Normalisieren Sie die Gewichtswerte, um sie auf den Bereich [0, 1] abzubilden
            normalized_weights = (weights - weights.min()) / (weights.max() - weights.min())
            print(f"{layer_name}: Min Weight: {weights.min()} \t Max Weight: {weights.max()} ")

            # Erstellen Sie ein Bild (Matrix) entsprechend den Gewichtswerten
            # Hier verwenden wir ein viridis-Farbmap, aber Sie können andere Colormaps auswählen
            # Je niedriger der Wert, desto dunkler Blau; je höher der Wert, desto röter
            image = plt.get_cmap('viridis')(normalized_weights)

            # Zeigen Sie das Bild an
            plt.imshow(image)
            plt.colorbar()  # Fügen Sie eine Farbskala hinzu
            plt.title(f'Gewichtsvisualisierung des Layers: {layer_name}')
            plt.show()




def addModelStats():
    existing_csv_file = f"{script_folder}\ModelStats.csv"
    df = pd.read_csv(existing_csv_file)

    new_row = {"SEQ_LEN": SEQ_LEN, "FUTURE_PERIOD_PREDICT": FUTURE_PERIOD_PREDICT, "BATCH_SIZE":BATCH_SIZE, "NUM_EPOCHS":NUM_EPOCHS, "ENCODING_LENGTH": ENCODING_LENGTH, "TOKENIZING_LENGTH":TOKENIZING_LENGTH, 
               "hidden_size": hidden_size, "num_layers": num_layers, "expansion_factor": expansion_factor, "embedding_dropout_prob": embedding_dropout_prob, "lstm_dropout_prob": lstm_dropout_prob, "dropout_prob": dropout_prob,
               "l2_lambda": l2_lambda, "acc": highest_acc, "params": print_information(model)}
    
    new_row_df = pd.DataFrame(new_row, index=[0])
    
    df = pd.concat([df, new_row_df], ignore_index=True)
    
    df.to_csv(existing_csv_file, index=False)
    
    print(new_row)


print(model.state_dict().keys())


if __name__ == "__main__":
    print_information(model)
    highest_acc = 0
    best_model = None
    
    for epoch in range(NUM_EPOCHS):

        print(20*"-")
        
        train(epoch)
        acc,loss = valid(epoch, highest_acc)
        
        
        
        
        if acc > highest_acc:
            highest_acc = acc
            best_model = copy.deepcopy(model)
            #torch.save(best_model, f"{script_folder}/Models/Acc-{acc:.2f}.pt")
                    
        print(20*"-")
        plot_outputs(epoch)
        lr_scheduler.step(loss)
    #addModelStats()
    
    #f best_model != None:
    torch.save(best_model, f"{script_folder}/Models/Acc-{acc:.2f}.pt")
        
    print_weights_state()
    plot_outputs(showing=True)
    
    print(f"Enc.Length: {ENCODING_LENGTH}, Tok. Length: {TOKENIZING_LENGTH},  num_layers: {num_layers}, exp.factor: {forward_expansion}, drop: {dropout_prob}, Total Params: {print_information(model)}")

