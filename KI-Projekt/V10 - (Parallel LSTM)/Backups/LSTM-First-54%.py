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
NUM_EPOCHS = 500

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
hidden_size = 100
num_layers = 1
expansion_factor = 2

embedding_dropout_prob = 0
lstm_dropout_prob = 0
dropout_prob = 0.3

l2_lambda = 0.00004#0.000001
leaky_slope = 0.01
WEIGHT_CLIP_VALUE = 0.5
GRADIENT_CLIP_VALUE = 1.2#2
learning_rate = 0.001 #0.00005 bei Batch 64
#--------------------------------Model Class----------------------------------------
input_size = TOKENIZING_LENGTH
output_size = 1
n_embeddings = ENCODING_LENGTH
dim = TOKENIZING_LENGTH

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.leaky_slope = leaky_slope
        
        # Initialisiere den versteckten Zustand
        self.h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        self.c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        #Layers
        self.emb = Embedding(n_embeddings, dim)
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout_prob) #
        
        self.fc1 = nn.Linear(hidden_size,int(hidden_size*expansion_factor))
        self.fc2 = nn.Linear(int(hidden_size*expansion_factor), hidden_size)
        self.fc3 = nn.Linear(hidden_size,output_size)

        #self.bn1 = nn.BatchNorm1d(int(hidden_size*expansion_factor))
        #self.bn2 = nn.BatchNorm1d(int(hidden_size))
        
        #Dropout und Aktivierungsfunktionen
        
        #self.bn1 = nn.BatchNorm1d(int(hidden_size/4))
        #self.dropout_features = nn.Dropout(p=features_dropout_prob)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.emb_dropout = nn.Dropout(p=embedding_dropout_prob)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        
        x = self.emb(x)
        x = self.emb_dropout(x)

        # Führe die Vorwärtsberechnung des RNN durch
        x, (h_n, c_n) = self.lstm(x, (self.h0, self.c0))
        
        
        #x = x.mean(dim=1) #x[:, -1, :]
        x = x[:, -1, :]

        x = self.fc1(x)
        #x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=self.leaky_slope)
        x = self.dropout(x)
        
        x = self.fc2(x)
        #x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=self.leaky_slope)
        x = self.dropout(x)
        
        # Verwende den letzten Ausgabe-Zeitschritt des RNN
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x



classes_categories = ["sell", "buy"]

model = LSTM(input_size, hidden_size, num_layers, output_size,dropout_prob).to(device)
#model = model.apply(initialize_weights)


criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda) #, weight_decay=l2_lambda



def train(epoch):
    model.train()
    
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
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_VALUE)
        
        optimizer.step()
        
        clip_weights(model, clip_value=WEIGHT_CLIP_VALUE)
        
        total_loss += loss.item()
        
        optimizer.zero_grad()
        
        
        
        if (i+1) % 1000 == 0:
            print(f"Epoch[{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{n_total_steps}], Loss: {(total_loss/(i+1)):.4f}")

def test(epoch, highest_acc):
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
        print(f"\n\nAccuracy in Epoch {epoch+1}: {acc:.4f}%     Loss: {avg_loss:.4f}\t\t {n_correct}+{n_false} \nHighest_Acc was: {highest_acc:.4f}%\n\n")
        
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

#visualize_weights_state(model)
#plot_weights()
#plot_outputs(showing=False)
#print_weights_state()
if __name__ == "__main__":
    print_information(model)
    highest_acc = 0
    best_model = None
    
    for epoch in range(NUM_EPOCHS):

        print(20*"-")
        
        train(epoch)
        acc,loss = test(epoch, highest_acc)
        
        
        
        
        if acc > highest_acc:
            highest_acc = acc
            best_model = copy.deepcopy(model)
            #torch.save(best_model, f"{script_folder}/Models/Acc-{acc:.2f}.pt")
                    
        print(20*"-")
        plot_outputs(epoch)
    addModelStats()
    
    if best_model != None:
        torch.save(best_model, f"{script_folder}/Models/Acc-{acc:.2f}.pt")
        
    print_weights_state()
    plot_outputs(showing=True)
    
    print(f"Enc.Length: {ENCODING_LENGTH}, Tok. Length: {TOKENIZING_LENGTH}, Hiddensize: {hidden_size}, num_layers: {num_layers}, exp.factor: {expansion_factor}, lstm_drop: {lstm_dropout_prob}, drop: {dropout_prob}, Total Params: {print_information(model)}")












#    weight_decay_values = [] #,0.55,0.3     0.9,0.9,0.9,0.9,0.9,0.9
#    print_weights_state()


"""
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
        """









"""# Iteriere über die einzelnen Schichten des Modells und gib die Größe der Parameter aus
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Schichtname: {name}, Größe der Parameter: {param.size()}")"""