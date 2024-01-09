import pandas as pd
import os
from sklearn import preprocessing
from collections import deque
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import matplotlib.pyplot as plt

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
empty_folder('V7 -  (RNN Bugfix)/Graphs')


SEQ_LEN = 30
FUTURE_PERIOD_PREDICT = 20
EPOCHS = 20
BATCH_SIZE = 512
NUM_OF_PERIODS = 10
num_epoch = 1000



input_size = 13
hidden_size = 500
num_layers = 1
output_size = 1
dropout_prob = 0.1
l2_lambda = 0.01

#implement rnn Net
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Initialisiere den versteckten Zustand
        self.h0 = torch.zeros(self.num_layers, 13, self.hidden_size).to(device)
        self.h0 = self.h0.to(device)
        # layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_size,int(hidden_size/4))
        self.fc2 = nn.Linear(int(hidden_size/4), int(hidden_size/8))
        self.fc3 = nn.Linear(int(hidden_size/8), int(hidden_size/8))
        self.fc4 = nn.Linear(int(hidden_size/8), output_size)
        
        self.dropout = nn.Dropout(p=dropout_prob)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        # Initialisiere den versteckten Zustand
        #h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Führe die Vorwärtsberechnung des RNN durch
        x, y = self.rnn(x, self.h0)
        
        #print(x.shape)
        #print(y.shape)
        #x = x.mean(dim=1) #x[:, -1, :]
        x = x[:, -1, :]
        #print(x.shape, y.shape)

        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.relu(x)
        
        x = self.dropout(x)
        
        # Verwende den letzten Ausgabe-Zeitschritt des RNN
        x = self.fc4(x)
        x = self.sigmoid(x)
        
        return x



classes_categories = ["sell", "buy"]

model = RNN(input_size, hidden_size, num_layers, output_size,dropout_prob).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.33, patience=10, verbose=True)

def print_information(model):
    # Zähle die Gesamtzahl der Parameter im Modell
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Gesamtzahl der Parameter: {total_params}")

    """# Iteriere über die einzelnen Schichten des Modells und gib die Größe der Parameter aus
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Schichtname: {name}, Größe der Parameter: {param.size()}")"""





if __name__ == "__main__":
    print_information(model)

    print("Layers:")
    for name, module in model.named_modules():
        print(name)
        

import matplotlib.pyplot as plt
import numpy as np
import copy
def print_weights_state(layer_name = "fc1"):
    # Nehmen wir an, 'weights' ist ein NumPy-Array mit den Gewichtswerten eines bestimmten Layers
    weights_tensor_copy = copy.deepcopy(model.state_dict()[f"{layer_name}.weight"]).to("cpu")
    weights = weights_tensor_copy.numpy()

    # Normalisieren Sie die Gewichtswerte, um sie auf den Bereich [0, 1] abzubilden
    normalized_weights = (weights - weights.min()) / (weights.max() - weights.min())

    # Erstellen Sie ein Bild (Matrix) entsprechend den Gewichtswerten
    # Hier verwenden wir ein viridis-Farbmap, aber Sie können andere Colormaps auswählen
    # Je niedriger der Wert, desto dunkler Blau; je höher der Wert, desto röter
    image = plt.get_cmap('viridis')(normalized_weights)

    # Zeigen Sie das Bild an
    plt.imshow(image)
    plt.colorbar()  # Fügen Sie eine Farbskala hinzu
    plt.title('Gewichtsvisualisierung des Layers')
    plt.show()
    
print_weights_state()














