"""import matplotlib.pyplot as plt
import numpy as np

# Liste mit Werten zwischen 0 und 1
data = np.random.random(50000)

# Balkenabstand und Anzahl der Balken
bin_width = 0.05
num_bins = int(1 / bin_width)

# Histogramm erstellen, um die Anzahl der Elemente in jedem Balken zu berechnen
counts, bins, _ = plt.hist(data, bins=np.linspace(0, 1, num=num_bins + 1), rwidth=0.8)

# Balkendiagramm erstellen
plt.bar(bins[:-1], counts, width=bin_width, align='edge', edgecolor='black')

# Achsenbeschriftungen und Titel
plt.xlabel('Werte')
plt.ylabel('Anzahl der Elemente')
plt.title('Balkendiagramm der Daten')

# Plot anzeigen
plt.show()"""

import random
import torch
"""for _ in range(10):
    print(random.randint(0,1))"""
    
    
t = torch.zeros(256,30,13)
print(t.shape)
t = t.view(256, -1)
print(t.shape)


print(int(torch.tensor(1.5)))
