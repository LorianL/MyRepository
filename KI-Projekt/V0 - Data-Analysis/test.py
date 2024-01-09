import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
"""
# Generieren Sie zufällige Daten, die einer Normalverteilung folgen
mu, sigma = 0, 0.2 # Mittelwert und Standardabweichung
daten = np.random.normal(mu, sigma, 100000)

# Erstellen Sie eine Liste von Intervallen mit Schrittweite 0,01
intervalle = [i/100 for i in range(-100, 101)]

# Erstellen Sie ein Histogramm mit den Daten und Intervallen
plt.hist(daten, bins=intervalle)

# Beschriften Sie die Achsen und den Titel
plt.xlabel('Wertebereich')
plt.ylabel('Anzahl')
plt.title('Normalverteiltes Balkendiagramm')

# Zeigen Sie das Diagramm an
plt.show()"""

df = pd.DataFrame()

df["A"] = [np.nan,None,None,None]
print(df)