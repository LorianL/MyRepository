import pandas as pd
import os
from sklearn import preprocessing
from collections import deque
import numpy as np
import random
import matplotlib.pyplot as plt
import math


df = pd.read_csv("C:/Users/Lorian/Desktop/Trading_AI/V4 - {Normalisation, Indicators}/BTC-USD.csv")
volume = df["volume"].tolist()
print(max(volume))
print(min(volume))
#print(volume)