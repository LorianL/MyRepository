import pandas as pd
import os
from sklearn import preprocessing
from collections import deque
import numpy as np
import random

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3


#Falls der Preis steigt, so wird eine 1 zurück gegeben, falls er fällt oder gleich bleibt so 0
def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess_df(df):
    df = df.drop("future",axis=1)
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df = df.dropna()
            df[col] = preprocessing.scale(df[col].values)
            df = df.dropna()

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days),i[-1]])
    random.shuffle(sequential_data)

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

    sequential_data = buys + sells

    random.shuffle(sequential_data)

    x = []
    y = []

    for seq,target in sequential_data:
        x.append(seq)
        y.append(target)
    return np.array(x),y
    #-------------------------------------------------------------------------------

df = pd.read_csv("BTC_Data.csv") # time      open      high       low     close      volume
df = df.set_index("time")


df["future"] = df["close"].shift(-FUTURE_PERIOD_PREDICT)
df["target"] = list(map(classify, df["close"], df["future"]))
df = df.dropna()
#-------------------------------------------------------------------------------

times = sorted(df.index.values)
last_5pct = times[- int(0.05*len(times))]


validation_df = df[(df.index >= last_5pct)]
main_df = df[(df.index < last_5pct)]

#train_x,train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_df)

print(f"train_data: validation:{len(validation_x)} ") #{len(train_x)}
#print(f"Sells: {train_y.count(0)} Buys: {train_y.count(1)}")
