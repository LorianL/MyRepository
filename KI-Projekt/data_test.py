import pandas as pd
import os
from sklearn import preprocessing
from collections import deque
import numpy as np
import random
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.regularizers import l2

SEQ_LEN = 20
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "LTC-USD"
EPOCHS = 5
BATCH_SIZE = 512
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{RATIO_TO_PREDICT}-PAIR-{int(time.time())}"

print("Is GPU available: ", tf.config.list_physical_devices('GPU'))

def preprocess_df(df):
    df = df.drop("future",axis=1)
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            df[col] = preprocessing.scale(df[col].values)
            df = df.dropna()
    print(df)
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append(np.array(i[:-1]))
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
    return np.array(x),np.array(y)

#Falls der Preis steigt, so wird eine 1 zurück gegeben, falls er fällt oder gleich bleibt so 0
def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


#df = pd.read_csv("V3/crypto_data/BCH-USD.csv", names=["time","close", "volume"])
main_df = pd.DataFrame()


ratios = ["LTC-ETH", "LTC-BTC","BTC-USD","ETH-USD","LTC-USD"] #, "BCH-USD"
for ratio in ratios:
    data_set = f"V3/crypto_data/{ratio}.csv"
    df = pd.read_csv(data_set)
    print(ratio, df.shape)
    df["sma50"] = df['close'].rolling(window=50).mean()
    df["sma100"] = df['close'].rolling(window=100).mean()
    df["sma150"] = df['close'].rolling(window=150).mean()
    df["sma200"] = df['close'].rolling(window=200).mean()
    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume", "sma50": f"{ratio}_sma50","sma100": f"{ratio}_sma100","sma150": f"{ratio}_sma150","sma200": f"{ratio}_sma200"}, inplace=True)
    df.set_index("time", inplace=True)
    df = df[[f"{ratio}_close",f"{ratio}_volume",f"{ratio}_sma50",f"{ratio}_sma100",f"{ratio}_sma150",f"{ratio}_sma200"]]
    
    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)



main_df["future"] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)
main_df["target"] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"]))
main_df = main_df.dropna()
preprocess_df(main_df)


print(main_df.head())
