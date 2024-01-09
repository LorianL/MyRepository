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


SEQ_LEN = 30
FUTURE_PERIOD_PREDICT = 3
EPOCHS = 10
BATCH_SIZE = 512
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"
print("Is GPU available: ", tf.config.list_physical_devices('GPU'))
def get_data():
    def classify(current, future):
        if float(future) > float(current):
            return 1
        else:
            return 0
        
    def add_rsi(df, period=14):
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 1 - (1 / (1 + rs))
        df[f'rsi_{period}'] = rsi
        return df

    def add_atr(df, period=14):
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        df[f'atr_{period}'] = atr
        return df

    def add_mfi(df, period=14):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        positive_flow = raw_money_flow.where(df['close'] > df['close'].shift(), 0)
        negative_flow = raw_money_flow.where(df['close'] < df['close'].shift(), 0)
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        money_ratio = positive_mf / negative_mf
        mfi = 1 - (1 / (1 + money_ratio))
        df[f'mfi_{period}'] = mfi
        return df

    def add_cmf(df, period=20):
        mfm = ((2 * df['close'] - df['low'] - df['high']) / (df['high'] - df['low'])) * df['volume']
        cmf = mfm.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        df[f'cmf_{period}'] = cmf
        return df

    def add_roc(df, period=12):
        roc = (df['close'] - df['close'].shift(periods=period)) / df['close'].shift(periods=period)
        df[f'roc_{period}'] = roc
        return df

    def add_vo(df, short_period=14, long_period=28):
        short_ma = df['volume'].rolling(window=short_period).mean()
        long_ma = df['volume'].rolling(window=long_period).mean()
        vo = (short_ma - long_ma) / long_ma
        df[f'vo_{short_period}_{long_period}'] = vo
        return df

    def add_cci(df, period=20):
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = np.abs(tp - sma).rolling(window=period).mean()
        cci = (tp - sma) / (0.015 * mad)
        df[f'cci_{period}'] = cci
        return df

    def add_cmo(df, period=14):
        diff = df['close'].diff()
        pos_sum = diff.where(diff > 0, 0).rolling(window=period).sum()
        neg_sum = np.abs(diff.where(diff < 0, 0)).rolling(window=period).sum()
        cmo = ((pos_sum - neg_sum) / (pos_sum + neg_sum)) * 100
        df[f'cmo_{period}'] = cmo
        return df

    def add_eom(df, period=14):
        high = df['high']
        low = df['low']
        vol = df['volume']
        mid = (high + low) / 2
        box_ratio = (high - low) / vol
        one_period_eom = (mid.diff(1) / box_ratio)
        eom = one_period_eom.rolling(window=period).mean()
        df[f'eom_{period}'] = eom
        return df

    def add_bb_bandwidth(df, window=20, std=2):
        df['bb_upper'] = df['close'].rolling(window=window).mean() + std * df['close'].rolling(window=window).std()
        df['bb_lower'] = df['close'].rolling(window=window).mean() - std * df['close'].rolling(window=window).std()
        df[f'bb_bandwidth_{window}'] = (df['bb_upper'] - df['bb_lower']) / df['close'].rolling(window=window).mean()
        df.drop(['bb_upper', 'bb_lower'], axis=1, inplace=True)
        return df

    def add_ema(df, span=20):
        df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
        return df

    def add_ema_ribbon(df, ema_lengths=[20,30,40,50,60], sma_length=1000):
        for ema_length in ema_lengths:
            ema = df['close'].ewm(span=ema_length).mean()
            sma = df['close'].rolling(window=sma_length).mean()
            ema_sma_ratio = (ema / sma)*10 -10
            col_name = f'ema_ribbon_{ema_length}'
            df[col_name] = ema_sma_ratio
        return df

    def add_target(df):
        future_df = pd.DataFrame()
        future_df["future"] = df["close"].shift(-FUTURE_PERIOD_PREDICT)
        df["target"] = list(map(classify, df["close"], future_df["future"]))
        return df
        

    df = pd.read_csv("C:/Users/Lorian/Desktop/Trading_AI/V4/BTC-USD.csv")
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
    df = add_ema_ribbon(df)
    df = add_target(df)

    df = df.dropna()

    df = df.drop(columns=["open","high","low"])
    #print(df.head(5))
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
                sequential_data.append([np.array(prev_days),i[-1]])
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

        sequential_data = buys + sells

        random.shuffle(sequential_data)
        return sequential_data

    def normalize(sequence_data, indicator_names):
        normalized_indicator_names = ["close", "volume","atr_14","roc_12","cci_20","cmo_14","eom_14","bb_bandwidth_20"]
        normalized_indicator_index = []
        for name in normalized_indicator_names:
            index = indicator_names.index(name)
            normalized_indicator_index.append(index)
        normalized_sequence_data = []

        for i,sequence in enumerate(sequence_data): 
            if i % 100000 == 0:
                print(f"{int(i/len(sequence_data)*100)}%")
            seq = sequence[0]
            indicator_list = list(zip(*seq))
            normalized_indicator_list = []
            for i,ind in enumerate(indicator_list):
                if i in normalized_indicator_index:
                    min_val = min(ind)
                    max_val = max(ind)
                    if max_val == min_val:
                        scaling_factor = 100000
                        print("ERROR")
                    else:
                        scaling_factor = 1 / (max_val-min_val)
                    norm_ind =[(x - min_val) * scaling_factor for x in ind]
                    normalized_indicator_list.append(norm_ind)
                else:
                    normalized_indicator_list.append(ind)
            normalized_seq = list(zip(*normalized_indicator_list))
            normalized_sequence_data.append((normalized_seq,sequence[1]))
        
        
        return normalized_sequence_data
    
    sequential_data,indicators = sequentialise(df)
    print(indicators)
    sequential_data = balancing(sequential_data)
    sequential_data = normalize(sequential_data,indicators)
    
    x = []
    y = []
    
    for seq,target in sequential_data:
        x.append(seq)
        y.append(target)
    return np.array(x),np.array(y)

df = get_data()
train_df,valid_df = divide_df(df)
train_x,train_y = preprocessing(train_df)
valid_x,valid_y = preprocessing(valid_df)
print(f"train_data: {len(train_x)} validation:{len(valid_x)} ")

#------------------------------------------------------------------------------
"""

model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2,activation="softmax"))

opt = tf.keras.optimizers.Adam(learning_rate=0.001)



# Festlegen der GPU als Berechnungsressource
with tf.device('/GPU:0'):
  # Kompilieren des Modells
  model.compile(loss="sparse_categorical_crossentropy",optimizer=opt, metrics=["accuracy"])
"""
model = Sequential()

model.add(LSTM(128, input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2,activation="softmax"))

opt = tf.keras.optimizers.Adam(learning_rate=0.001)





# Festlegen der GPU als Berechnungsressource
with tf.device('/GPU:0'):
  # Kompilieren des Modells
  model.compile(loss="sparse_categorical_crossentropy",optimizer=opt, metrics=["accuracy"])




tensorboard = TensorBoard(log_dir=f"log/{NAME}")

filepath = "RNN_Final-{epoch:02d}-{val_accuracy:.3f}"
checkpoint_path = "models/{}.h5".format(filepath)

checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max",
    save_weights_only=False,
    save_format="h5"
)

# Starten des Trainings und Überprüfen der GPU-Auslastung
with tf.device('/GPU:0'):

  # Trainieren des Modells unter Verwendung der LambdaCallback-Funktion zur Überprüfung der GPU-Auslastung
    history = model.fit(train_x,train_y,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_data=(valid_x,valid_y),callbacks=[tensorboard,checkpoint])
