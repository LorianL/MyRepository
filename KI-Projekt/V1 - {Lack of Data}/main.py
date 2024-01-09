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

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"



def preprocess_df(df):
    df = df.drop("future",axis=1)
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            df[col] = preprocessing.scale(df[col].values)
            df = df.dropna()

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

train_x,train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_df)

print(f"train_data: {len(train_x)} validation:{len(validation_x)} ")
#print(f"Sells: {train_y.count(0)} Buys: {train_y.count(1)}")
#print(f"Sells: {validation_y.count(0)} Buys: {validation_y.count(1)}")
#-------------------------------------------------------------------------------

model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2,activation="softmax"))

opt = tf.keras.optimizers.Adam(learning_rate=0.01)

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


history = model.fit(train_x,train_y,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_data=(validation_x,validation_y),callbacks=[tensorboard,checkpoint])
