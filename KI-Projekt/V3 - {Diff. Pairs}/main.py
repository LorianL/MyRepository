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
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
#Variabeln
SEQ_LEN = 30
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "LTC-USD"
EPOCHS = 30
BATCH_SIZE = 512
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{RATIO_TO_PREDICT}-PAIR-{int(time.time())}"

#Hyperparameter
lr = 0.001
lstm_size = 512
lstm_dropout = 0.01
dense_size = 96
l2_value = 0.01




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

    print(df.head())

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


file_names_list = os.listdir("V3/crypto_data")
if f"{RATIO_TO_PREDICT}.csv" in file_names_list:
    ratios = [RATIO_TO_PREDICT]
    file_names_list.remove(f"{RATIO_TO_PREDICT}.csv")
    for _ in range(4):
        file_name = random.choice(file_names_list)
        file_names_list.remove(file_name) 
        ratio = file_name.replace(".csv", "")
        ratios.append(ratio)
    print(f"\n\n=> {RATIO_TO_PREDICT}")
    print(ratios)
    print("\n")
else:
    print("Invalid: Ratio_to_Predict")


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
#print(main_df.head())
#-------------------------------------------------------------------------------

times = sorted(main_df.index.values)
last_5pct = times[- int(0.05*len(times))]


validation_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

train_x,train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_df)

print(f"train_data: {len(train_x)} validation:{len(validation_x)} ")
#-------------------------------------------------------------------------------
model = Sequential()

model.add(LSTM(lstm_size, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(lstm_dropout))
model.add(BatchNormalization())

model.add(LSTM(lstm_size, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(lstm_dropout))
model.add(BatchNormalization())

model.add(LSTM(lstm_size, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(lstm_dropout))
model.add(BatchNormalization())

model.add(LSTM(lstm_size, input_shape=(train_x.shape[1:])))
model.add(Dropout(lstm_dropout))
model.add(BatchNormalization())

model.add(Dense(dense_size, activation="relu",  kernel_regularizer=l2(l2_value)))
model.add(Dense(2,activation="softmax"))


try:
    model.load_weights("V3/Transferlearning_model_17_4_2023.h5")
    model.summary()
except:
    print("Fehler beim Laden der Gewichte.")

opt = tf.keras.optimizers.Adagrad(learning_rate=lr)

# Festlegen der GPU als Berechnungsressource
with tf.device('/GPU:0'):
  # Kompilieren des Modells
  model.compile(loss='sparse_categorical_crossentropy',optimizer=opt, metrics=["accuracy"]) #"sparse_categorical_crossentropy"




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
#reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
# Starten des Trainings und Überprüfen der GPU-Auslastung
with tf.device('/GPU:0'):

  # Trainieren des Modells unter Verwendung der LambdaCallback-Funktion zur Überprüfung der GPU-Auslastung
    history = model.fit(train_x,train_y,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_data=(validation_x,validation_y),callbacks=[tensorboard,checkpoint])


model.save_weights('V3/Transferlearning_model_17_4_2023.h5')


