import pandas as pd
import os
from sklearn import preprocessing
from collections import deque
import numpy as np
import random
import time



SEQ_LEN = 50
FUTURE_PERIOD_PREDICT = 30
EPOCHS = 15
BATCH_SIZE = 16
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"



def get_data():
 #-------------------------------------------------------   


    def z_score_normalize(df_series):
        # Berechne den Mittelwert
        mean = df_series.mean()
        
        # Berechne die Standardabweichung
        std_dev = df_series.std()
        
        # Wende die Z-Score-Normalisierung auf die Daten an
        normalized_data = (df_series - mean) / std_dev
        
        return pd.Series(normalized_data)

    def classify(current, future):
        if float(future) > float(current):
            return 1
        else:
            return 0
#------------------------------------------------

#none        
    def add_rsi(df, period=14, norm_fn = z_score_normalize):
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 1 - (1 / (1 + rs))
        rsi = norm_fn(rsi)
        #rsi = min_max_normalize(rsi)
        df[f'rsi_{period}'] = rsi
        return df
#winsorize
    def add_atr(df, period=14, norm_fn = z_score_normalize):
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        atr = norm_fn(atr)
        #atr = min_max_normalize(atr)
        df[f'atr_{period}'] = atr
        return df
#none
    def add_mfi(df, period=14, norm_fn = z_score_normalize):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        positive_flow = raw_money_flow.where(df['close'] > df['close'].shift(), 0)
        negative_flow = raw_money_flow.where(df['close'] < df['close'].shift(), 0)
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        money_ratio = positive_mf / negative_mf
        mfi = 1 - (1 / (1 + money_ratio))
        mfi = norm_fn(mfi)
        #mfi = min_max_normalize(mfi)
        df[f'mfi_{period}'] = mfi
        return df
#winsorize
    def add_cmf(df, period=20, norm_fn = z_score_normalize):
        mfm = ((2 * df['close'] - df['low'] - df['high']) / (df['high'] - df['low'])) * df['volume']
        cmf = mfm.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        cmf = norm_fn(cmf)
        #cmf = min_max_normalize(cmf)
        df[f"cmf_{period}"] = cmf
        return df
#min_max_normalize
    def add_roc(df, period=12, norm_fn = z_score_normalize):
        roc = (df['close'] - df['close'].shift(periods=period)) / df['close'].shift(periods=period)
        roc = norm_fn(roc)
        #roc = min_max_normalize(roc)
        df[f'roc_{period}'] = roc
        return df
#none
    def add_vo(df, short_period=14, long_period=28, norm_fn = z_score_normalize):
        short_ma = df['volume'].rolling(window=short_period).mean()
        long_ma = df['volume'].rolling(window=long_period).mean()
        vo = (short_ma - long_ma) / long_ma
        vo = norm_fn(vo)
        #vo = min_max_normalize(vo)
        df[f'vo_{short_period}_{long_period}'] = vo
        return df
#z_score_normalize
    def add_cci(df, period=20, norm_fn = z_score_normalize):
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = np.abs(tp - sma).rolling(window=period).mean()
        cci = (tp - sma) / (0.015 * mad)
        cci = norm_fn(cci)
        #cci = min_max_normalize(cci)
        df[f'cci_{period}'] = cci
        return df
#none
    def add_cmo(df, period=14, norm_fn = z_score_normalize):
        diff = df['close'].diff()
        pos_sum = diff.where(diff > 0, 0).rolling(window=period).sum()
        neg_sum = np.abs(diff.where(diff < 0, 0)).rolling(window=period).sum()
        cmo = ((pos_sum - neg_sum) / (pos_sum + neg_sum))
        cmo = norm_fn(cmo)
        #cmo = min_max_normalize(cmo)
        df[f'cmo_{period}'] = cmo
        return df
#Sma + winsorize
    def add_eom(df, period=14, norm_fn = z_score_normalize):
        high = df['high']
        low = df['low']
        vol = df['volume']
        mid = (high + low) / 2
        box_ratio = (high - low) / vol
        one_period_eom = (mid.diff(1) / box_ratio)
        eom = one_period_eom.rolling(window=period).mean()
        #eom = sma_normalize(eom)
        eom = norm_fn(eom)
        #eom = min_max_normalize(eom)
        df[f'eom_{period}'] = eom
        return df
#min_max_normalize
    def add_bb_bandwidth(df, window=20, std=2 , norm_fn = z_score_normalize):
        df['bb_upper'] = df['close'].rolling(window=window).mean() + std * df['close'].rolling(window=window).std()
        df['bb_lower'] = df['close'].rolling(window=window).mean() - std * df['close'].rolling(window=window).std()
        bb_bandwidth = (df['bb_upper'] - df['bb_lower']) / df['close'].rolling(window=window).mean()
        bb_bandwidth = norm_fn(bb_bandwidth)
        #bb_bandwidth = min_max_normalize(bb_bandwidth)
        df[f'bb_bandwidth_{window}'] = bb_bandwidth
        df.drop(['bb_upper', 'bb_lower'], axis=1, inplace=True)
        return df
#sma + z_score_normalize
    def add_ema(df, span=20, norm_fn = z_score_normalize):
        ema = df['close'].ewm(span=span).mean()
        #ema = sma_normalize(ema)
        ema = norm_fn(ema)
        #ema = min_max_normalize(ema)
        col_name = f'ema_{span}'
        df[col_name] = ema
        
        return df

    def add_new_target(df):
        def difference_target(current, future):
            return (future / current)*100 - 100
        def classify_mod(difference,lim):
            if float(difference) > float(lim):
                return 1
            else:
                return 0
        temp_df = pd.DataFrame()
        temp_df["future"] = df["close"].shift(-FUTURE_PERIOD_PREDICT)
        temp_df["diff"] = list(map(difference_target, df["close"], temp_df["future"]))
        #Berechnung des Grenzwertes:
        data = temp_df['diff'].tolist()
        data.sort()
        lim = data[int(0.70*len(data))]
        print(f"lim: {round(lim,4)}")
        #---
        df["target"] = list(map(lambda x: classify_mod(x, lim), temp_df["diff"]))
        return df
        

    df = pd.read_csv("C:/Users/Lorian/Desktop/Trading_AI/V4 - {Normalisation, Indicators}/BTC-USD.csv")
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
    df = add_ema(df)
    df = add_new_target(df)

    df["close"] = z_score_normalize(df["close"])
    df["volume"] = z_score_normalize(df["volume"])


    df = df.drop(columns=["open","high","low"])
    df = df.dropna()

    print(df.head(5))
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

    
    sequential_data,indicators = sequentialise(df)
    print(indicators)
    sequential_data = balancing(sequential_data)
    print(sequential_data[0])
    
    return np.array(sequential_data)

df = get_data()
train_df,valid_df = divide_df(df)
train_data = preprocessing(train_df)
valid_data = preprocessing(valid_df)
print(f"train_data: {len(train_data)} validation:{len(valid_data)} ")

#------------------------------------------------------------------------------