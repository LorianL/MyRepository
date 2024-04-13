import pandas as pd
import os


filename = "EURUSD.csv"



script_path = os.path.abspath(__file__)
script_folder = os.path.dirname(script_path)
print(script_folder)






def ticks_to_bars(output_file, timeframe):
    # Lese die CSV-Datei
    df = pd.read_csv(f"{script_folder}\{filename}")

    # Setze die Datums- und Zeitspalten zusammen und konvertiere sie in ein DateTime-Objekt
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    
    # Gruppiere die Daten nach dem gewünschten Zeitrahmen (z.B. '5T' für 5 Minuten)
    grouped = df.groupby(pd.Grouper(key='datetime', freq=timeframe))
    
    # Berechne OHLCV für jede Gruppe
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    ohlc_data = grouped.agg({'bid': ohlc_dict, 'ask': ohlc_dict}).dropna()
    print(ohlc_data)
    # Speichere die Ergebnisse in einer neuen CSV-Datei
    ohlc_data.to_csv(output_file)

# Beispielaufruf des Skripts
ticks_to_bars(f'{filename}_bars.csv', '1T')  # Hier '5T' steht für 5 Minuten