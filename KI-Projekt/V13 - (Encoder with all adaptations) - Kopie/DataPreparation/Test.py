import pandas as pd

# Beispiel DataFrame
df = pd.DataFrame({'ROC': [-1.5, -0.5, 0, 0.5, 1, 1.5]})

# Definiere die Intervallgrenzen, einschließlich aller Werte zwischen -1 und 0.99
bins = [-float("inf"),-1,-0.5,0,0.5,1]
labels = [1,2,3,4,5]

# Wende pd.cut an, um die Intervalle zuzuweisen
df['OneHotEncoded'] = pd.cut(df['ROC'], bins=bins, labels=labels, include_lowest=True)

print(df)