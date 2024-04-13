import random
import torch
from torch.nn import Embedding

"""n_embeddings, dim = 10,4

emb_1 = Embedding(n_embeddings, dim)


#print(emb_1.weight)

#inp = torch.LongTensor([[1,3], [5,5]])
inp2 = torch.LongTensor([[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]])
print(inp2)
print(emb_1(inp2))


emb_2 = Embedding(n_embeddings,dim, padding_idx=5)
print(emb_2.weight)"""
"""
import torch
import torch.nn as nn

# Beispiel One-Hot-codierter Tensor
one_hot_tensor = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Größe des Vokabulars und Dimension des Embedding-Vektors festlegen
vocab_size = 3  # Anzahl der eindeutigen Kategorien
embedding_dim = 4  # Dimension des Embedding-Vektors

# Embedding-Layer erstellen
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# One-Hot-codierten Tensor in das Embedding-Layer übergeben
embedded_vector = embedding_layer(one_hot_tensor)

# Ergebnisse anzeigen
print("One-Hot-codierter Tensor:")
print(one_hot_tensor)

print("\nEmbedding-Vektor:")
print(embedded_vector)"""


