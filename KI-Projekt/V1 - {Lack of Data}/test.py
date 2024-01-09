import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("GPU gefunden. Konfigurieren...")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("Keine GPU gefunden. Das Modell wird auf der CPU trainiert.")



import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
#print(tf.test.is_built_with_cuda())


"""
import torch
print(torch.cuda.is_available())

print("-----------------------------")
print(torch.backends.cudnn.version())"""

"""import torch

# Prüfen, ob CUDA verfügbar ist und gegebenenfalls aktivieren
if torch.cuda.is_available():
    device = torch.device("cuda")          # CUDA-Gerätobjekt
    print("CUDA wird verwendet")
else:
    device = torch.device("cpu")           # CPU-Gerätobjekt
    print("CUDA ist nicht verfügbar, CPU wird verwendet")"""