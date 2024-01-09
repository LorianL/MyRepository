import random
import torch


t = torch.ones(3,4,12)

print(t)



print(t[:,1,:])

dropout_features_prob = 0.5

def dropout_features(input_tensor):
    batchsize = input_tensor.shape[0]
    sequence_length = input_tensor.shape[1]
    features = input_tensor.shape[2]
    
    t0 = torch.zeros(t.shape[0],t.shape[1])
    #Liste mit allen Möglichen Feature indexen
    all_features = list(range(features))
    #Entnimmt zufällig die features
    selected_features = random.sample(all_features, int(features*dropout_features_prob))
    
    for f in selected_features:
        input_tensor[:,:,f] = t0

    return input_tensor



print(dropout_features(t))


