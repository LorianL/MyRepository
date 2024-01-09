import numpy as np
normalized_indicator_index = [1]
seq_list =[ [([1,2,3],[1,3,4],[1,5,1]),1] ]
normalized_seq_list = []

for sequence in seq_list: 
    seq = sequence[0]
    indicator_list = list(zip(*seq))
    normalized_indicator_list = []
    for i,ind in enumerate(indicator_list):
        if i in normalized_indicator_index:
            min_val = min(ind)
            max_val = max(ind)
            if max_val == min_val:
                scaling_factor = 0.5
            else:
                scaling_factor = 1 / (max_val-min_val)
            norm_ind =[(x - min_val) * scaling_factor for x in ind]
            normalized_indicator_list.append(norm_ind)
        else:
            normalized_indicator_list.append(ind)
    normalized_seq = list(zip(*normalized_indicator_list))
    normalized_seq_list.append((normalized_seq,sequence[1]))
print(normalized_seq_list)








"""l = [[0,1,2,3],[0,1,2,3]]
print(l)
zip1=list(zip(*l))
print(zip1)

print(list(zip(*zip1)))"""