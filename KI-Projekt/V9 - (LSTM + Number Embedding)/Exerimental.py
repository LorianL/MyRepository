import pandas as pd
import os
import torch

"""
script_path = os.path.abspath(__file__)
script_folder = os.path.dirname(script_path)
print(script_folder)


df = pd.read_csv(f"{script_folder}\ModelStats.csv")
print(df)"""


weight_tensor_list = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.tensor([7, 8, 9])]


max_weight = max([torch.max(tensor).item() for tensor in weight_tensor_list])

