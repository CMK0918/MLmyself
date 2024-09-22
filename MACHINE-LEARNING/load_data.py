import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Data.csv")
print(dataset)

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 3]
print(x)
print(y)