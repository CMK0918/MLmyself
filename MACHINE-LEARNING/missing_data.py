import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Data.csv")
print(dataset)
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 3]

# 填補缺失資料
dataset["Age"].fillna(np.mean(dataset["Age"]), inplace=True)
dataset["Salary"].fillna(np.mean(dataset["Salary"]), inplace=True)
print(dataset)