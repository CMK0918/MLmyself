import numpy as np
import pandas as pd

size = [5,10,12,14,18,30,33,55,65,80,100,150]
distance = [50,20,70,100,200,150,30,50,70,35,40,20]
price = [300,400,450,800,1200,1400,2000,2500,2800,3000,3500,9000]

df = pd.DataFrame({
    "X1":size,
    "X2":distance,
    "y":price

})

print(df)