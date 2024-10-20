from sklearn import datasets
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# load data
df = pd.read_csv("boston_house_prices.csv")
X = (df.iloc[:, :13]).astype("float32")
y = (df.iloc[:, 13]).astype("float32")

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# model train
RF = RandomForestRegressor(random_state=0)
RF.fit(X_train, y_train)

def train_accuracy():
    print("train score: ", RF.score(X_train, y_train))
    
def test_accuracy():
    print("test score: ", RF.score(X_test, y_test))

def cross_accuracy():
    print("10 fold score: ", cross_val_score(RF, X_train, y_train, cv=10).mean())

def prediction():
    y_pred = RF.predict(X_test)
    print("prediction: ", y_pred)
