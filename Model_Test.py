# Load Dataset and Use Trained Model

import joblib
from sklearn import datasets
from sklearn.model_selection import train_test_split 

# Load Trained Model
model = joblib.load("RandomForest_model.pkl")

# Load New Data
data = datasets.load_digits()
X = data.data
y = data.target

# Predict Dataset
prediction = model.predict(X)
print("predict: ", prediction[:15])
print("truth: ", y[:15])
print("Accuracy: ", model.score(X, y))