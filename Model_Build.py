# Train RandomForest Model For 8*8 digits Datasets

from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import joblib

# Load Data
data = datasets.load_digits()
X = data.data
y = data.target

# Split Datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Model Build
model = RandomForestClassifier(max_features=0.1,
                               min_samples_leaf=1,
                               min_samples_split=2,
                               max_depth=10,
                               n_estimators=136,
                               criterion="entropy",
                               random_state=0,
                               n_jobs=-1)
model.fit(X_train,y_train)

# Model Evaluate
cross_score = cross_val_score(model, X_train, y_train, cv=10).mean()
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
y_pred = model.predict(X_test)

print("train score: ", train_score)
print("test score: ", test_score)
print("5 fold cross score: ", cross_score)
print("predict: ", y_pred[:15])
print("truth: ", y_test[:15])

# Model Save
joblib.dump(model, 'RandomForest_model.pkl')