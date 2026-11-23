from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pickle

X, y = load_iris(return_X_y=True)
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained: model/model.pkl")

