import pickle
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# cargar dataset
data = load_iris()
X = data.data
y = data.target

# crear modelo
model = DecisionTreeClassifier()

# entrenar modelo
model.fit(X, y)

# guardar modelo
pickle.dump(model, open("model.pkl", "wb"))

print("Modelo guardado")