import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Cargar los datos
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', header=None)
data.columns = ['id', 'clump_thickness', 'uniformity_cell_size', 'uniformity_cell_shape',
                'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei',
                'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']

# Reemplazar los valores faltantes con la media
data = data.replace('?', pd.NA)
data = data.fillna(data.mean())

# Dividir los datos en conjuntos de entrenamiento y prueba
X = data.drop(['id', 'class'], axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Entrenar el modelo
model = GaussianNB()
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')