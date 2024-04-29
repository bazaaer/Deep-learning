# Installeer TensorFlow in de terminal van VS Code met het commando: pip install tensorflow
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

# Dummy-data genereren (vervang dit door je eigen data)
from sklearn import datasets
iris = datasets.load_iris()

X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Model bouwen
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(4,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

# Model compileren
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Model samenvatting weergeven
model.summary()
# Model trainen
model.fit(X_train, y_train, epochs=5)

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)



# Model evalueren
print("accuracy_score: " + str(accuracy_score(y_test, y_pred)))
