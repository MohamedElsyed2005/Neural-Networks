import numpy as np 
import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.losses import CategoricalCrossentropy
import matplotlib.pyplot as plt

# 1 - read the data 
iris = load_iris(as_frame = False)
X = iris.data
y = iris.target.reshape(-1, 1)

# y is of shape (150, 1)
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# 2- Divide the data into training data (50 %), validation data(20 %) and testing data (30 %)
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size = 0.5, stratify = y_encoded, random_state = 42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 0.6, stratify = y_temp, random_state = 42)

# 3- Build keras network 
# input layer = 4,
# dense layer = 10, activation function = Sigmoid 
# dense output layer = 3, activation function = softmax 
model = Sequential()
model.add(Dense(10, input_dim = 4, activation = 'sigmoid'))
model.add(Dense(3, activation='softmax'))

# 4- Compile the model with loss= cross-entropy, optimization = adam  
model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])

# 5- Fit the model using number of epochs = 500.
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, verbose=0)

# 6- Draw a curve to represent the epochs on x axis and the error on the validation 
# sample on y-axis, to determine the epoch number that has the better model.
plt.plot(history.history['val_loss'])
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.title("Validation Loss vs Epochs")
plt.grid(True)
plt.savefig("learning_curve.png", dpi=300)
plt.show()

# Best epoch can be determined by:
best_epoch = np.argmin(history.history['val_loss']) + 1
print(f"Best epoch: {best_epoch}")

# 7- Predicted the y values by using the test data.
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# 8- Compare the actual values with the predicted values and compute accuracy of 
# the model through confusion matrix.
conf_mat = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(conf_mat, display_labels=iris.target_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_plot.png", dpi=300)
plt.show()

accuracy = accuracy_score(y_true, y_pred) # 1.00
print(f"Test Accuracy: {accuracy:.2f}")