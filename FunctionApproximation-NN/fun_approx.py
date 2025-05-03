import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 1- Generate 1000 samples for each variable [-1..1]
n_samples = 1000
X = np.random.uniform(-1, 1, (n_samples, 4))

# Function: y = sin(2πx1) * x2 * x3 * x4 * e^-(x1+x2+x3+x4)
x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
y = np.sin(2 * np.pi * x1) * x2 * x3 * x4 * np.exp(-(x1 + x2 + x3 + x4))
y = y.reshape(-1, 1)

#2- Divide the above data into training data (70%),
#  validation data (15%), and testing data (15%). 
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 3- Build keras network (input layer = 4 ,
# dense layer (number of neuron = 50,  activation function = sigmoid),
# dense output layer (number of neuron =1, activation function = linear).

model = Sequential()
model.add(Dense(50, input_dim=4, activation='sigmoid'))  # Hidden layer
model.add(Dense(1, activation='linear'))  # Output layer

# 4- Compile the model with loss= mean square error, optimization = adam 
model.compile(optimizer=Adam(), loss='mean_squared_error')

# 5- Fit the model using number of epochs = 500. 
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, verbose=0)

# 6- Draw a curve to represent the epochs on x axis and the loss on the validation 
# sample on y-axis to determine the epoch number that has the better model 
plt.plot(history.history['val_loss'])
plt.xlabel("Epochs")
plt.ylabel("Validation Loss (MSE)")
plt.title("Validation Loss vs Epochs")
plt.grid(True)
plt.savefig("val_loss_curve.png", dpi = 300)
plt.show()
best_epoch = np.argmin(history.history['val_loss']) + 1
print(f"Best epoch (lowest validation loss): {best_epoch}")

# 7- Predicted the y values by using the test data.
y_pred = model.predict(X_test)

# 8- Compare the actual values with the predicted values and compute root mean square error.  
rmse = np.sqrt(mean_squared_error(y_test, y_pred))# 0.4029
print(f"Root Mean Square Error (RMSE) on test data: {rmse:.4f}")