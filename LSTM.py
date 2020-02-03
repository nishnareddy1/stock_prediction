[8:16 PM, 2/2/2020] Soumya Psu: # Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix


# Importing the training set
train_data = pd.read_csv('Google_Stock_Price_Train.csv')
training_data = train_data.iloc[:, 4:5].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_scaled = sc.fit_transform(training_data)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
Y_train = []
for i in range(30, 1258):
    X_train.append(training_scaled[i-30:i, 0])
    Y_train.append(training_scaled[i, 0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X_train.shape)

# Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# Initialising the RNN
model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 40, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 40, return_sequences = True))
model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 40, return_sequences = True))
model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 40))
model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(units = 1))

# Compiling the RNN

model.compile(optimizer= 'adam',loss = 'mean_squared_error',metrics=['mae'])

# Fitting the RNN to the Training set
model.fit(X_train, Y_train, epochs = 100)


# Making the predictions and visualising the results

# Getting the real stock price of 2017
test_data = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_data.iloc[:, 4:5].values


# Getting the predicted stock price of 2017
dataset_total = pd.concat((train_data['Open'], test_data['Open']), axis = 0)
test_input = dataset_total[len(dataset_total) - len(test_data) - 30:].values
test_input = test_input.reshape(-1,1)
test_input = sc.transform(test_input)
X_test = []
for i in range(30, 50):
    X_test.append(test_input[i-30:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
print(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
[8:16 PM, 2/2/2020] Soumya Psu: lstm
[8:16 PM, 2/2/2020] Soumya Psu: from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
d2_train_dataset = X_train.reshape((1228,30*1))
print(d2_train_dataset.shape)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(d2_train_dataset, Y_train)

d_test_dataset = X_test.reshape((20,30*1))
y_pred_svr = svr_rbf.predict(d_test_dataset)
print(real_stock_price.shape)
y_pred_svr = np.reshape(y_pred_svr, (20, 1))
y_pred_svr = sc.inverse_transform(y_pred_svr)

#print(d_test_dataset)
#print(y_pred_svr)
mean_absolute_error(real_stock_price, y_pred_svr, multioutput='raw_values')
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(y_pred_svr, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction using svr')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()