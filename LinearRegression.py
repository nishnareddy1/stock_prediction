from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
lr = LinearRegression()
# Train the model
lr.fit(d2_train_dataset, Y_train)
# predicting the values
y_pred_lr = lr.predict(d_test_dataset)
y_pred_lr = np.reshape(y_pred_lr, (20, 1))
y_pred_lr = sc.inverse_transform(y_pred_lr)
# visualizing the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(y_pred_lr, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
# calculating the mean_absolute_error
mae_lr = mean_absolute_error(real_stock_price, y_pred_lr, multioutput='raw_values')
print("mean_absolute_error of lr",mae_lr)