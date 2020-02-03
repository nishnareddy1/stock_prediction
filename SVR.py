from sklearn.metrics import mean_absolute_error
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