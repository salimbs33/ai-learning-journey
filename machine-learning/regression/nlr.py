import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

df = pd.read_csv('advertising.csv')

X = df[['TV']]
Y = df['Sales']

polynomial = PolynomialFeatures(degree = 2 , include_bias=False)
Xnew = polynomial.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    Xnew,
    Y,
    test_size=0.2,
    random_state=15)

model = LinearRegression()
model.fit(X_train, y_train)

print('b0 =', round(model.intercept_, 2))
print('b1 =', model.coef_[0])
print('b2 =', model.coef_[1])

Y_pred = model.predict(X_test)

mae = metrics.mean_absolute_error(y_test, Y_pred)
mse = metrics.mean_squared_error(y_test, Y_pred)
rmse = np.sqrt(mse)

print('MAE =' , mae)
print('MSE =' , mse)
print('RMSE =', rmse)