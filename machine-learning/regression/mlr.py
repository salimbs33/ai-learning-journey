import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


df = pd.read_csv('lr/advertising.csv')

print(df.isnull().sum())

# Features
X = df[['TV', 'Radio', 'Newspaper']]

# Target
y = df['Sales']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=15
)

# Model
model = LinearRegression()

# Training
model.fit(X_train, y_train)

# Coefficients
print('b0 =', round(model.intercept_, 2))
print('b1 =', model.coef_)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print('MAE =' , mae)
print('MSE =' , mse)
print('RMSE =', rmse)