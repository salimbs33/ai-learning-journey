import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


df = pd.read_csv('lr/advertising.csv')

# التأكد من عدم وجود قيم مفقودة
print(df.isnull().sum())

# Feature و Target
X = df[['TV']]
y = df['Sales']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=15
)

# إنشاء النموذج
model = LinearRegression()

# تدريب النموذج
model.fit(X_train, y_train)

# معاملات النموذج
print('b0 =', round(model.intercept_, 2))
print('b1 =', round(model.coef_[0], 2))

# التنبؤ
y_pred = model.predict(X_test)

# تقييم النموذج
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print('MAE =', mae)
print('MSE =', mse)
print('RMSE =', rmse)