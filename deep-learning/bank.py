import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense 

np.random.seed(42)
tf.random.set_seed(42)

#قراءة البيانات و القاء نظرة سريعة
df = pd.read_csv('datasets/Churn_Modelling.csv')
print(df.columns)
print(df.info)
value_counts = df['Exited'].value_counts()
print(value_counts)

#تصحيح العمود exited
input_cul = df.drop('Exited',axis=1)
class_cul = df['Exited']

RS = RandomOverSampler(random_state=0)

input_cul_R,class_cul_R = RS.fit_resample(input_cul,class_cul)
df_blansed = pd.concat([input_cul_R,class_cul_R],axis=1)

print(df_blansed['Exited'].value_counts())


#تقسيم البيانات
X = df_blansed.iloc[:,3:13]
Y = df_blansed.iloc[:,13]

print(X.columns)

LIGender = LabelEncoder()
X["Gender"] = LIGender.fit_transform(X["Gender"])
print(X["Gender"])

Gougrafy_unics = pd.unique(X['Geography'])
print(Gougrafy_unics)
#['France', 'Spain', 'Germany']

print(type(X))
ct = ColumnTransformer(transformers=[
    ('encoder',OneHotEncoder(),['Geography'])
    ],remainder='passthrough'
    )
X = ct.fit_transform(X)
print(type(X))



x_train,x_test,y_train,y_test = train_test_split(X,Y,random_state=0)


sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


model = Sequential()

input_dim = len(x_train[0])

model.add(Dense(6,activation='relu',input_dim=input_dim))
model.add(Dense(6,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(x_train,y_train,batch_size= 10,epochs=20)

