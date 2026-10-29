import numpy as np
from keras.models import Sequential
from keras.layers import Dense

x = np.array([[0,0,0],[0,0,1],[0,1,0],[1,0,0]])
y = np.array([[0],[0],[1],[1]])

model = Sequential()

model.add(Dense(input_shape = (3,), units=4,activation='relu'))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(x, y,epochs=300,batch_size=2)

loss,acc = model.evaluate(x,y)
print(acc*100)

y_pred = model.predict(x)
print(y_pred)
threshold = 0.5
ypclass = (y_pred > threshold).astype(int)
print(ypclass)