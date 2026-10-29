import keras
import numpy as np

from keras.models import Sequential
from keras.layers import Input, Dense

model = Sequential()

model.add(Input(shape=(1,)))
model.add(Dense(units=1, activation='linear'))

model.compile(
    optimizer='sgd',
    loss='mean_absolute_error'
)

x_train = np.array([1, 2, 3, 4])
y_train = np.array([5, 6, 7, 8])

epochs = 1000
batch = 2

model.fit(
    x=x_train,
    y=y_train,
    epochs=epochs,
    batch_size=batch
)

w, b = model.get_weights()

print(w)
print(b)