from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPooling2D, Convolution2D
# from tensorflow.keras.callbacks import Tensorboard

from utils import load_data

# tensorboard = Tensorboard(logs="logs/")

x_train, y_train = load_data()

model = Sequential()

model.add(Convolution2D(32, (5,5), input_shape=(96,96,1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Convolution2D(128, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Convolution2D(30, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(64, activation="relu"))

model.add(Flatten())
model.add(Dense(30, activation="relu"))

model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=100, batch_size=200, verbose=1, validation_split=0.2)

model.save("glasses_model.h5")