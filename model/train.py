import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import csv
import ast
import tensorflowjs as tfjs
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, Reshape, Bidirectional
from keras.layers import LSTM
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib 

saveModel = True

def create_dataset():
    X_values = []
    y_values = []
    with open('data.csv') as csvfile:
        rows = csv.reader(csvfile, delimiter=';')
        # Skip the header
        next(rows, None)
        maxSequenceLength = 0
        for row in rows:
            y_values.append(row[0])
            sequence = ast.literal_eval(row[1])
            if(len(sequence) > maxSequenceLength):
                maxSequenceLength = len(sequence)
            X_values.append(sequence)

        for x_value in X_values:
            for i in range(len(x_value), maxSequenceLength):
                x_value.append([0.0,0.0])

    return X_values, y_values, maxSequenceLength

X_values, y_values, maxSequenceLength = create_dataset()

print("maxSequenceLength: " + str(maxSequenceLength))

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_values)
y_encoded = encoder.transform(y_values)
# convert integers to dummy variables (i.e. one hot encoded)
y_values = np_utils.to_categorical(y_encoded)

np.random.seed(7)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_values, y_values, test_size=0.33)

# Reshape data
X_train = np.reshape(X_train, (np.array(X_train).shape[0], np.array(X_train).shape[1], 2))
X_test = np.reshape(X_test, (np.array(X_test).shape[0], np.array(X_test).shape[1], 2))

# Create model
#model = Sequential()
#model.add(LSTM(100))
#model.add(Dense(3, activation='softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
#history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20)

model = Sequential()
model.add(Bidirectional(LSTM(20, return_sequences=True)))
model.add(Bidirectional(LSTM(20)))
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=35)

training_loss = history.history['loss']
test_loss = history.history['val_loss']
#training_accuracy = history.history['acc']
#test_accuracy = history.history['val_acc']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

if(saveModel):
    tfjs.converters.save_keras_model(model, 'models/shapedetection_model')
    #model.save('models/shapedetection_model.h5')

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
