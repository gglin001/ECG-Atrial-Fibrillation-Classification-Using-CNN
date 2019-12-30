import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Dense, Conv1D, Dropout, MaxPool1D, Flatten
from sklearn.model_selection import train_test_split


train_df = pd.read_pickle('df_train_set.pickle')
test_df = pd.read_pickle('df_test_set.pickle')

X_train = train_df['ecg'].values
X_train = np.vstack(X_train)
y_train = train_df['label'].values
X_test = test_df['ecg'].values
X_test = np.vstack(X_test)
y_test = test_df['label'].values

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


model = Sequential()
model.add(Conv1D(filters=512, kernel_size=32, padding='same',
                 kernel_initializer='normal', activation='relu', input_shape=(512, 1)))
model.add(Conv1D(filters=512, kernel_size=32, padding='same', kernel_initializer='normal', activation='relu'))
# This is the dropout layer. It's main function is to inactivate 20% of neurons in order to prevent overfitting
model.add(Dropout(0.2))
model.add(Conv1D(filters=256, kernel_size=32, padding='same', kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(filters=256, kernel_size=32, padding='same', kernel_initializer='normal', activation='relu'))
# We use MaxPooling with a filter size of 128. This also contributes to generalization
model.add(MaxPool1D(pool_size=128))
model.add(Dropout(0.2))

# The prevous step gices an output of multi dimentional data, which cannot be fead directly into the feed forward neural network. Hence, the model is flattened
model.add(Flatten())
# One hidden layer of 128 neurons have been used in order to have better classification results
model.add(Dense(units=128, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))
# The final neuron HAS to be 1 in number and cannot be more than that. This is because this is a binary classification problem and only 1 neuron is enough to denote the class '1' or '0'
model.add(Dense(units=1, activation='sigmoid'))

model.summary()


optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.5)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X_train, y_train,  batch_size=10, epochs=5, validation_data=(X_test, y_test))


plt.figure(0)
plt.plot(history.history['acc'], 'r', linewidth=3.0, label='Training Accuracy')
plt.plot(history.history['val_acc'], 'b', linewidth=3.0, label='Testing Accuracy')
plt.legend(fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)

plt.figure(1)
plt.plot(history.history['loss'], 'g', linewidth=3.0, label='Training Loss')
plt.plot(history.history['val_loss'], 'y', linewidth=3.0, label='Testing Loss')
plt.legend(fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)

plt.show()
