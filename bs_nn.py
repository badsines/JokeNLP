import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from time import time
import numpy as np
from pandas import read_json, DataFrame
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# For a single-input model with 2 classes (binary classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=300))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

t0 = time()
with open('jokes.df.pickle', 'rb') as pickle_file:
    df = pickle.load(pickle_file)
print("done in %0.3fs" % (time() - t0))

print('Flattening {} d2v lists ...'.format(df.d2v.values.shape[0]))
X = np.reshape([r for r in df.d2v.values], (df.d2v.values.shape[0], 300))
#y = pd.cut(df.score, [-1, 15, 50000], labels=['MEH', 'GOOD'])
y = pd.cut(df.score, [-1, 15, 50000], labels=[0, 1])
print(df.score.describe())
print("done in %0.3fs" % (time() - t0))

print('Splitting into a training and testing set ...')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
print("done in %0.3fs" % (time() - t0))

# Train the model, iterating on the data in batches of 32 samples
model.fit(X_train, y_train, epochs=10, batch_size=32)
score_test = model.evaluate(X_test, y_test, batch_size=32)
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)
print ("Train Results\n")
print(confusion_matrix(y_train, y_pred_train))
#print(classification_report(y_train, y_pred_train))
print ("Test Results\n")
print(classification_report(y_test, y_pred_test))
print(confusion_matrix(y_test, y_pred_test))

score_train = model.evaluate(X_train, y_train, batch_size=32)
