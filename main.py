# -*- coding: utf-8 -*-
"""
Created on Thu May 21 14:35:33 2020

@author: paulv
"""

# Import modules
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# Load the data
data = loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = data[:,:8]
y = data[:,8]

# Create the neural network
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the neural network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the neural network to the data
model.fit(X, y, epochs=150, batch_size=10)

# Evaluate the neural network
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
