import csv
from sklearn.preprocessing import LabelEncoder, StandardScaler

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import keras.optimizers
from keras.utils import plot_model



from sklearn.model_selection import train_test_split
from sets import Set
import numpy as np

output_file = open('nn-results-73-nohttp.csv', 'wb')

#load our training and testing data
train_test_data = np.load('train_test_data-73-nohttp.npz')

X_train = train_test_data['X_train']
print ("X_train: ",X_train.shape)

Y_train = train_test_data['Y_train']
print ("Y_train: ",Y_train.shape)

X_test = train_test_data['X_test']
print ("X_test: ",X_test.shape)

#Y_test = train_test_data['Y_test']
#print ("Y_test: ",Y_test.shape)

#this is our number of features
input_size = 73

#500,300,100
model = Sequential()
model.add(Dense(1000,input_dim=input_size,kernel_initializer="glorot_uniform",activation="sigmoid"))
model.add(Dropout(0.5))
model.add(Dense(600,kernel_initializer="glorot_uniform",activation="sigmoid"))
model.add(Dropout(0.5))
model.add(Dense(200,kernel_initializer="glorot_uniform",activation="sigmoid"))
model.add(Dropout(0.5))

#we have 5 categories
categories = 5

model.add(Dense(categories,kernel_initializer="glorot_uniform",activation="softmax"))
model_optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#model_optimizer = 'rmsprop'

model.compile(loss='categorical_crossentropy',
			  optimizer=model_optimizer,
			  metrics=['accuracy'])

history = model.fit(X_train,Y_train,
		  epochs=11,
		  validation_split=0.10,
		  batch_size=32,
		  verbose=2,
		  shuffle=True)

results = model.predict_classes(X_test, batch_size=64, verbose=0)

output = []
for i in range(len(results)):
	output.append(('{},{}'.format(i,results[i])))

output_file.write('Id,Category\n')
for line in output:
	output_file.write(line + '\n')
output_file.close()

'''scores = model.evaluate(X_test, Y_test, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))'''