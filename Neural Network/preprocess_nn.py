# -*- coding: utf-8 -*-
import csv
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sets import Set
import numpy as np

file_out_noscaling = open('cleaned_x_train.csv', 'wb')
file_out_scaled = open('cleaned_x_train-sc.csv', 'wb')

#define our complete alphabet to count our letters
def get_alphabet():
	en = 'abcdefghijklmnopqrstuvwxyz'
	special_chars =' !?¿¡'
	german = 'äöüß'
	french = 'àâæçéèêëîïôœùûüÿ'
	spanish = 'áéíóúüñ'
	slovak = 'áäčďdzdžéíĺľňóôŕšťúýž'
	polish = 'ąćęłńóśźż'
	empty = ''

	full_alphabet = en + special_chars + german + french + spanish + slovak + polish + empty
	#print(full_alphabet)
	#remove any duplicates by converting to set, and then back to a list
	full_alphabet = list(set(list(full_alphabet.decode('utf-8'))))

	full_alphabet = sorted(full_alphabet);
	return full_alphabet


#simple bag-of-letters counting of chars
def count_chars(text,alphabet):
	alphabet_counts = []
	for letter in alphabet:
		count = text.count(letter)
		alphabet_counts.append(count)
	return alphabet_counts

#	returns a list of sentences, each consisting of a list of characters
def preprocess_x():
	train_file = open('train_set_x.csv', 'rb')
	reader = csv.reader(train_file, delimiter = ',')

	rows = []
	for line in reader:
		rows.append(line)
	del(rows[0])

	for row in rows:
		del(row[0])

	#The following line does a lot.
	#we decode each row into an unicode string
	#each entry becomes a list of characters
	rows = map(lambda x: x[0].lower().decode('utf-8'), rows)
	rows = map(lambda x: re.sub(r'(\s)http\w+','',x), rows)

	
	#print(rows);

	#cleanup emojis, random error symbols
	#get data in the form:
	# 'aace' -> [2,0,1,0,1]
	valid_alphabet = get_alphabet()

	#for each row we will count the chars and write out to a csv 
	x = []
	for i in range(len(rows)):
		bagged = count_chars(rows[i], valid_alphabet)
		file_out_noscaling.write(str(bagged) + '\n')
		x.append(bagged)

	#we want to scale our data for our optimization algorith
	standard_scaler = StandardScaler().fit(x)
	x = standard_scaler.transform(x)

	#print(x)
	return x;
		


#	returns one-hot encoded y data
def preprocess_y():
	train_file = open('train_set_y.csv', 'rb')
	reader = csv.reader(train_file, delimiter = ',')

	rows = []
	for a,b in reader:
		rows.append(b)
	del(rows[0])

	#traansform string to int
	rows = map(lambda x: int(x), rows)

	#one-hot encoding with keras
	y = keras.utils.to_categorical(rows, num_classes=5)

	return y



def preprocess_testdata_x():
	file = open('test_set_x.csv', 'rb')
	reader = csv.reader(file, delimiter = ',')

	entries = []
	for a,b in reader:
		entries.append(b)
	del(entries[0])

	rows = []
	for string in entries:
		s1 = string.lower().decode('utf-8')
		s1 = ''.join(s1.split())
		rows.append(s1)
	
	x = []
	for i in range(len(rows)):
		bagged = count_chars(rows[i], get_alphabet())
		x.append(bagged)

	#we want to scale our data for our optimization algorith
	standard_scaler = StandardScaler().fit(x)
	x_scaled = standard_scaler.transform(x)

	return x_scaled
	#return x



#takes as input a SCALED X, and ONE-HOT encoded Y
def output_train_test_files(X,Y,x_t):
	#seed to randomize
	#seed = 216

	#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=seed)

	#save our train/test data 
	#np.savez_compressed('train_test_data.npz',X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test)
	np.savez_compressed('train_test_data-scaled.npz',X_train=X,Y_train=Y,X_test=x_t)
	print("Data has been saved.")


#output_train_test_files(preprocess_x(),preprocess_y())



output_train_test_files(preprocess_x(),preprocess_y(), preprocess_testdata_x())
#preprocess_testdata_x()

