import numpy
import csv
import re
from langdetect import detect_langs
from collections import Counter
from random import randint

#Here we shuffle both the first array and the second in the exact same way
#to keep the labelling correct
def shuffle(array_a, array_b):
    state = numpy.random.get_state()
    numpy.random.shuffle(array_a)
    numpy.random.set_state(state)
    numpy.random.shuffle(array_b)

print("READING")
#Here we first read everything in the provided training csv file 
#Then we put these in two list (dataLabel and text_row)
with open("train_set_x.csv",'r',encoding='utf8') as datafile, open("train_set_y.csv",'r',encoding='utf8') as labelfile:
	readDataCSV = csv.reader(datafile,delimiter=',')
	readLabelCSV = csv.reader(labelfile,delimiter=',')
	dataLabel = [];
	text_row = []
	
	for row in readLabelCSV:
		dataLabel.append(row[1]);

	index = 0;
	total_english = 0;
	for row in readDataCSV: #Here we sort the labels into the right array in the Language struct
		text_row.append(row[1])

print("SHUFFLING")		
shuffle(dataLabel,text_row);
data_length = (len(text_row)/100)*70

print("WRITING")
#Here we write everything to the right file keeping 70% of the training data as training
#and 30% as test data.
with open("new_trainfile_x.csv",'w',encoding='utf8',newline='') as newTrainFileX, open("new_trainfile_y.csv",'w',encoding='utf8',newline='') as newTrainFileY,open("new_testfile_x.csv",'w',encoding='utf8',newline='') as newTestFileX, open("new_testfile_y.csv",'w',encoding='utf8',newline='') as newTestFileY:
	XTrainDataCSV = csv.writer(newTrainFileX,delimiter=',')
	YTrainDataCSV = csv.writer(newTrainFileY,delimiter=',')
	XTrainDataCSV.writerow(["Id","Text"]);
	YTrainDataCSV.writerow(["Id","Category"]);
	
	XTestDataCSV = csv.writer(newTestFileX,delimiter=',')
	YTestDataCSV = csv.writer(newTestFileY,delimiter=',')
	XTestDataCSV.writerow(["Id","Text"]);
	YTestDataCSV.writerow(["Id","Category"]);
	test_index = 0;
	for index in range(len(text_row)):
		if(index <= data_length):
			XTrainDataCSV.writerow([index,text_row[index]]);
			YTrainDataCSV.writerow([index,dataLabel[index]]);
		elif(index > data_length ):
			splitted = text_row[index].split();
			no_space = ''.join(splitted);
			only_char = list(no_space);
			numpy.random.shuffle(only_char);
			joined = ' '.join(only_char);
			if(len(joined) > 40):
				XTestDataCSV.writerow([test_index,joined[0:40]]);
				YTestDataCSV.writerow([test_index,dataLabel[index]]);
			else:
				XTestDataCSV.writerow([test_index,joined]);
				YTestDataCSV.writerow([test_index,dataLabel[index]]);
			test_index = test_index+1;
		