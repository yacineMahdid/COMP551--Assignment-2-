# -*- coding: utf-8 -*-

#Labels goes as follow:
#0 = Slovak, 1 = French, 2 = Spanish, 3 = German and 4 = Polish
#Training examples = 276 517
#Testing Examples = 118 508

#Each row of training = first 10 word of an utterance, no punctuation
from __future__ import print_function
import csv
import re
from langdetect import detect_langs
from collections import Counter
from random import randint

#Calculate the probability that this string of characters belong to a particular language
def calculateProb(charArray,probList,allChar,prob,total_label):
	total = 1;
	for char in charArray:
		if char in probList:
			total = total*probList[char]; #Simply multiply all characters probabilities
		else: #This part is useless but was left in for security reason
			if char in allChar:
				total = total*(1);
			else:
				total = total*1;
	return total*prob; #At the end multiply by the language probability
	
	#This function calculate the probability for each languages and return the most likely
def calculateMostLikely(charArray,probabilityStruct,allChar,fr_label,slo_label,ger_label,spa_label,pol_label,total_label):
	probFra = calculateProb(charArray,probabilityStruct.probFrench,allChar,fr_label/total_label,total_label)
	probSlo = calculateProb(charArray,probabilityStruct.probSlovak,allChar,slo_label/total_label,total_label)
	probSpa = calculateProb(charArray,probabilityStruct.probSpanish,allChar,spa_label/total_label,total_label)
	probGer = calculateProb(charArray,probabilityStruct.probGerman,allChar,ger_label/total_label,total_label)
	probPol = calculateProb(charArray,probabilityStruct.probPolish,allChar,pol_label/total_label,total_label)

	if(probFra > probSlo and probFra > probSpa and probFra > probGer and probFra > probPol):
		return 1;
	elif(probSlo > probFra and probSlo > probSpa and probSlo > probGer and probSlo > probPol):
		return 0;
	elif(probSpa > probSlo and probSpa > probFra and probSpa > probGer and probSpa > probPol):
		return 2;
	elif(probGer > probSlo and probGer > probSpa and probGer > probFra and probGer > probPol):
		return 3;
	elif(probPol > probSlo and probPol > probSpa and probPol > probGer and probPol > probFra):
		return 4;
	else:
		return -1;
	
	
	#Initiating structure and variable
class dataStruct:
	def __init__(self):
		self.slovak = [];
		self.french = [];
		self.spanish = [];
		self.german = [];
		self.polish = [];
		self.total = [];
		self.all = set();

class probStruct:
	def __init__(self):
		self.probSlovak = {};
		self.probFrench = {};
		self.probSpanish = {};
		self.probGerman = {};
		self.probPolish = {};

languages = dataStruct();
languagesProb = probStruct();

fr_tot = 0;
slo_tot = 0;
spa_tot = 0;
ger_tot = 0;
pol_tot = 0;

fr_label = 0;
slo_label = 0;
spa_label = 0;
ger_label = 0;
pol_label = 0;
total_label = 0;

#Reading training file and parsing the training instances into languages
#
with open("train_set_x.csv",'r',encoding='utf8') as datafile, open("train_set_y.csv",'r',encoding='utf8') as labelfile:
	readDataCSV = csv.reader(datafile,delimiter=',')
	readLabelCSV = csv.reader(labelfile,delimiter=',')
	dataLabel = [];
	for row in readLabelCSV:
		dataLabel.append(row[1]);

	index = 0;
	total_english = 0;
	for row in readDataCSV: #Here we sort the labels into the right array in the Language struct
		text = row[1].lower(); #Set the string to lowercase 
		newtext = re.sub(r'(\s)http\w+','',text);#remove the links
		
		if(dataLabel[index] == '0'):
			slo_label = slo_label + 1;
		elif(dataLabel[index] == '1'):
			fr_label = fr_label + 1;
		elif(dataLabel[index] == '2'):
			spa_label = spa_label + 1;
		elif(dataLabel[index] == '3'):
			ger_label = ger_label + 1;
		elif(dataLabel[index] == '4'):
			pol_label = pol_label + 1;

		total_label = total_label + 1;
		#Now should split each text string into token and store them
		splittedtext = newtext.split();
		for token in splittedtext:
			for char in list(token): #Store all the characters
				if(dataLabel[index] == '0'):
					languages.slovak.append(char);
					slo_tot = slo_tot + 1;
				elif(dataLabel[index] == '1'):
					languages.french.append(char);
					fr_tot = fr_tot + 1;
				elif(dataLabel[index] == '2'):
					languages.spanish.append(char);
					spa_tot = spa_tot + 1;
				elif(dataLabel[index] == '3'):
					languages.german.append(char);
					ger_tot = ger_tot + 1;
				elif(dataLabel[index] == '4'):
					languages.polish.append(char);
					pol_tot = pol_tot + 1;
				languages.all.add(char);
				languages.total.append(char);
		index = index + 1;


#Count the character
charCountFrench = Counter(languages.french);
charCountSlovak = Counter(languages.slovak);
charCountSpanish = Counter(languages.spanish);
charCountGerman = Counter(languages.german);
charCountPolish = Counter(languages.polish);
charCountTotal = Counter(languages.total);

charFrequenciesFrench = {};
charFrequenciesSlovak = {};
charFrequenciesSpanish = {};
charFrequenciesGerman = {};
charFrequenciesPolish = {};
charFrequenciesTotal = {}

total_tot = fr_tot + slo_tot + spa_tot + ger_tot + pol_tot;

#Calculate the character frequencies which are used as probabilities
for char in languages.all:
	if not (char in charCountFrench):
		charFrequenciesFrench[char] = 1/fr_tot;
	else:
		charFrequenciesFrench[char] = charCountFrench[char] / fr_tot;
	
	if not (char in charCountSlovak):
		charFrequenciesSlovak[char] = 1/slo_tot;
	else:
		charFrequenciesSlovak[char] = charCountSlovak[char] / slo_tot;
	
	if not (char in charCountSpanish):
		charFrequenciesSpanish[char] = 1/spa_tot;
	else:
		charFrequenciesSpanish[char] = charCountSpanish[char] / spa_tot;
	
	if not (char in charCountGerman):
		charFrequenciesGerman[char] = 1/ger_tot;
	else:
		charFrequenciesGerman[char] = charCountGerman[char] / ger_tot;
		
	if not (char in charCountPolish):
		charFrequenciesPolish[char] = 1/pol_tot;
	else:
		charFrequenciesPolish[char] = charCountPolish[char] / pol_tot;
	
	charFrequenciesTotal[char] = charFrequenciesFrench[char] + charFrequenciesSlovak[char] + charFrequenciesSpanish[char] + charFrequenciesGerman[char] + charFrequenciesPolish[char];
	
	languagesProb.probSlovak[char] = charFrequenciesSlovak[char];
	languagesProb.probFrench[char] = charFrequenciesFrench[char];
	languagesProb.probSpanish[char] = charFrequenciesSpanish[char];
	languagesProb.probGerman[char] = charFrequenciesGerman[char];
	languagesProb.probPolish[char] = charFrequenciesPolish[char];
	
	#Read the test file and write to a guess file to submit to Kaggle
with open("test_set_x.csv",'r',encoding='utf8') as testfile, open("Naive_Bayes_Guess.csv",'w',encoding='utf8',newline='') as guessfile:
	testDataCSV = csv.reader(testfile,delimiter=',')
	guessDataCSV = csv.writer(guessfile,delimiter=',')
	
	slo = 0;
	fr = 0;
	spa = 0;
	germ = 0;
	pol = 0;
	ran = 0;
	
	index = -1;
	for row in testDataCSV:
		if (index == -1):
			guessDataCSV.writerow(["Id","Category"]);
			index = index + 1;
			continue;
			
			#Here we make our guess
		result = calculateMostLikely(row[1].split(),languagesProb,languages.all,fr_label,slo_label,ger_label,spa_label,pol_label,total_label);
		
		if(result == 0):
			slo = slo + 1;
		elif(result == 1):
			fr = fr + 1;
		elif(result == 2):
			spa = spa +  1;
		elif(result == 3):
			germ = germ + 1;
		elif(result == 4):
			pol = pol+1;
		else:
			ran = ran + 1;
			result = randint(0,4);
		
		guessDataCSV.writerow([str(index),str(result)]);
		index = index + 1;
		
print("Slovak: " + str(slo) + " and Train: " + str(slo_label));
print("French: " + str(fr) + " and Train: " + str(fr_label));
print("Spanish: " + str(spa) + " and Train: " + str(spa_label));
print("German: " + str(germ) + " and Train: " + str(ger_label));
print("Polish: " + str(pol) + " and Train: " + str(pol_label));
print("Random Guesses: " + str(ran));

		
