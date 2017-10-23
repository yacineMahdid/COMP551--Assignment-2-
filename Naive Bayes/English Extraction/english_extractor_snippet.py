
#In this code snipped that insert into the reading module of the
#Naive Bayes algorithm we check if the string is in fact an English utterance
if len(newtext) > 5:
			try:
			#We isolate the main language and its probability
			#If its bigger than a cutoff then this line is removed and we 
			#increment a english count
				lang = detect_langs(newtext);
				first_lang = str(lang[0]);
				sub_lang = first_lang[0:2];
				prob = float(first_lang[3:7]);
				if(sub_lang == 'en' and prob > 0.90):
					print(str(index) + " [" +str(dataLabel[index]) + "] : " + newtext )
					index = index + 1;
					total_english = total_english+1;
					
					if(dataLabel[index] == '0'):
						eng_slo = eng_slo + 1;
					elif(dataLabel[index] == '1'):
						eng_fr = eng_fr + 1;
					elif(dataLabel[index] == '2'):
						eng_spa = eng_spa + 1;
					elif(dataLabel[index] == '3'):
						eng_ger = eng_ger + 1;
					elif(dataLabel[index] == '4'):
						eng_pol = eng_pol + 1;
					continue;
			except:
				print("!!!!!!!!!!!!!!!!!! ERROR : " + newtext + "!!!!!!!!!!!!!!!!!!!!!!")
				
print("Done Writing to output file!")

print("Slovak English: " + str(eng_slo));
print("French English: " + str(eng_fr));
print("Spanish English: " + str(eng_spa));
print("German English: " + str(eng_ger));
print("Polish English: " + str(eng_pol));
print("Total English: " + str(total_english));