from nltk.collocations import BigramCollocationFinder
import re
import numpy as np
import string
import csv

#intialized all the counters and empty lists
counter = 0
words_all = []
seq_all_slo = []
seq_all_fre = []
seq_all_spa = []
seq_all_ger = []
seq_all_pol = []

#translate table for punctuations created
translate_table = dict((ord(char), None) for char in string.punctuation)

with open("train_set_x.csv", 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    #data cleaning and bi grams for individual langauges created
    for row in reader:
        if row['Category'] == '0':
            line = row['Text']
            # print(line)
            line = line.lower()  # to lower case
            line = ''.join([i for i in line if not i.isdigit()])  # remove digits
            line = re.sub(r'(\s)http\w+', '', line)
            if len(line) != 0:
                line = line.translate(translate_table)
                # print(line)
                words_all = [line[i:i + 2] for i in range(len(line) - 1)]
                # print(words_all)
                for i in words_all:
                    re.sub(' +', ' ', i)  # replace series of spaces with single space
                # print(words_all)
                seq_all_slo.extend(words_all)
        elif row['Category'] == '1':
            line = row['Text']
            # print(line)
            line = line.lower()  # to lower case
            line = ''.join([i for i in line if not i.isdigit()])  # remove digits

            if len(line) != 0:
                line = line.translate(translate_table)
                # print(line)
                words_all_fre = [line[i:i + 2] for i in range(len(line) - 1)]
                # print(words_all)
                for i in words_all_fre:
                    re.sub(' +', ' ', i)  # replace series of spaces with single space
                # print(words_all)
                seq_all_fre.extend(words_all_fre)
        elif row['Category'] == '2':
            line = row['Text']
            # print(line)
            line = line.lower()  # to lower case
            line = ''.join([i for i in line if not i.isdigit()])  # remove digits

            if len(line) != 0:
                line = line.translate(translate_table)
                # print(line)
                words_all_spa = [line[i:i + 2] for i in range(len(line) - 1)]
                # print(words_all)
                for i in words_all_spa:
                    re.sub(' +', ' ', i)  # replace series of spaces with single space
                # print(words_all)
                seq_all_spa.extend(words_all_spa)
        elif row['Category'] == '3':
            line = row['Text']
            # print(line)
            line = line.lower()  # to lower case
            line = ''.join([i for i in line if not i.isdigit()])  # remove digits

            if len(line) != 0:
                line = line.translate(translate_table)
                # print(line)
                words_all_ger = [line[i:i + 2] for i in range(len(line) - 1)]
                # print(words_all)
                for i in words_all_ger:
                    re.sub(' +', ' ', i)  # replace series of spaces with single space
                # print(words_all)
                seq_all_ger.extend(words_all_ger)
        elif row['Category'] == '4':
            line = row['Text']
            # print(line)
            line = line.lower()  # to lower case
            line = ''.join([i for i in line if not i.isdigit()])  # remove digits

            if len(line) != 0:
                line = line.translate(translate_table)
                # print(line)
                words_all_pol = [line[i:i + 2] for i in range(len(line) - 1)]
                # print(words_all)
                for i in words_all_pol:
                    re.sub(' +', ' ', i)  # replace series of spaces with single space
                # print(words_all)
                seq_all_pol.extend(words_all_pol)

    # extracting the bi-grams and sorting them according to their frequencies
    finder_slo = BigramCollocationFinder.from_words(seq_all_slo)
    finder_slo.apply_freq_filter(5)
    bigram_model_slo = finder_slo.ngram_fd.items()
    bigram_model_slo = sorted(finder_slo.ngram_fd.items(), key=lambda item: item[1], reverse=True)

    finder_fre = BigramCollocationFinder.from_words(seq_all_fre)
    finder_fre.apply_freq_filter(5)
    bigram_model_fre = finder_fre.ngram_fd.items()
    bigram_model_fre = sorted(finder_fre.ngram_fd.items(), key=lambda item: item[1], reverse=True)

    finder_spa = BigramCollocationFinder.from_words(seq_all_spa)
    finder_spa.apply_freq_filter(5)
    bigram_model_spa = finder_spa.ngram_fd.items()
    bigram_model_spa = sorted(finder_spa.ngram_fd.items(), key=lambda item: item[1], reverse=True)

    finder_ger = BigramCollocationFinder.from_words(seq_all_ger)
    finder_ger.apply_freq_filter(5)
    bigram_model_ger = finder_ger.ngram_fd.items()
    bigram_model_ger = sorted(finder_ger.ngram_fd.items(), key=lambda item: item[1], reverse=True)

    finder_pol = BigramCollocationFinder.from_words(seq_all_pol)
    finder_pol.apply_freq_filter(5)
    bigram_model_pol = finder_pol.ngram_fd.items()
    bigram_model_pol = sorted(finder_pol.ngram_fd.items(), key=lambda item: item[1], reverse=True)

    # saving the language models
    np.save("Slovak.npy", bigram_model_slo)
    np.save("French.npy", bigram_model_fre)
    np.save("Spanish.npy", bigram_model_spa)
    np.save("German.npy", bigram_model_ger)
    np.save("Polish.npy", bigram_model_pol)
