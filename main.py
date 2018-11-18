from readFile import readFile, bagOfWords, readCSV
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import time


tstart = time.time()

stemlinesOfReal_uni, stemlinesOfFake_uni = readFile("clean_real-Train.txt","clean_fake-Train.txt", 1, stemed = True)
stemlinesOfReal_bi, stemlinesOfFake_bi = readFile("clean_real-Train.txt","clean_fake-Train.txt", 2, stemed = True)
linesOfReal_uni, linesOfFake_uni = readFile("clean_real-Train.txt","clean_fake-Train.txt", 1, stemed = False)
linesOfReal_bi, linesOfFake_bi = readFile("clean_real-Train.txt","clean_fake-Train.txt", 2, stemed = False)

print("Unigram with stopEnglish and stemming:")
bagOfWords(stemlinesOfReal_uni, stemlinesOfFake_uni, 1, stopEnglish = "english", stemed=True)
print("\nUnigram with stopEnglish and non-stemming:")
bagOfWords(linesOfReal_uni, linesOfFake_uni, 1, stopEnglish = "english", stemed=False)
print("\nUnigram with stopNone and stemming:")
bagOfWords(stemlinesOfReal_uni, stemlinesOfFake_uni, 1, stopEnglish = None, stemed=True)
print("\nUnigram with stopNone and non-stemming:")
bagOfWords(linesOfReal_uni, linesOfFake_uni, 1, stopEnglish = None, stemed=False)
print("\nBigram with stopEnglish and stemming:")
bagOfWords(stemlinesOfReal_bi, stemlinesOfFake_bi, 2, stopEnglish = "english", stemed=True)
print("\nBigram with stopEnglish and non-stemming:")
bagOfWords(linesOfReal_bi, linesOfFake_bi, 2, stopEnglish = "english", stemed=False)
print("\nBigram with stopNone and stemming:")
bagOfWords(stemlinesOfReal_bi, stemlinesOfFake_bi, 2, stopEnglish = None, stemed=True)
print("\nBigram with stopNone and non-stemming:")
bagOfWords(linesOfReal_bi, linesOfFake_bi, 2, stopEnglish = None, stemed=False)


tend = time.time()
print("\nRuntime: " + str(tend-tstart))