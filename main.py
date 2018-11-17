from readFile import readFile, bagOfWords, readCSV
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


stemlinesOfReal, stemlinesOfFake = readFile("clean_real-Train.txt","clean_fake-Train.txt", stemed = True)
linesOfReal, linesOfFake = readFile("clean_real-Train.txt","clean_fake-Train.txt", stemed = False)
print("Unigram with stopEnglish and stemming:")
bagOfWords(stemlinesOfReal, stemlinesOfFake, 1, stopEnglish = "english", stemed=True)
print("\nUnigram with stopEnglish and non-stemming:")
bagOfWords(linesOfReal, linesOfFake, 1, stopEnglish = "english", stemed=False)
print("\nUnigram with stopNone and stemming:")
bagOfWords(stemlinesOfReal, stemlinesOfFake, 1, stopEnglish = None, stemed=True)
print("\nUnigram with stopNone and non-stemming:")
bagOfWords(linesOfReal, linesOfFake, 1, stopEnglish = None, stemed=False)
print("\nBigram with stopEnglish and stemming:")
bagOfWords(stemlinesOfReal, stemlinesOfFake, 2, stopEnglish = "english", stemed=True)
print("\nBigram with stopEnglish and non-stemming:")
bagOfWords(linesOfReal, linesOfFake, 2, stopEnglish = "english", stemed=False)
print("\nBigram with stopNone and stemming:")
bagOfWords(stemlinesOfReal, stemlinesOfFake, 2, stopEnglish = None, stemed=True)
print("\nBigram with stopNone and non-stemming:")
bagOfWords(linesOfReal, linesOfFake, 2, stopEnglish = None, stemed=False)


