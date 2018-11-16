from readFile import readFile, bagOfWords, readCSV
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize



stopEnglish = "english"
stopNone = None
linesOfReal, linesOfFake = readFile("clean_real-Train.txt","clean_fake-Train.txt", stemed = True)
print("Unigram stopEnglish:")
bagOfWords(linesOfReal, linesOfFake, stopEnglish, 1, stemed=True)
print("\nUnigram stopNone:")
bagOfWords(linesOfReal, linesOfFake, stopNone, 1, stemed=True)
#print("\nBigram stopEnglish:")
#bagOfWords(linesOfReal, linesOfFake, stopEnglish, 2, stemed=True)
print("\nBigram stopNone:")
bagOfWords(linesOfReal, linesOfFake, stopNone, 2, stemed=True)


