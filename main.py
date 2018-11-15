from readFile import readFile, bagOfWords, readCSV
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


stopEnglish = "english"
stopNone = None
linesOfReal, linesOfFake = readFile("clean_real-Train.txt","clean_fake-Train.txt")
print("Unigram stopEnglish:")
bagOfWords(linesOfReal, linesOfFake, stopEnglish, 1)
print("\nUnigram stopNone:")
bagOfWords(linesOfReal, linesOfFake, stopNone, 1)
print("\nBigram stopEnglish:")
bagOfWords(linesOfReal, linesOfFake, stopEnglish, 2)
print("\nBigram stopNone:")
bagOfWords(linesOfReal, linesOfFake, stopNone, 2)

#print("trying new repo...")

#v = CountVectorizer(ngram_range=(2, 2))
#print(v.fit(["an apple a day keeps the doctor away"]).vocabulary_)

