from readFile import readFile, bagOfWords
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


linesOfReal, linesOfFake = readFile("clean_real-Train.txt","clean_fake-Train.txt")
print("Unigram:")
bagOfWords(linesOfReal,linesOfFake, 1)
print("\nBigram:")
bagOfWords(linesOfReal,linesOfFake, 2)

#print("trying new repo...")

#v = CountVectorizer(ngram_range=(2, 2))
#print(v.fit(["an apple a day keeps the doctor away"]).vocabulary_)