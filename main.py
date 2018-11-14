from readFile import readFile, bagOfWords
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


linesOfReal, linesOfFake = readFile("clean_real-Train.txt","clean_fake-Train.txt")
bagOfWords(linesOfReal,linesOfFake)

print("trying new repo...")