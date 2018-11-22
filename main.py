from readFile import readFile
from bagOfWords import bagOfWords, tenWordsWithCondProb
from naiveBayes import understandData, findTenNonStopWords
import time


"""
files was read 4 times for different operations.

1-For unigram operations, operations are stemming or non-stemming
2-For bigram operations, operations are also stemming or non-stemming
but this time additionally, line tokens was added to the processes.    
"""
def programWorkStation():
    stemlinesOfReal_uni, stemlinesOfFake_uni = readFile("clean_real-Train.txt", "clean_fake-Train.txt", 1, stemed=True)
    stemlinesOfReal_bi, stemlinesOfFake_bi = readFile("clean_real-Train.txt", "clean_fake-Train.txt", 2, stemed=True)
    linesOfReal_uni, linesOfFake_uni = readFile("clean_real-Train.txt", "clean_fake-Train.txt", 1, stemed=False)
    linesOfReal_bi, linesOfFake_bi = readFile("clean_real-Train.txt", "clean_fake-Train.txt", 2, stemed=False)

    print("Please wait till the end of program. It may take some time....\n")
    print("Unigram (with stopWord,with stem):")
    returnValuesOfUnigram1 = bagOfWords(stemlinesOfReal_uni, stemlinesOfFake_uni, 1, stopEnglish="english", stemed=True)
    print("\nUnigram (with stopWord,without stem):")
    returnValuesOfUnigram2 = bagOfWords(linesOfReal_uni, linesOfFake_uni, 1, stopEnglish="english", stemed=False)
    print("\nUnigram (without stopWord, with stem):")
    returnValuesOfUnigram3 = bagOfWords(stemlinesOfReal_uni, stemlinesOfFake_uni, 1, stopEnglish=None, stemed=True)
    print("\nUnigram (without stopWord, without stem):")
    returnValuesOfUnigram4 = bagOfWords(linesOfReal_uni, linesOfFake_uni, 1, stopEnglish=None, stemed=False)

    countOfRealsDict = returnValuesOfUnigram4[0]
    countOfFakesDict = returnValuesOfUnigram4[1]
    tenWordsWithCondProb(countOfRealsDict, countOfFakesDict)

    print("\nStart & End tokens of lines is added to the bigram calculations...")
    print("\nBigram (with stopWord, with stem):")
    returnValuesOfBigram1 = bagOfWords(stemlinesOfReal_bi, stemlinesOfFake_bi, 2, stopEnglish="english", stemed=True)
    print("\nBigram (with stopWord, without stem):")
    returnValuesOfBigram2 = bagOfWords(linesOfReal_bi, linesOfFake_bi, 2, stopEnglish="english", stemed=False)
    print("\nBigram (without stopWord, with stem):")
    returnValuesOfBigram3 = bagOfWords(stemlinesOfReal_bi, stemlinesOfFake_bi, 2, stopEnglish=None, stemed=True)
    print("\nBigram (without stopWord, without stem):")
    returnValuesOfBigram4 = bagOfWords(linesOfReal_bi, linesOfFake_bi, 2, stopEnglish=None, stemed=False)

tstart = time.time()

programWorkStation()

tend = time.time()
print("\nRuntime: " + str(tend-tstart))