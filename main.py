from readFile import readFile
from bagOfWords import bagOfWords
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
    print("Unigram with stopWord and stemming:")
    bagOfWords(stemlinesOfReal_uni, stemlinesOfFake_uni, 1, stopEnglish="english", stemed=True)
    print("\nUnigram with stopWord and non-stemming:")
    bagOfWords(linesOfReal_uni, linesOfFake_uni, 1, stopEnglish="english", stemed=False)
    print("\nUnigram without stopWord and stemming:")
    bagOfWords(stemlinesOfReal_uni, stemlinesOfFake_uni, 1, stopEnglish=None, stemed=True)
    print("\nUnigram without stopWord and non-stemming:")
    bagOfWords(linesOfReal_uni, linesOfFake_uni, 1, stopEnglish=None, stemed=False)

    print("\nStart & End tokens of lines is added to the bigram calculations...")
    print("\nBigram with stopWord and stemming:")
    bagOfWords(stemlinesOfReal_bi, stemlinesOfFake_bi, 2, stopEnglish="english", stemed=True)
    print("\nBigram with stopWord and non-stemming:")
    bagOfWords(linesOfReal_bi, linesOfFake_bi, 2, stopEnglish="english", stemed=False)
    print("\nBigram without stopWord and stemming:")
    bagOfWords(stemlinesOfReal_bi, stemlinesOfFake_bi, 2, stopEnglish=None, stemed=True)
    print("\nBigram without stopWord and non-stemming:")
    bagOfWords(linesOfReal_bi, linesOfFake_bi, 2, stopEnglish=None, stemed=False)

tstart = time.time()

programWorkStation()

tend = time.time()
print("\nRuntime: " + str(tend-tstart))