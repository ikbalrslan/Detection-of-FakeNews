from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from math import log10



def readFile(trainReal, trainFake):

    with open(trainReal) as fileReal:
        linesOfReal = fileReal.read().splitlines() ##split text file into lines and store in list

    with open(trainFake) as fileFake:
        linesOfFake = fileFake.read().splitlines() ##split text file into lines and store in list
    return linesOfReal,linesOfFake

def bagOfWords(linesOfReal,linesOfFake):
    # create the transform / stop_words="english"
    vectorizerReal = CountVectorizer(lowercase=True, analyzer='word', ngram_range=(1, 1), max_df=1.0, min_df=1,
                                     max_features=None)

    trainReal = linesOfReal[10:]
    testReal = linesOfReal[:10]

    vectorizerReal.fit(trainReal)  ##fit in vector
    indexDictOfWordReal = vectorizerReal.vocabulary_  ##assign index for each word by vocab function
    bag_of_words_real = vectorizerReal.transform(trainReal)
    BoWOfReal = bag_of_words_real.toarray()  ##bag of words array
    uniqlistOfRealWords = vectorizerReal.get_feature_names()  ## create unique list by feature function

    # create the transform / stop_words="english"
    vectorizerFake = CountVectorizer(lowercase=True, analyzer='word', ngram_range=(1, 1), max_df=1.0, min_df=1,
                                     max_features=None)

    trainFake = linesOfFake[10:]
    testFake = linesOfFake[:10]

    vectorizerFake.fit(trainFake)  ##fit in vector
    indexDictOfWordFake = vectorizerFake.vocabulary_  ##assign index for each word by vocab function
    bag_of_words_fake = vectorizerFake.transform(trainFake)
    BoWOfFake = bag_of_words_fake.toarray()  ##bag of words array
    uniqlistOfFakeWords = vectorizerFake.get_feature_names()  ## create unique list by feature function

    #print(trainReal)
    #print(indexDictOfWordReal["trump"]) ##index of trump word in the BoW
    #print(bag_of_words_real.shape)
    #print(type(bag_of_words_real))
    #print(BoWOfReal)
    print()
    #print(trainFake)
    #print(indexDictOfWordFake["trump"]) ##index of trump word in the BoW
    #print(bag_of_words_fake.shape)
    #print(type(bag_of_words_real))
    #print(BoWOfFake)
    #print(uniqlistOfFakeWords) #feature names

    testHeadLines = list(testReal + testFake)

    countOfRealsDict = {}
    countOfFakesDict = {}

    frequenciesOfReal = np.sum(BoWOfReal,axis=0) ## sum counts of the same worlds
    for word in indexDictOfWordReal.keys():
        countOfRealsDict[word] = frequenciesOfReal[indexDictOfWordReal[word]]
    frequenciesOfFake = np.sum(BoWOfFake,axis=0) ## sum counts of the same worlds
    for word in indexDictOfWordFake.keys():
        countOfFakesDict[word] = frequenciesOfFake[indexDictOfWordFake[word]]




    for line in testHeadLines:
        #stop_words="english"
        vectorizerTest = CountVectorizer(lowercase=True, analyzer='word', ngram_range=(1, 1),
                                         max_df=1.0, min_df=1, max_features=None)
        temp = []
        temp.append(line)
        vectorizerTest.fit(temp)  ##fit in vector
        testindexDictOfWord = vectorizerTest.vocabulary_  ##assign index for each word by vocab function
        test_bag_of_words = vectorizerTest.transform(temp)
        test_BoW= test_bag_of_words.toarray()  ##bag of words array
        test_uniqlistOfWords = vectorizerTest.get_feature_names()  ## create unique list by feature function

        naiveBayes(countOfRealsDict, BoWOfReal, countOfFakesDict, BoWOfFake, testindexDictOfWord, test_BoW)

def naiveBayes(countOfRealsDict, BoWOfReal, countOfFakesDict, BoWOfFake, testindexDictOfWord, test_BoW):
    uniqWordsofFiles = list(set(list(countOfRealsDict.keys()) + list(countOfFakesDict.keys()))) ##for bayes calculations
    wordCountofReal = np.sum(list(countOfRealsDict.values()))  ## sum counts of the worlds
    wordCountofFake = np.sum(list(countOfFakesDict.values()))  ## sum counts of the worlds

    ## Priors...
    realRatio = len(BoWOfReal) / (len(BoWOfReal) + len(BoWOfFake))
    fakeRatio = len(BoWOfFake) / (len(BoWOfReal) + len(BoWOfFake))

    ## Conditional Probabilities of real class
    realCondProb = {} ##holds condition probabilities of test headline words
    smoothing = 1 ##smoothing number
    realProbability = log10(realRatio) ## initially
    for word in testindexDictOfWord.keys():
        try:
            realCondProb[word] = (countOfRealsDict[word] + smoothing) / (wordCountofReal + len(uniqWordsofFiles))
        except:
            realCondProb[word] = (0 + smoothing) / (wordCountofReal + len(uniqWordsofFiles))
        ## take test index of word and find count of word in test_BoW then multiply with log of word count
        realProbability += (test_BoW[0][testindexDictOfWord[word]] * log10(realCondProb[word]))

    print("real probability: ", realProbability)

    ## Conditional Probabilities of fake class
    fakeCondProb = {} ##holds condition probabilities of test headline words
    fakeProbability = log10(fakeRatio)  ## initially
    for word in testindexDictOfWord.keys():
        try:
            fakeCondProb[word] = (countOfFakesDict[word] + smoothing) / (wordCountofFake + len(uniqWordsofFiles))
            #print(fakeCondProb[word], word, countOfFakesDict[word])
        except:
            fakeCondProb[word] = (0 + smoothing) / (wordCountofFake + len(uniqWordsofFiles))
        ## take test index of word and find count of word in test_BoW then multiply with log of word count
        fakeProbability += (test_BoW[0][testindexDictOfWord[word]] * log10(fakeCondProb[word]))

    print("fake probability: ", fakeProbability)
    print("--------------------------------------")

    if(realProbability > fakeProbability):
        return "real"
    else:
        return "fake"

