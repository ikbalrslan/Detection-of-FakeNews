from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from readFile import readCSV
from naiveBayes import naiveBayes,calculationofAccuracy



def bagOfWords(linesOfReal,linesOfFake, ngram, stopEnglish, stemed): ## ngram_range=(1, 1) for unigram, ngram_range=(2, 2) for bigram
    # create the transform / stop_words="english"
    vectorizerReal = CountVectorizer(lowercase=True, stop_words=stopEnglish, analyzer='word', ngram_range=(ngram, ngram), max_df=1.0, min_df=1,
                                     max_features=None)

    trainReal = linesOfReal[:]


    vectorizerReal.fit(trainReal)  ##fit in vector
    indexDictOfWordReal = vectorizerReal.vocabulary_  ##assign index for each word by vocab function
    bag_of_words_real = vectorizerReal.transform(trainReal)
    BoWOfReal = bag_of_words_real.toarray()  ##bag of words array
    uniqlistOfRealWords = vectorizerReal.get_feature_names()  ## create unique list by feature function
    #print("uniq real: ",len(uniqlistOfRealWords))
    #print("uniq real: ",uniqlistOfRealWords)

    # create the transform / stop_words="english"
    vectorizerFake = CountVectorizer(lowercase=True, stop_words=stopEnglish, analyzer='word', ngram_range=(ngram, ngram), max_df=1.0, min_df=1,
                                     max_features=None)

    trainFake = linesOfFake[:]

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
    #print()
    #print(trainFake)
    #print(indexDictOfWordFake["trump"]) ##index of trump word in the BoW
    #print(bag_of_words_fake.shape)
    #print(type(bag_of_words_real))
    #print(BoWOfFake)
    #print("uniq fake: ",len(uniqlistOfFakeWords)) #feature names

    countOfRealsDict = {}
    countOfFakesDict = {}

    frequenciesOfReal = np.sum(BoWOfReal,axis=0) ## sum counts of the same worlds
    for word in indexDictOfWordReal.keys():
        countOfRealsDict[word] = frequenciesOfReal[indexDictOfWordReal[word]]
    frequenciesOfFake = np.sum(BoWOfFake,axis=0) ## sum counts of the same worlds
    for word in indexDictOfWordFake.keys():
        countOfFakesDict[word] = frequenciesOfFake[indexDictOfWordFake[word]]

    testHeadLines = readCSV("test.csv", ngram, stemed)

    correctnessCount = 0
    for line in testHeadLines.keys():
        #stop_words="english"
        vectorizerTest = CountVectorizer(lowercase=True, stop_words=stopEnglish, analyzer='word', ngram_range=(ngram, ngram),
                                         max_df=1.0, min_df=1, max_features=None)
        temp = []
        temp.append(testHeadLines[line]["Id"])
        #print(temp)
        vectorizerTest.fit(temp)  ##fit in vector
        testindexDictOfWord = vectorizerTest.vocabulary_  ##assign index for each word by vocab function
        test_bag_of_words = vectorizerTest.transform(temp)
        test_BoW= test_bag_of_words.toarray()  ##bag of words array
        test_uniqlistOfWords = vectorizerTest.get_feature_names()  ## create unique list by feature function


        #print(testHeadLines[line]["Id"])
        result = naiveBayes(countOfRealsDict, BoWOfReal, countOfFakesDict, BoWOfFake, testindexDictOfWord, test_BoW)
        #print(result)
        if result == testHeadLines[line]["Category"]:
            correctnessCount += 1
        #print("--------------------------------------")

    uniqWordsofFiles = list(set(list(countOfRealsDict.keys()) + list(countOfFakesDict.keys())))  ##for bayes calculations
    print("uniq: ", len(uniqWordsofFiles))  # feature names
    print("correctness count: ", correctnessCount)
    accuracy = calculationofAccuracy(correctnessCount, len(testHeadLines.keys()))
    print("Accuracy = ", accuracy)

    #print("Understand Data: ")
    #understandData(countOfRealsDict,countOfFakesDict, uniqlistOfRealWords, uniqlistOfFakeWords)