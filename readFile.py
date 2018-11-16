from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from math import log10
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize



def readCSV(testFile,stemed):
    testHeadlines = pd.read_csv(testFile, sep=',', error_bad_lines=False, encoding="latin-1", index_col=False,
                                  warn_bad_lines=False, low_memory=False)

    testHeadlines.columns = ['Id', 'Category']
    testHeadlines.set_index("Id", "Category")
    testHeadDict = testHeadlines.to_dict(orient="index")

    for i in testHeadDict.keys():
        testHeadDict[i]['Id'] = 's_s_s {0} e_e_e'.format(testHeadDict[i]['Id'])
    #print(testHeadDict)

    ps = PorterStemmer()
    if stemed == True:
        #print("test stemmed: ")
        for line in testHeadDict.keys():
            words = word_tokenize(testHeadDict[line]['Id'])
            stemmedLine = []
            for word in words:
                word = ps.stem(word)
                # print(word)
                stemmedLine.append(word)
            stemLine = ' '.join(stemmedLine)
            #print(stemLine)
            testHeadDict[line]['Id'] = stemLine
        #print(testHeadDict)

    return testHeadDict # return test dataframe

def readFile(trainReal, trainFake, stemed):

    with open(trainReal) as fileReal:
        linesOfReal = fileReal.read().splitlines() ##split text file into lines and store in list

    with open(trainFake) as fileFake:
        linesOfFake = fileFake.read().splitlines() ##split text file into lines and store in list

    new_linesOfReal = ['s_s_s {0} e_e_e'.format(i) for i in linesOfReal] ## add start end words to lines
    new_linesOfFake = ['s_s_s {0} e_e_e'.format(i) for i in linesOfFake] ## add start end words to lines
    ##########################################################
    ps = PorterStemmer()
    #print("reals: ")
    stemLinesOfReal = []
    for line in new_linesOfReal:
        words = word_tokenize(line)
        stemmedLine = []
        for word in words:
            word = ps.stem(word)
            #print(word)
            stemmedLine.append(word)
        stemLine = ' '.join(stemmedLine)
        stemLinesOfReal.append(stemLine)
    #print(stemLinesOfReal)

    #print("Fakes: ")
    stemLinesOfFake = []
    for line in new_linesOfFake:
        words = word_tokenize(line)
        stemmedLine = []
        for word in words:
            word = ps.stem(word)
            #print(word)
            stemmedLine.append(word)
        stemLine = ' '.join(stemmedLine)
        stemLinesOfFake.append(stemLine)
    #print(stemLinesOfFake)
    ##########################################################
    if stemed == True:
        return stemLinesOfReal, stemLinesOfFake
    else:
        return new_linesOfReal,new_linesOfFake


def bagOfWords(linesOfReal,linesOfFake, stopwords, ngram, stemed): ## ngram_range=(1, 1) for unigram, ngram_range=(2, 2) for bigram
    # create the transform / stop_words="english"
    vectorizerReal = CountVectorizer(lowercase=True, stop_words=stopwords, analyzer='word', ngram_range=(ngram, ngram), max_df=1.0, min_df=1,
                                     max_features=None)

    trainReal = linesOfReal[:]


    vectorizerReal.fit(trainReal)  ##fit in vector
    indexDictOfWordReal = vectorizerReal.vocabulary_  ##assign index for each word by vocab function
    bag_of_words_real = vectorizerReal.transform(trainReal)
    BoWOfReal = bag_of_words_real.toarray()  ##bag of words array
    uniqlistOfRealWords = vectorizerReal.get_feature_names()  ## create unique list by feature function
    #print("uniq real: ",len(uniqlistOfRealWords))
    print("uniq real: ",uniqlistOfRealWords)

    # create the transform / stop_words="english"
    vectorizerFake = CountVectorizer(lowercase=True, stop_words=stopwords, analyzer='word', ngram_range=(ngram, ngram), max_df=1.0, min_df=1,
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

    testHeadLines = readCSV("test.csv", stemed)

    correctnessCount = 0
    for line in testHeadLines.keys():
        #stop_words="english"
        vectorizerTest = CountVectorizer(lowercase=True, stop_words=stopwords, analyzer='word', ngram_range=(ngram, ngram),
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
    understandData(countOfRealsDict,countOfFakesDict)


def naiveBayes(countOfRealsDict, BoWOfReal, countOfFakesDict, BoWOfFake, testindexDictOfWord, test_BoW):
    uniqWordsofFiles = list(set(list(countOfRealsDict.keys()) + list(countOfFakesDict.keys()))) ##for bayes calculations
    #print("uniq: ", len(uniqWordsofFiles))  # feature names
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
        if word in countOfRealsDict.keys():
            realCondProb[word] = (countOfRealsDict[word] + smoothing) / (wordCountofReal + len(uniqWordsofFiles))
            #print(realCondProb[word] + 1, word, countOfRealsDict[word])
        elif word in countOfFakesDict.keys():
            realCondProb[word] = (0 + smoothing) / (wordCountofReal + len(uniqWordsofFiles))
        else:
            realCondProb[word] = (0 + smoothing) / (wordCountofReal + len(uniqWordsofFiles)) ##for words that does not include in the model data
        ## take test index of word and find count of word in test_BoW then multiply with log of word count
        realProbability += (test_BoW[0][testindexDictOfWord[word]] * log10(realCondProb[word]))

    #print("real probability: ", realProbability)

    ## Conditional Probabilities of fake class
    fakeCondProb = {} ##holds condition probabilities of test headline words
    fakeProbability = log10(fakeRatio)  ## initially
    for word in testindexDictOfWord.keys():
        if word in countOfFakesDict.keys():
            fakeCondProb[word] = (countOfFakesDict[word] + smoothing) / (wordCountofFake + len(uniqWordsofFiles))
            #print(fakeCondProb[word] + 1, word, countOfFakesDict[word])
        elif word in countOfRealsDict.keys():
            fakeCondProb[word] = (0 + smoothing) / (wordCountofFake + len(uniqWordsofFiles))
        else:
            fakeCondProb[word] = (0 + smoothing) / (wordCountofFake + len(uniqWordsofFiles)) ##for words that does not include in the model data
        ## take test index of word and find count of word in test_BoW then multiply with log of word count
        fakeProbability += (test_BoW[0][testindexDictOfWord[word]] * log10(fakeCondProb[word]))

    #print("fake probability: ", fakeProbability)


    if(realProbability >= fakeProbability):
        return "real"
    else:
        return "fake"

def calculationofAccuracy(correctnessCount, testSize):
    accuracy = 100 * (correctnessCount / testSize)
    return accuracy

def understandData(countOfRealsDict,countOfFakesDict):
    #sort dicts
    sortedRealDict = [(k, countOfRealsDict[k]) for k in sorted(countOfRealsDict, key=countOfRealsDict.get, reverse=True)]
    sortedFakeDict = [(k, countOfFakesDict[k]) for k in sorted(countOfFakesDict, key=countOfFakesDict.get, reverse=True)]
    counterReal = 0
    counterFake = 0
    #print(sortedRealDict)
    #print(sortedFakeDict)

