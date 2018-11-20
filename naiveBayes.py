import numpy as np
from math import log10



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
            ##for words that does not include in the model data
            realCondProb[word] = (0 + smoothing) / (wordCountofReal + len(uniqWordsofFiles))
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
            ##for words that does not include in the model data
            fakeCondProb[word] = (0 + smoothing) / (wordCountofFake + len(uniqWordsofFiles))
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

def understandData(countOfRealsDict,countOfFakesDict, uniqlistOfRealWords, uniqlistOfFakeWords):
    #sort dicts
    sortedRealDict = [{k:countOfRealsDict[k]} for k in sorted(countOfRealsDict, key=countOfRealsDict.get, reverse=True)]
    sortedFakeDict = [{k:countOfFakesDict[k]} for k in sorted(countOfFakesDict, key=countOfFakesDict.get, reverse=True)]
    #print(sortedRealDict)
    counterReal = 0
    topThreeReal = []
    counterFake = 0
    topThreeFake = []

    ##########################################################################################################
    # Looks for uniq top 3 real words which does not inside the fakes. Same process with the fake words again#
    ##########################################################################################################
    for item in sortedRealDict:
        id_word = (list(item.keys()))[0]
        if ((id_word not in uniqlistOfFakeWords) and counterReal < 3):
            #print(id_word)
            topThreeReal.append(item)
            counterReal += 1
    print("TOP Reals: ", topThreeReal)

    for item in sortedFakeDict:
        id_word = (list(item.keys()))[0]
        if ((id_word not in uniqlistOfRealWords) and counterFake < 3):
            #print(id_word)
            topThreeFake.append(item)
            counterFake += 1
    print("TOP Fakes: ", topThreeFake)