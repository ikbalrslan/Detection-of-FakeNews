from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


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

    #print(trainReal)
    #print(indexDictOfWordReal["trump"]) ##index of trump word in the BoW
    #print(bag_of_words_real.shape)
    #print(type(bag_of_words_real))
    #print(BoWOfReal)

    print()

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

    #print(trainFake)
    #print(indexDictOfWordFake["trump"]) ##index of trump word in the BoW
    #print(bag_of_words_fake.shape)
    #print(type(bag_of_words_real))
    #print(BoWOfFake)
    #print(uniqlistOfFakeWords) #feature names



    testHeadLines = list(testReal + testFake)


    for line in testHeadLines:
        #stop_words="english"
        vectorizerTest = CountVectorizer(lowercase=True, analyzer='word', ngram_range=(1, 1),
                                         max_df=1.0, min_df=1, max_features=None)
        temp = []
        temp.append(line)
        #print(temp)
        vectorizerTest.fit(temp)  ##fit in vector
        testindexDictOfWord = vectorizerTest.vocabulary_  ##assign index for each word by vocab function
        test_bag_of_words = vectorizerTest.transform(temp)
        test_BoW= test_bag_of_words.toarray()  ##bag of words array
        test_uniqlistOfWords = vectorizerTest.get_feature_names()  ## create unique list by feature function

        #print(test_BoW)
        naiveBayes(indexDictOfWordReal, indexDictOfWordFake, BoWOfReal, BoWOfFake, testindexDictOfWord, test_BoW)

def naiveBayes(indexDictOfWordReal, indexDictOfWordFake, BoWOfReal, BoWOfFake, testindexDictOfWord, test_BoW):
    uniqWordsofFiles = list(set(list(indexDictOfWordReal.keys()) + list(indexDictOfWordFake.keys())))  ##for bayes calculations
    allCountOfReal = 0

    """
    Burada real datadaki toplam kelime sayısı lazım uniq ve tekrarlı
    """

    print(list(testindexDictOfWord.keys()))
    for word in testindexDictOfWord.keys():
        indexOfTest = testindexDictOfWord[word]
        countFromClass = 0
        try:
            indexOfReal = indexDictOfWordReal[word]
            for i in BoWOfReal:
                countFromClass += i[indexOfReal]
            print(word, countFromClass)
            #print(test_BoW[0][index])
        except:
            countFromClass = 0




    print("----------------------------------------------")





    """
    for i in range(len(BoWOfReal)):
        if BoWOfReal[i][2821] > 1:
            print(i, BoWOfReal[i][2821])

    """
