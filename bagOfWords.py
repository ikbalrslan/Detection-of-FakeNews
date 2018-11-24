from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from readFile import readCSV
from naiveBayes import naiveBayes,calculationofAccuracy,understandData
from math import log10
import csv




## ngram_range=(1, 1) for unigram, ngram_range=(2, 2) for bigram
def bagOfWords(linesOfReal,linesOfFake, ngram, stopEnglish, stemed):
    # create the transform / stop_words="english"
    vectorizerReal = CountVectorizer(lowercase=True, stop_words=stopEnglish, analyzer='word',
                                     ngram_range=(ngram, ngram), max_df=1.0, min_df=1,max_features=None)

    trainReal = linesOfReal[:]


    vectorizerReal.fit(trainReal)  ##fit in vector
    indexDictOfWordReal = vectorizerReal.vocabulary_  ##assign index for each word by vocab function
    bag_of_words_real = vectorizerReal.transform(trainReal)
    BoWOfReal = bag_of_words_real.toarray()  ##bag of words array
    uniqlistOfRealWords = vectorizerReal.get_feature_names()  ## create unique list by feature function
    #print("uniq real: ",len(uniqlistOfRealWords))
    #print("uniq real: ",uniqlistOfRealWords)

    # create the transform / stop_words="english"
    vectorizerFake = CountVectorizer(lowercase=True, stop_words=stopEnglish, analyzer='word',
                                     ngram_range=(ngram, ngram), max_df=1.0, min_df=1,max_features=None)

    trainFake = linesOfFake[:]

    vectorizerFake.fit(trainFake)  ##fit in vector
    indexDictOfWordFake = vectorizerFake.vocabulary_  ##assign index for each word by vocab function
    bag_of_words_fake = vectorizerFake.transform(trainFake)
    BoWOfFake = bag_of_words_fake.toarray()  ##bag of words array
    uniqlistOfFakeWords = vectorizerFake.get_feature_names()  ## create unique list by feature function

    countOfRealsDict = {}
    countOfFakesDict = {}

    frequenciesOfReal = np.sum(BoWOfReal,axis=0) ## sum counts of the same worlds
    for word in indexDictOfWordReal.keys():
        countOfRealsDict[word] = frequenciesOfReal[indexDictOfWordReal[word]]
    frequenciesOfFake = np.sum(BoWOfFake,axis=0) ## sum counts of the same worlds
    for word in indexDictOfWordFake.keys():
        countOfFakesDict[word] = frequenciesOfFake[indexDictOfWordFake[word]]

    testHeadLines = readCSV("test.csv", ngram, stemed)

    #write results to the csv file
    with open('output.csv', mode='w') as csv_file:
        fieldnames = ['Id', 'Category']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        correctnessCount = 0
        for line in testHeadLines.keys():
            #stop_words="english"
            vectorizerTest = CountVectorizer(lowercase=True, stop_words=stopEnglish, analyzer='word',
                                             ngram_range=(ngram, ngram), max_df=1.0, min_df=1, max_features=None)
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

            ## write results to csv
            writer.writerow({'Id': testHeadLines[line]["Id"], 'Category': result})


            #print(result)
            if result == testHeadLines[line]["Category"]:
                correctnessCount += 1
            #print("--------------------------------------")

        uniqWordsofFiles = list(set(list(countOfRealsDict.keys()) + list(countOfFakesDict.keys())))##for bayes calculations
        #print("uniq word Count: ", len(uniqWordsofFiles))  # feature names
        print("correctness count: ", correctnessCount)
        accuracy = calculationofAccuracy(correctnessCount, len(testHeadLines.keys()))
        print("Accuracy = ", accuracy)
        return [countOfRealsDict, countOfFakesDict, uniqlistOfRealWords, uniqlistOfFakeWords]

def tenWordsWith_Tf_Idf(linesOfReal, linesOfFake):
    tf_real = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')
    txt_fitted = tf_real.fit(linesOfReal)
    txt_transformed = txt_fitted.transform(linesOfReal)
    indexDictOfReals = tf_real.vocabulary_

    real_idf = tf_real.idf_
    idf_words = dict(zip(txt_fitted.get_feature_names(), real_idf))

    # sort dict
    sortedReal_idf = [{k: idf_words[k]} for k in sorted(idf_words, key=idf_words.get, reverse=True)]
    #print(sortedReal_idf)

    # get feature names
    feature_names = np.array(tf_real.get_feature_names())
    sorted_by_idf = np.argsort(tf_real.idf_)
    #print("Real Features with lowest idf:\n{}".format(feature_names[sorted_by_idf[:10]]))
    #print("\nReal Features with highest idf:\n{}".format(feature_names[sorted_by_idf[-10:]]))

    newReal = tf_real.transform(linesOfReal)
    # find maximum value for each of the features over all of dataset:
    max_val = newReal.max(axis=0).toarray().ravel()
    # sort weights from smallest to biggest and extract their indices
    sort_by_tfidf = max_val.argsort()

    print("\n10 words whose presence most strongly predicts that the news is real.:")
    for i in range(len(feature_names[sort_by_tfidf[-10:]])):
        print(i + 1, "- ", feature_names[sort_by_tfidf[-10:]][i])
    print("\n10 words whose absence most strongly predicts that the news is real.:")
    for i in range(len(feature_names[sort_by_tfidf[:10]])):
        print(i + 1, "- ", feature_names[sort_by_tfidf[:10]][i])


    tf_fake = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')
    txt_fitted = tf_fake.fit(linesOfFake)
    txt_transformed = txt_fitted.transform(linesOfFake)
    indexDictOfReals = tf_fake.vocabulary_

    real_idf = tf_fake.idf_
    idf_words = dict(zip(txt_fitted.get_feature_names(), real_idf))

    # sort dict
    sortedReal_idf = [{k: idf_words[k]} for k in sorted(idf_words, key=idf_words.get, reverse=True)]
    #print(sortedReal_idf)

    # get feature names
    feature_names = np.array(tf_fake.get_feature_names())
    sorted_by_idf = np.argsort(tf_fake.idf_)
    #print("Fake Features with lowest idf:\n{}".format(feature_names[sorted_by_idf[:10]]))
    #print("\nFake Features with highest idf:\n{}".format(feature_names[sorted_by_idf[-10:]]))

    newFake = tf_fake.transform(linesOfReal)
    # find maximum value for each of the features over all of dataset:
    max_val = newFake.max(axis=0).toarray().ravel()
    # sort weights from smallest to biggest and extract their indices
    sort_by_tfidf = max_val.argsort()

    print("\n10 words whose presence most strongly predicts that the news is fake.: ")
    for i in range(len(feature_names[sort_by_tfidf[-10:]])):
        print(i+1, "- ", feature_names[sort_by_tfidf[-10:]][i])
    print("\n10 words whose absence most strongly predicts that the news is fake.: ")
    for i in range(len(feature_names[sort_by_tfidf[:10]])):
        print(i + 1, "- ", feature_names[sort_by_tfidf[:10]][i])



def tenWordsWithCondProb(countOfRealsDict, countOfFakesDict):

    uniqWordsofFiles = list(set(list(countOfRealsDict.keys()) + list(countOfFakesDict.keys())))  ##for bayes calculations
    # print("uniq: ", len(uniqWordsofFiles))  # feature names
    wordCountofReal = np.sum(list(countOfRealsDict.values()))  ## sum counts of the worlds
    wordCountofFake = np.sum(list(countOfFakesDict.values()))  ## sum counts of the worlds

    ## Conditional Probabilities of real class
    realCondProb = {}  ##holds condition probabilities of test headline words
    ## Conditional Probabilities of fake class
    fakeCondProb = {}  ##holds condition probabilities of test headline words

    smoothing = 1  ##smoothing number

    testHeadLines = readCSV("test.csv", 1, stemed=False)
    for line in testHeadLines.keys():
        # stop_words="english"
        vectorizerTest = CountVectorizer(lowercase=True, stop_words="english", analyzer='word',
                                         ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None)
        temp = []
        temp.append(testHeadLines[line]["Id"])
        # print(temp)
        vectorizerTest.fit(temp)  ##fit in vector
        testindexDictOfWord = vectorizerTest.vocabulary_  ##assign index for each word by vocab function

        for word in testindexDictOfWord.keys():
            if word in countOfRealsDict.keys():
                realCondProb[word] = log10((countOfRealsDict[word] + smoothing) / (wordCountofReal + len(
                    uniqWordsofFiles)))  # print(realCondProb[word] + 1, word, countOfRealsDict[word])
            elif word in countOfFakesDict.keys():
                realCondProb[word] = log10((0 + smoothing) / (wordCountofReal + len(uniqWordsofFiles)))
            else:
                ##for words that does not include in the model data
                realCondProb[word] = log10((0 + smoothing) / (wordCountofReal + len(uniqWordsofFiles)))

        for word in testindexDictOfWord.keys():
            if word in countOfFakesDict.keys():
                fakeCondProb[word] = log10((countOfFakesDict[word] + smoothing) / (wordCountofFake + len(
                    uniqWordsofFiles)))  # print(fakeCondProb[word] + 1, word, countOfFakesDict[word])
            elif word in countOfRealsDict.keys():
                fakeCondProb[word] = log10((0 + smoothing) / (wordCountofFake + len(uniqWordsofFiles)))
            else:
                ##for words that does not include in the model data
                fakeCondProb[word] = log10((0 + smoothing) / (wordCountofFake + len(uniqWordsofFiles)))

    print("10 words with condtional probabilities whose presence most strongly predicts that the news is real:")
    # sort dict
    sortedRealCondProb = [{k: realCondProb[k]} for k in sorted(realCondProb, key=realCondProb.get, reverse=True)]
    for i in range(len(sortedRealCondProb[:10])):
        print(i+1,"- ",sortedRealCondProb[i])

    print("\n10 words with condtional probabilities whose presence most strongly predicts that the news is fake:")
    # sort dict
    sortedFakeCondProb = [{k: fakeCondProb[k]} for k in sorted(fakeCondProb, key=fakeCondProb.get, reverse=True)]
    #print(sortedFakeCondProb[:10])
    for i in range(len(sortedFakeCondProb[:10])):
        print(i+1,"- ",sortedFakeCondProb[i])


