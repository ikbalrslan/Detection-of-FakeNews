import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize



def readCSV(testFile, ngram, stemed):
    testHeadlines = pd.read_csv(testFile, sep=',', error_bad_lines=False, encoding="latin-1", index_col=False,
                                  warn_bad_lines=False, low_memory=False)

    testHeadlines.columns = ['Id', 'Category']
    testHeadlines.set_index("Id", "Category")
    testHeadDict = testHeadlines.to_dict(orient="index")

    ## if ngram == 2 add tokens to the lines start and end
    if ngram == 2:
        for i in testHeadDict.keys():
            testHeadDict[i]['Id'] = 's_s_s {0} e_e_e'.format(testHeadDict[i]['Id'])  # print(testHeadDict)

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

def readFile(trainReal, trainFake, ngram, stemed):

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

    ## if ngram == 1 do not add tokens to the lines start and end. But if ngram == 2 add those to the lines
    if ngram == 1:
        copyLinesOfReal = linesOfReal.copy()
    elif ngram == 2:
        copyLinesOfReal = new_linesOfReal.copy()

    for line in copyLinesOfReal:
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

    ## if ngram == 1 do not add tokens to the lines start and end. But if ngram == 2 add those to the lines
    if ngram == 1:
        copyLinesOfFake = linesOfFake.copy()
    elif ngram == 2:
        copyLinesOfFake = new_linesOfFake.copy()
    for line in copyLinesOfFake:
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
    elif stemed == False:
        if ngram == 2:
            return new_linesOfReal,new_linesOfFake
        elif ngram == 1:
            return linesOfReal, linesOfFake







