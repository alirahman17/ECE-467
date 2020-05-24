import nltk
from tqdm import tqdm
import json
import string
import math

def train(inputFile, outputFile):
    corpora = open(inputFile, "r")          # Opening Training File
    lines = corpora.read().splitlines()
    porterStemmer = nltk.stem.PorterStemmer()
    dictionary = dict()                     # Dictionary Creation for Statistics
    print("Training")
    bar = tqdm(total=len(lines))
    for line in lines:
        tmp = line.split()
        file = open(tmp[0], "r")    # Open Document
        cat = tmp[1]                # Category

        if cat not in dictionary.keys():
            # Adding Category to Dictionary and Setting Values
            dictionary[cat] = {}
            dictionary[cat]["numFiles"] = 1
            dictionary[cat]["numTokens"] = 0
            dictionary[cat]["tokens"] = {}
        else:
            dictionary[cat]["numFiles"] += 1    # Incrementing Statistic for Category

        tokens = nltk.word_tokenize(file.read())    # Tokenize document
        for token in tokens:
            stem = porterStemmer.stem(token)    # Obtaining Stem of word
            if stem not in dictionary[cat]["tokens"].keys():
                # Adding tokenized stem to dictionary if it isn't already in list
                dictionary[cat]["tokens"][stem] = 1
            else:
                dictionary[cat]["tokens"][stem] += 1 # Incrementing Statistic for token
            dictionary[cat]["numTokens"] += 1   # Incrementing Statistic for Category
        bar.update()
    bar.close()
    # Writing Statistics to Output File if Specified
    if outputFile != "":
        output = open(outputFile, "w")
        json.dump(dictionary, output, indent = 4)
    return dictionary

def test(inputDict, inputFile, outputFile, flag):
    global stats
    if flag == "1":
        # Load statistics dictionary from variable
        stats = inputDict
    else:
        # Load statistics dictionary from file
        stats = json.loads(open(inputDict).read())
    categories = stats.keys()   # Obtaining Categories from File
    numFiles = 0
    for category in categories:
        numFiles += stats[category]["numFiles"] # Getting N_doc
    inFile = open(inputFile, "r")               # Opening Testing File
    lines = inFile.read().splitlines()
    porterStemmer = nltk.stem.PorterStemmer()
    output = []
    print("Classifying")
    bar = tqdm(total=len(lines))
    # Line Iteration of File for Predictions
    for line in lines:
        file = open(line, "r")      # Open Specified Document
        tokens = nltk.word_tokenize(file.read())    # Tokenize document
        tokenDict = dict()          # Dictionary of Tokens
        for token in tokens:
            stem = porterStemmer.stem(token)        # Obtain stem of token
            if stem not in string.punctuation:
                # skip punctuation tokens
                if stem not in tokenDict.keys():
                    # Adding stem to dictionary and setting value
                    tokenDict[stem] = 1
                else:
                    tokenDict[stem] += 1    # Incrementing stem value
        vocabSize = len(tokenDict)  # Getting V in Eq. 4.14
        global cW
        logTotalProb = dict()
        for category in categories:
            # Using Eq. 4.11 - 4.14 but with alpha-smoothing factor instead of add-1
            logPrior = math.log(stats[category]["numFiles"] / numFiles)
            # log probability of Eq. 4.11
            alpha = 0.05 # additive Smoothing
            logCategory = 0
            for token in tokenDict.keys():
                # Count of word in Eq. 4.14
                if token in stats[category]["tokens"].keys():
                    cW = stats[category]["tokens"][token]
                else:
                    cW = 0
                logTokenCategory = (math.log((cW + alpha) / (
                    stats[category]["numTokens"]
                    + alpha*vocabSize)) * tokenDict[token]) # log probability of Eq. 4.14
                logCategory += logTokenCategory
            logTotalProb[category] = logPrior + logCategory # Max Likelihood Sum
            # Choose Max probability for category decision
        category = max(logTotalProb, key=logTotalProb.get)
        output.append(line + " " + category + "\n")
        bar.update()
    bar.close()
    # Writing Predictions to Output File
    outFile = open(outputFile, "w", newline="\n")
    for decision in output:
        outFile.write(decision)
    outFile.close()

if __name__ == "__main__":
    # Testing Purposes # flag = input("Enter 1 for training file or 2 for statistics file:")
    flag = "1"
    statFile = ""
    if flag == "1":
        inputFile = input("Enter training file name: ")
        # Testing Purposes # statFile = input("Enter training output file name:")
        testFile = input("Enter test input file name: ")
        outputFile = input("Enter test output file name: ")
        dictionary = train(inputFile, statFile)
        test(dictionary, testFile, outputFile, flag)
    else:
        statFile = input("Enter statistics file name: ")
        testFile = input("Enter test input file name: ")
        outputFile = input("Enter test output file name: ")
        test(statFile, testFile, outputFile, flag)
