import os
import numpy as np

def splitFiles(input, train, test, valid):
    np.random.seed(31415)   # seed for shuffling data
    file = file = open(input, "r")
    lines = file.read().splitlines()
    index = np.arange(len(lines))
    np.random.shuffle(index)
    trainFile = open(train, "w", newline="\n")
    testFile = open(test, "w", newline="\n")
    trueTestFile = open(valid, "w", newline="\n")
    val = int(len(lines) * 0.20)
    trainIndex = index[val:]
    testIndex = index[:val]
    for i in trainIndex:
        trainFile.write(lines[i])
        trainFile.write("\n")
    for i in testIndex:
        line = lines[i].split()
        trueTestFile.write(lines[i])
        trueTestFile.write("\n")
        testFile.write(line[0])
        testFile.write("\n")

if __name__ == "__main__":
    inputFile = input("Enter training file name: ")
    trainingFile = input("Enter training output file name: ")
    testingFile = input("Enter testing output file name: ")
    trueTestFile = input("Enter true test output file name: ")
    splitFiles(inputFile, trainingFile, testingFile, trueTestFile)
