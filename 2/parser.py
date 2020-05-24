# Ali Rahman
import string

class Node(object):
    # Node object to create tree for parsing
    def __init__(self,symbol,l,r=None):
        self.symbol = symbol
        self.l = l
        self.r = r

def genParse(node):
    # Function to generate text output for syntax tree in bracket form
    global text
    if node.r == None:
        text = "[" + node.symbol + " " + node.l + "]"
    else:
        text = "[" + node.symbol + " " + genParse(node.l) + " " + genParse(node.r) + "]"
    return text

def parse(grammarFile):
    # Grammar File Reading and Storing
    lines = grammarFile.read().splitlines()
    grammarRules = []
    # Creating rules for each line and saving for cky algorithm
    for line in lines:
        tmp = []
        line = line.split()

        tmp.append(line[0])
        tmp.append(line[2])
        if(len(line) == 4):
            tmp.append(line[3])
        grammarRules.append(tmp)

    # Remove punctuation
    translator = str.maketrans("","", string.punctuation)

    while(True):
        sentence = input("Enter sentence for parse check (quit to exit): ")
        translatedSentence = sentence.translate(translator).lower()
        # Exit Condition
        if translatedSentence == "quit":
            print("Completed Parsing Session")
            exit(0)
        # Space Allocation for table used in cky algorithm
        tokens = translatedSentence.split()
        table = [[[] for i in range(len(tokens) - j)] for j in range(len(tokens))]

        i = 0
        for token in tokens:
            for rule in grammarRules:
                if token == rule[1]:
                    table[0][i].append(Node(rule[0],token))
            i += 1

        # CKY Algorithm (fill in table based on rules)
        # words = remaining words
        # cell = cell in table
        # left = left partition
        for words in range(2, len(tokens) + 1):
            for cell in range(0, (len(tokens) - words + 1)):
                for left in range(1, words):
                    right = words - left
                    leftCell = table[left - 1][cell]
                    rightCell = table[right - 1][cell + left]

                    for rule in grammarRules:
                        leftNodes = []
                        for n in leftCell:
                            if n.symbol == rule[1]:
                                leftNodes.append(n)
                        if leftNodes:
                            rightNodes = []
                            for n in rightCell:
                                if len(rule) == 3:
                                    if n.symbol == rule[2]:
                                        rightNodes.append(n)
                            for leftNode in leftNodes:
                                for rightNode in rightNodes:
                                    table[words - 1][cell].append(Node(rule[0], leftNode, rightNode))

        # Creation of different parse trees from end to beginning
        startSymbol = "S"
        nodes = []
        for node in table[-1][0]:
            if node.symbol == startSymbol:
                nodes.append(node)

        # Printing Parse Trees
        if nodes:
            i = 1
            for node in nodes:
                print("\nParse #" + str(i) + ": ")
                print(genParse(node) + "\n")
                i += 1
        else:
            print("NO VALID PARSES\n")


if __name__ == "__main__":
    inputFile = input("Enter file name containing CNF of grammar: ")
    grammarFile = open(inputFile, "r")
    parse(grammarFile)
