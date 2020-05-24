from tqdm import tqdm
import numpy as np
import tensorflow as tf

class Model(tf.Module):
    def __init__(self, maxWords=10000, numClasses=2):
        # Using A Keras Sequential Model
        self.nn = tf.keras.Sequential()
        self.model(maxWords, numClasses)
        # Take a look at the model summary
        self.nn.summary()
        self.nn.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

    def model(self, maxWords, numClasses):
        self.nn.add(tf.keras.layers.Dense(512, input_shape=(maxWords,)))
        self.nn.add(tf.keras.layers.Activation(tf.nn.relu))
        self.nn.add(tf.keras.layers.Dropout(0.5))
        self.nn.add(tf.keras.layers.Dense(256))
        self.nn.add(tf.keras.layers.Activation(tf.nn.elu))
        self.nn.add(tf.keras.layers.Dropout(0.5))
        self.nn.add(tf.keras.layers.Dense(512))
        self.nn.add(tf.keras.layers.Activation(tf.nn.elu))
        self.nn.add(tf.keras.layers.Dense(numClasses,activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))

    def __call__(self, feature, label):
        self.nn.fit(feature,
            label,
            batch_size=256,
            validation_split=0.1,
            epochs=30)

    def evaluate(self, test_feature):
        return self.nn.predict(test_feature)

def train(inputFile):
    x = []
    y = []
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
    corpora = open(inputFile, "r")          # Opening Training File
    lines = corpora.read().splitlines()
    dictionary = dict()                     # Dictionary Creation for Class Count
    print("Preprocessing Input for Training:")
    bar = tqdm(total=len(lines))
    for line in lines:
        tmp = line.split()
        file = open(tmp[0], "r")    # Open Document
        cat = tmp[1]                # Category
        if cat not in dictionary.keys():
            # Adding Category to Dictionary and Setting Values
            dictionary[cat] = {}
        x.append(file.read())
        y.append(cat)
        bar.update()
    bar.close()
    num_classes = len(dictionary)
    # Update vocabulary and One hot encode input documents
    tokenizer.fit_on_texts(x)
    x = tokenizer.texts_to_matrix(x, mode='binary')
    # Creating two dictionaries for class name to/from index
    mydict={}
    dict2={}
    i = -1
    # dictionary maps creation
    for item in y:
        if(i > -1 and item in mydict):
            continue
        else:
           i = i+1
           mydict[item] = i
           dict2[i] = item
    index_mapping=[]
    for item in y:
        index_mapping.append(mydict[item])
    # Conversion of mapping to one hot for neural network model
    y = tf.keras.utils.to_categorical(index_mapping)
    # Model Creation
    model = Model(numClasses=num_classes)
    # Model Training
    model(x, y)
    # Return These Attributes To Ensure Correct Preprocessing and Postprocessing
    return dict2, tokenizer, model

def test(indexMapping, inputFile, outputFile, tokenizer, model):
    x = []
    corpora = open(inputFile, "r")          # Opening testing File
    lines = corpora.read().splitlines()
    print("Preprocessing Input for Testing:")
    bar = tqdm(total=len(lines))
    for line in lines:
        file = open(line, "r")
        x.append(file.read())
        bar.update()
    bar.close()
    # Use the same tokenizer as training to ensure consistency in document encoding
    x = tokenizer.texts_to_matrix(x, mode='binary')
    # Class Prediction using argmax on tf evaluate
    predictions = np.argmax(model.evaluate(x), axis=-1)
    outFile = open(outputFile, "w", newline="\n")
    # Write Output with IndexMapping for Correct Output Category
    # This format is used for analyze perl script
    for line, p in zip(lines, predictions):
        outFile.write("{} {}\n".format(line,indexMapping[p]))
    outFile.close()

if __name__ == "__main__":
    inputFile = input("Enter training file name: ")
    testFile = input("Enter test input file name: ")
    outputFile = input("Enter test output file name: ")
    dictionary, tokenizer, model = train(inputFile)
    test(dictionary, testFile, outputFile, tokenizer, model)
