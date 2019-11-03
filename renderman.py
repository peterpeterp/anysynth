import os, os.path
import numpy as np
import tensorflow as tf 
import random
from random import shuffle
import itertools

def loadCsvIntoArray(folderPath, fileName):
    
    # Get all files from directory into a list only if they end with the
    # file type '.csv'. These files are unordered and will need sorting.
    allFiles = []
    for file in os.listdir(folderPath):
        if os.path.isfile(os.path.join(folderPath, file)) and file[-4:] == '.csv' and file.find(fileName)!=-1:
            allFiles.append(file)
    
    # Sort the files by numerical value, not alphanumerical values. It
    # gets the number by index slicing the file name and the file type
    # out of the string and casting the resulting number to an int. If
    # the string is empty, like it is for the zero'th element, then 
    # return 1 so it is the first element. 
    sortedFiles=allFiles
    
    # Get dimensions of matrix from csv. They should all be the same
    # size.
    try:
        rows,cols = np.loadtxt(folderPath + sortedFiles[0]).shape
    except:
        rows= np.loadtxt(folderPath + sortedFiles[0]).shape[0]
        cols= 1

    depth = len(sortedFiles)
    
    # Using the dimensions, we can now create an empty array to fill.
    array = np.empty((depth, rows, cols))
    
    # Loop through the sorted files and load each csv.
    for i, file in enumerate(sortedFiles):
        if cols != 1:
            array[i] = np.loadtxt((folderPath + file))
        else:
            array[i,:,0] = np.loadtxt((folderPath + file))
    return array

# Path to our folder of audio features from each render.
featurePath = os.getenv('HOME')
featurePath += '/synthyZeug/'
featureFileName = 'feature_'

# Path to the parameters of each render.
parameterPath = os.getenv('HOME')
parameterPath += '/synthyZeug/'
parameterFileName = 'parameters_'

# Features.
X = loadCsvIntoArray(featurePath, featureFileName)

# Labels.
Y = loadCsvIntoArray(parameterPath, parameterFileName)


def getRoleMask(labels,
                amountTrain, 
                amountValidation, 
                amountTest):
    
    # They should all add up to be one!
    assert (amountTrain + amountValidation + amountTest) == 1.0
    
    # Get how many examples for each one.
    totalSize = len(labels)
    trainSize = int(round(amountTrain * totalSize))
    testSize  = int(round(amountTest * totalSize))
    validSize = totalSize - trainSize - testSize
    
    # Create lists of assigned indices.
    trainList = [0] * trainSize
    testList  = [1] * testSize
    validList = [2] * validSize
    totalList = trainList + testList + validList
    
    # Sanity check.
    assert len(totalList) == len(labels)
    
    shuffle(totalList)
    return totalList
    

def splitTrainTestValidation(features,
                             labels,
                             seed = 8,
                             amountTrain = 0.8,
                             amountValidation = 0.07,
                             amountTest = 0.13):
    
    # We should have as many features as labels...
    assert len(labels) == len(features)
    
    # Seed the shuffling to keep it deterministic.
    random.seed(seed)
    
    # Get a mask to help distrubute examples to the correct
    # set.
    roleMaskList = getRoleMask(labels,
                               amountTrain,
                               amountValidation,
                               amountTest)
    
    # Lists of tuples of features and labels that represent
    # the examples.
    trainExamples = []
    testExamples  = []
    validExamples = []

    # Loop through all examples and pack them into tuples
    # of feature / label pairs. Add them to the correct list.
    for i, role in enumerate(roleMaskList):
        
        # Flatten each tuple member into a 1D list. This is
        # because I was getting tensorflow placeholder errors
        # and didn't know how to correctly define the placeholder
        # to contain nD lists so just made them 1D. As I understand
        # it this shouldn't be a problem regarding the training
        # of the neural network.
        
        # Remove the vst parameter indices from the csv (should
        # probably do this in RenderMan!)
        labelsNoIndices = map(list, zip(*labels[i]))[0]
        
        # Flattened the features from a 2D array to 1D
        flattenedFeatures = [item for sublist in features[i] for item in sublist]
        
        # Pack the treated data into an example tuple.
        exampleTuple = (flattenedFeatures, labelsNoIndices)

        if (role == 0): trainExamples.append(exampleTuple)
        elif (role == 1): testExamples.append(exampleTuple)    
        else: validExamples.append(exampleTuple)
            
    # Finally shuffle all the examples so there is no
    # order to the dataset and return the lists.
    shuffle(trainExamples)
    shuffle(validExamples)
    shuffle(testExamples)
    return trainExamples, validExamples, testExamples
    
# Fill all the correct sets with the correct tuple lists.
trainingSet, validationSet, testingSet = splitTrainTestValidation(X, Y)

# See step 1 to see where X and Y come from.
featureRows, featureCols = X[0].shape
labelRows, labelCols = Y[0].shape

trainX, trainY = zip(*trainingSet)
testX,  testY  = zip(*testingSet)
validX, validY = zip(*validationSet) 

# Network parameters.
numberInputNeurons = featureRows * featureCols
numberHiddenNeuronsLayer1 = 20
numberHiddenNeuronsLayer2 = 15
numberOutputNeurons = labelRows
activationFunction1 = 2
activationFunction2 = 0

# TF graph input.
x = tf.placeholder("float", [None, numberInputNeurons])
y = tf.placeholder("float", [None, numberOutputNeurons])

# Define the model.
def multiLayerPerceptron(x, weights, biases, a1, a2):

    # First hidden layer. 
    layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])

    if a1 == 0:
        layer1 = tf.nn.relu(layer1)
    elif a1 == 1:
        layer1 = tf.nn.elu(layer1)
    else:
        layer1 = tf.tanh(layer1)

    # Second hidden layer.
    layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])

    if a2 == 0:
        layer2 = tf.nn.relu(layer2)
    elif a2 == 1:
        layer2 = tf.nn.elu(layer2)
    else:
        layer2 = tf.tanh(layer2)

    # Output layer with linear activation (i.e nothing!)
    layerOut = tf.matmul(layer2, weights['out']) + biases['out']

    return layerOut


# Store eachlayer's weights and biases.
weights = {
    'h1': tf.Variable(tf.random_normal([numberInputNeurons, numberHiddenNeuronsLayer1])),
    'h2': tf.Variable(tf.random_normal([numberHiddenNeuronsLayer1, numberHiddenNeuronsLayer2])),
    'out': tf.Variable(tf.random_normal([numberHiddenNeuronsLayer2, numberOutputNeurons]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([numberHiddenNeuronsLayer1])),
    'b2': tf.Variable(tf.random_normal([numberHiddenNeuronsLayer2])),
    'out': tf.Variable(tf.random_normal([numberOutputNeurons]))
}

# 'Construct' the model. 
prediction = multiLayerPerceptron(x, 
                                  weights, 
                                  biases, 
                                  activationFunction1, 
                                  activationFunction2)

# Define loss.
#cost = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(prediction, y))))
cost = tf.reduce_sum(tf.abs(tf.subtract(prediction, y)))   


# Parameters.
learningRate = 0.001
trainingEpochs = 10000
batchSize = 50
displayStep = 1
maxConsecutiveEpochs = 15

# Where we save the biases and weights to.
modelName = 'model_'
modelName += str(numberHiddenNeuronsLayer1) + '_' 
modelName += str(numberHiddenNeuronsLayer2) + '_'
modelName += str(activationFunction1) + '_'
modelName += str(activationFunction2) + '_absCost.ckpt'

saveDirectory = os.getenv('HOME')
saveDirectory += '/synthyZeug/'
savePath = saveDirectory + modelName

optimiser = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

# Add an op to initialize the variables.
init = tf.global_variables_initializer()

# Add op to save and restore all the variables.
saver = tf.train.Saver()

################################################################################
# Commencing training.
##
with tf.Session() as sess:

    sess.run(init)
                                    
    consectutiveEpochs = 0
    lastAverageValidCost = 99999.

    # Training Cycle.
    for epoch in range(trainingEpochs):

        averageCost = 0
        averageValidCost = 0
        totalBatches = int(len(trainingSet) / batchSize)
        
        # Training the model.
        for i in range(totalBatches - 1):
            
            # For index slicing.
            start = i * batchSize
            end = (i + 1) * batchSize
            
            _, c = sess.run([optimiser, cost], feed_dict={ x: trainX[start:end],
                                                           y: trainY[start:end] })
            averageCost += c / totalBatches
            
            # Validation for early stopping.
            validCost = sess.run(cost, feed_dict={ x: validX, y: validY })  
            averageValidCost += validCost / totalBatches
        
        # Logic for early stopping here.
        if (averageValidCost > lastAverageValidCost): 
            
            consectutiveEpochs += 1
            
            # Stop training if the validation error has risen too many times.
            if consectutiveEpochs >= maxConsecutiveEpochs:
                print 'Early stopping: ' + str(maxConsecutiveEpochs) + ' consecutively greater epochs reached!'
                break
            
        elif (consectutiveEpochs > 0): 
            consectutiveEpochs -= 1       
                
        lastAverageValidCost = averageValidCost
            
        # Display information about the current epoch.
        if epoch % displayStep == 0:
            
            print "Epoch:", '%04d' % (epoch + 1)
            print "Training cost =", "{:.9f}".format(averageCost / batchSize)
            
            # Test the model.
            testingCost = sess.run(cost, feed_dict={ x: testX,
                                                     y: testY })
            
            print "Testing cost  =", "{:.9f}".format(testingCost / len(testY) / 2)
                                    
    # Save the variables to disk.
    save = saver.save(sess, savePath)
    print ("Model saved in file: %s" % save)
    
    

################################################################################
# Save scores to file
##
scoreFile = saveDirectory + "scores.txt"
with open(scoreFile, "a") as myfile:
    line = "Score: " + str(testingCost) + " --- " + modelName + "\n"
    myfile.write(line)


model = '/home/bla/synthyZeug/model_20_15_2_0_absCost.ckpt'
meta = '.meta'

with tf.Session() as sess:
    
    sess.run(init)
    
    newSaver = tf.train.import_meta_graph((model + meta))
    newSaver.restore(sess, model)
    
    exampleX, exampleY = zip(*testingSet)
    
    pre = sess.run([prediction], feed_dict={ x: exampleX })
    
    allTests = []
    
    for i in range(len(pre[0])):
        
        abs1 = abs(pre[0][i][0] - exampleY[i][0])
        abs2 = abs(pre[0][i][1] - exampleY[i][1])
        
        totalAbs = abs1+abs2
        
        absoluteDistanceTable = " " +("%04d" % i) + " : " + ("%.5f" % round(abs1, 5)) + " : " + ("%.5f" % round(abs2, 5)) + " : " + ("%.5f" % round(totalAbs, 5))
        
        predictFormatted = "[ " + ("%04.0f" % round(pre[0][i][0] * 1500)) + ", " + ("%04.1f" % round(pre[0][i][1] * 7.6 - 3.8, 1)) + " ]"
        actualFormatted = "[ " + ("%04.0f" % round(exampleY[i][0] * 1500)) + ", " + ("%04.1f" % round(exampleY[i][1] * 7.6 - 3.8, 1)) + " ]"
        
        # Print absoulate distance of parameters to
        line = absoluteDistanceTable + "        " + actualFormatted + "   " + predictFormatted
        
        allTests.append((totalAbs, line))
        
    allTests.sort(key=lambda x: x[0])
    
    print "\n\nIn order of most to least similar predictions compared to the actual parameters:\n\nIndex     abs1      abs2    Total abs          Actual         Predicted"
    print " __________________________________        _______________________________"
    for i in range(len(allTests)):
        print allTests[i][1]