import PIL.Image as pimage;
import os;
import numpy as np;
import math;
import sys;

imageHeight = 28;
imageWidth = 28;

firstLayerDepth = 40;
secondLayerDepth = 20;
outputLayerDepth = 10;

# theta_1 = np.ones((imageWidth * imageHeight, firstLayerDepth));
# theta_2 = np.ones((firstLayerDepth, secondLayerDepth));
# theta_3 = np.ones((secondLayerDepth, outputLayerDepth));

theta_1 = np.random.rand(imageWidth*imageHeight, firstLayerDepth)
theta_2 = np.random.rand(firstLayerDepth, secondLayerDepth)
theta_3 = np.random.rand(secondLayerDepth, outputLayerDepth)

alpha = 0.01;
_lambda = 0.01;

trainingSet = None;
trainingSetSize = None;
testSetSize = None;

cost = sys.maxsize;
accuracy = 0;

def init():
    # loads theta from file
    # if no theta found then leaves theta as an array of ones
    def loadTheta():
        thetaLayerOneFile = open(os.path.join("C:/Users", "Wesley", "Desktop", "Re-Domum", "theta_layer_one.bin.npy"),
                                "rb+");
        thetaLayerTwoFile = open(os.path.join("C:/Users", "Wesley", "Desktop", "Re-Domum", "theta_layer_two.bin.npy"),
                             "rb+");
        thetaLayerThreeFile = open(os.path.join("C:/Users", "Wesley", "Desktop", "Re-Domum", "theta_layer_three.bin.npy"),
                                "rb+");


        global theta_1;
        global theta_2;
        global theta_3;

        _layerOne = None;
        _layerTwo = None;
        _layerThree = None;

        try :
            _layerOne = np.load(thetaLayerOneFile);
            _layerTwo = np.load(thetaLayerTwoFile);
            _layerThree = np.load(thetaLayerThreeFile);
        except:
            print('failed to init _layerone')

        thetaLayerOneFile.close();
        thetaLayerTwoFile.close();
        thetaLayerThreeFile.close();

        if _layerOne is not None or _layerOne is not "":
           theta_1 = _layerOne;

        if _layerTwo is not None or _layerTwo is not "":
            theta_2 = _layerTwo;

        if _layerThree is not None or _layerThree is not "":
            theta_3 = _layerThree;

    # counts the number of files in the training set
    def countTrainingSets():
        global trainingSetSize;
        # path = os.path.join('C:/', 'Users', 'Wesley', 'Desktop', 'Re-Domum', 'training_set_processed')
        # trainingSetSize = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

        trainingSetSize = 0;

        path = os.path.join('C:/Users/Wesley/Desktop/handwritten/mnist_train.csv')

        with open(path) as file:
            for _line in file.readlines():
                trainingSetSize+=1;


    def countTestSets():
        global testSetSize;
        # path = os.path.join('C:/', 'Users', 'Wesley', 'Desktop', 'Re-Domum', 'test_set_processed')
        # testSetSize = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

        testSetSize = 0;

        path = os.path.join('C:/Users/Wesley/Desktop/handwritten/mnist_test.csv')

        with open(path) as file:
            for _line in file.readlines():
                testSetSize+=1;

    def loadMetaData():
        with open(os.path.join('C:/Users', 'Wesley', 'Desktop', 'Re-Domum', 'misc.txt')) as miscFile:
            _cost = float(miscFile.readline().strip('\n').strip('[').strip(']'));

            if _cost is not None or _cost is not "":
                global cost; cost = _cost;

            _accuracy = float(miscFile.readline().strip('\n'));

            if _accuracy is not None or _accuracy is not "":
               global accuracy; accuracy = _accuracy;

    def initTheta(layer):

        for row in range(0, layer.shape[0]):
            for c in range(0, len(layer[row])):
                layer[row][c] = layer[row][c] * (1/16000) - 1/32000

        return layer;

    # loadTheta();
    global theta_1; global theta_2; global theta_3;

    theta_1 = initTheta(theta_1)
    theta_2 = initTheta(theta_2)
    theta_3 = initTheta(theta_3)

    countTrainingSets();
    countTestSets();
    loadMetaData();

def getOutputLayer(a_3):

    # return sigmoid(theta_3.transpose().dot(a_3.transpose())).transpose()

    return sigmoid(a_3.dot(theta_3))

    # return (theta_3.transpose().dot(a_3.transpose())).transpose()

def getThirdActivationLayer(a_2):

    # print(theta_2.shape)
    # print(a_2.shape)

    # return theta_2.transpose().dot(a_2.transpose()).transpose()

    # return sigmoid(theta_2.transpose().dot(a_2.transpose())).transpose()

    return sigmoid(a_2.dot(theta_2));

def getSecondActivationLayer(a_1):

    # print(a_1.shape)
    # print(theta_1.shape)

    # return sigmoid(theta_1.transpose().dot(a_1.transpose())).transpose()

    return sigmoid(a_1.dot(theta_1))

    # return theta_1.transpose().dot(a_1.transpose()).transpose()

    # return sigmoid(a_1.dot(theta_1));

def backPropagation(DeltaOne, DeltaTwo, DeltaThree, trainingSetSize):

    global theta_1;
    global theta_2;
    global theta_3;

    matrixOne = updateMatrix(DeltaOne, trainingSetSize, theta_1)
    matrixTwo = updateMatrix(DeltaTwo, trainingSetSize, theta_2)
    matrixThree = updateMatrix(DeltaThree, trainingSetSize, theta_3)

    theta_1 = gradientDescent(matrixOne, theta_1);
    theta_2 = gradientDescent(matrixTwo, theta_2);
    theta_3 = gradientDescent(matrixThree, theta_3);

def updateMatrix(Delta, trainingSetSize, thetaLayer):

    for row in range(0, Delta.shape[0]):
        for i in range(0, len(Delta[row])):
            Delta[row][i] = (Delta[row][i]/trainingSetSize) + (_lambda * thetaLayer[row][i]);

    return Delta

def gradientDescent(Delta, layer):

    for row in range(0, Delta.shape[0]):
        for i in range(0, len(Delta[row])):
            layer[row][i] -= alpha*Delta[row][i];

    return layer;

def getFinalDelta(a_4, y):

    return np.subtract(a_4, y);

def getSubDelta2(thetaVector, deltaVector, activationVector):

    # print('getting shapes')
    # print(thetaVector.shape)
    # print(deltaVector.shape)
    # print(activationVector.shape)

    r = thetaVector.dot(deltaVector)

    # print(r.shape)
    # print(grad.shape)

    gPrime = np.multiply(r, sigmoidGradient(activationVector).transpose())

    # print(gPrime.shape)
    # print('')

    return gPrime;

def getSubDelta3(thetaVector, deltaVector, activationVector):

    r = thetaVector.dot(deltaVector.transpose())
    gPrime = np.multiply(r, sigmoidGradient(activationVector).transpose())



    return gPrime;

def squaredSumOfArray(array):
    sum = 0;

    for row in range(0, array.shape[0]):
        for i in range(0, len(array[row])):
            sum += math.pow(array[row][i], 2);

    return sum;

def costFunction(sumOfLabels, trainingSetSize, beginFile, layerOne, layerTwo, outputLayer):
    sumOfLabels *= (-1 / (trainingSetSize - beginFile));

    # sumOfLabels *= (-1 / 1);

    # # remove bias theta
    # np.delete(layerOne, [0], axis=1);
    # np.delete(layerTwo, [0], axis=1);
    # np.delete(outputLayer, [0], axis=1);

    layerOneSquaredSum = squaredSumOfArray(layerOne);
    layerTwoSquaredSum = squaredSumOfArray(layerTwo);
    outputLayerSquaredSum = squaredSumOfArray(outputLayer);

    networkSquaredSum = layerOneSquaredSum + layerTwoSquaredSum + outputLayerSquaredSum;

    networkSquaredSum *= (_lambda / (2 * (trainingSetSize - beginFile)));

    # networkSquaredSum *= (_lambda / (2))

    return sumOfLabels + networkSquaredSum;

def sigmoid(rawOutputVector):
    for row in range(0, rawOutputVector.shape[0]):
        for i in range(0, rawOutputVector.shape[1]):
            # try:
            rawOutputVector[row][i] = (1/(1+math.exp(-1 * rawOutputVector[row][i])))
            # except:
            #     rawOutputVector[row][i] = 1

    return rawOutputVector;

def sigmoidGradient(a):
    return np.multiply(a, np.subtract(1, a))

def forwardPropagation(inputVector, layerOne, layerTwo, outputLayer):
    return getOutputLayer(getThirdActivationLayer(getSecondActivationLayer(inputVector)));

def getTrainingDataFromFileWithIndex(index):
    return np.load(os.path.join('C:/Users', 'Wesley', 'Desktop', 'Re-Domum', 'training_set_processed', str(index)+'.bin.npy'));

def getTestDataFromFileWithIndex(index):
    return np.load(os.path.join('C:/Users', 'Wesley', 'Desktop', 'Re-Domum', 'test_set_processed', str(index)+'.bin.npy'));

def testLabelLookup(index):
    result = None;

    with open(os.path.join('C:/Users', 'Wesley', 'Desktop', 'Re-Domum', 'test_labels.txt')) as f:
        for line in f.readlines():
            if line.strip('\n').split(',')[0] == str(index):
                result = 'True' in line;
                break;

    if result:
        return np.array(([[1, 0]]));
    else:
        return np.array(([[0, 1]]));

def trainingLabelLookup(index):
    result = None;

    with open(os.path.join('C:/Users', 'Wesley', 'Desktop', 'Re-Domum', 'training_labels.txt')) as f:
        for line in f.readlines():
            if line.strip('\n').split(',')[0] == str(index):
                result = 'True' in line;
                break;

    if result:
        return np.array(([[1, 0]]));
    else:
        return np.array(([[0, 1]]));

def getLabelVector(label):
    tempArr = np.zeros((10))

    tempArr[int(label)] = 1

    return np.array(([tempArr]))


def nnIterator():
    # the number of iterations to take through the training set
    iterations = input('Number of iterations to perform? (default 50)');
    # the file to begin iterating from
    beginFile = input('File begin Index? (default 0)');

    # beginFile = 200;

    try:
        iterations = int(iterations);
        beginFile = int(beginFile)
    except:
        iterations = 500;
        beginFile = 0;

    for i in range(0, iterations):

        sumOfLabels = 0;

        DeltaOne = np.zeros(([imageHeight * imageWidth, firstLayerDepth]));
        DeltaTwo = np.zeros(([firstLayerDepth, secondLayerDepth]));
        DeltaThree = np.zeros(([secondLayerDepth, outputLayerDepth]));

        with open('C:/Users/Wesley/Desktop/handwritten/mnist_train.csv') as trainingSet:

            for line in trainingSet.readlines():
                # a_1 = np.array(([getTrainingDataFromFileWithIndex(index)]));
                # y = trainingLabelLookup(index);

                rawData = line.split(',')
                label = rawData.pop()

                a_1 = np.array(([rawData]), dtype='float64')

                y = getLabelVector(label);

                a_2 = getSecondActivationLayer(a_1);

                # print('a_2: {}'.format(a_2.shape))

                a_3 = getThirdActivationLayer(a_2);

                # print('a_3: {}'.format(a_3.shape))

                a_4 = getOutputLayer(a_3);

                # print('a_4: {}'.format(a_4.shape))

                delta_4 = getFinalDelta(a_4, y);

                # print('delta_4: {}'.format(delta_4.shape))

                delta_3 = getSubDelta3(theta_3, delta_4, (a_3));

                # print('delta_3: {}'.format(delta_3.shape))

                delta_2 = getSubDelta2(theta_2, delta_3, (a_2));

                # print('delta_2: {}'.format(delta_2.shape))

                DeltaThree = np.add(DeltaThree, delta_4.transpose().dot(a_3).transpose())

                # print('Delta: {}'.format(DeltaThree[0][0]))

                DeltaTwo = np.add(DeltaTwo, delta_3.dot(a_2).transpose())
                DeltaOne = np.add(DeltaOne, delta_2.dot(a_1).transpose())

                for row in range(0, y.shape[0]):
                    for j in range(0, len(y[row])):
                        sumOfLabels += ((y[row][j] * math.log10(a_4[row][j])) + ((1 - y[row][j]) * math.log10(1-a_4[row][j])))


                # print('Delta2: {}'.format(DeltaTwo))
                # print('Delta1: {}'.format(DeltaOne))

                # if index == 600:
                #     print('Delta3: {}'.format(DeltaThree[0][0]))
                #     print('Delta2: {}'.format(DeltaTwo[0][0]))
                #     print('Delta1: {}'.format(DeltaOne[0][0]))

            # print('sumOfLabels: {}'.format(sumOfLabels))
            _cost = costFunction(sumOfLabels, trainingSetSize, beginFile, theta_1, theta_2, theta_3);
            backPropagation(DeltaOne, DeltaTwo, DeltaThree, trainingSetSize - beginFile);

            global cost;
            global accuracy;

            # print("Accuracy: {}".format(test()*100));

            if float(cost) > float(_cost):
                cost = _cost;
                np.save(os.path.join("C:/Users", "Wesley", "Desktop", "Re-Domum", "theta_layer_one.bin"), theta_1);
                np.save(os.path.join("C:/Users", "Wesley", "Desktop", "Re-Domum", "theta_layer_two.bin"), theta_2);
                np.save(os.path.join("C:/Users", "Wesley", "Desktop", "Re-Domum", "theta_layer_three.bin"), theta_3);

                with open(os.path.join("C:/Users", "Wesley", "Desktop", "Re-Domum", "misc.txt"), 'r+') as miscFile:
                    miscFile.write(str(cost)+'\n');
                    miscFile.write(str(accuracy)+'\n');
                    miscFile.close();

            _accuracy = test() * 100;
            print('i: {} cost: {} with accuracy {}%'.format(i, _cost, _accuracy));

        # if float(cost) > float(_cost) and float(_accuracy) > float(accuracy):
        #
        #     cost = _cost;
        #     accuracy = _accuracy
        #
        #     np.save(os.path.join("C:/Users", "Wesley", "Desktop", "Re-Domum", "theta_layer_one.bin"), layerOne);
        #     np.save(os.path.join("C:/Users", "Wesley", "Desktop", "Re-Domum", "theta_layer_two.bin"), layerTwo);
        #     np.save(os.path.join("C:/Users", "Wesley", "Desktop", "Re-Domum", "theta_layer_three.bin"), finalLayer);
        #
        #     with open(os.path.join("C:/Users", "Wesley", "Desktop", "Re-Domum", "misc.txt"), 'r+') as miscFile:
        #         miscFile.write(cost);
        #         miscFile.write(str(accuracy));
        #         miscFile.close();
        #
        #     print('New optimum found: {} with accuracy {}%'.format( cost, accuracy));


def test():
    _accuracy = 0;

    # print('test set size: {}'.format(testSetSize))

    # for index in range(0, int(testSetSize)-1):
    #     testExample = np.array((getTestDataFromFileWithIndex(index)));
    #
    #     print(testExample[1])

    with open('C:/Users/Wesley/Desktop/handwritten/mnist_test.csv') as testSet:

        index = 0;

        for line in testSet.readlines():

            # inputVector = np.array(([getTestDataFromFileWithIndex(index)]));

            # labelVector = testLabelLookup(index);

            # print(line)

            data = line.split(',');

            # print(len(data))

            label = data.pop();

            labelVector = getLabelVector(label)

            inputVector = np.array(([data]), dtype='float64')


            # print(inputVector.shape)

            # predicatedVector = forwardPropagation(inputVector, theta_1, theta_2, theta_3);

            a_2 = getSecondActivationLayer(inputVector)

            # print('a_2: {}'.format(a_2))

            a_3 = getThirdActivationLayer(a_2)

            predicatedVector = getOutputLayer(a_3)

            # print('a_4: {}'.format(predicatedVector))

            # print(theta_1)
            # print(theta_2)
            # print(theta_3)

            # if index == 0:
            #     print('Predicted: {} : Label: {}'.format(predicatedVector, labelVector))
                # print('a_2: {}'.format(a_2))
                # print('a_3: {}'.format(a_3))
                # print('a_4: {}'.format(predicatedVector))
            # elif index == testSetSize/2:
                # print('Predicted: {} : Label: {}'.format(predicatedVector, labelVector))
                # print('a_2: {}'.format(a_2))
                # print('a_3: {}'.format(a_3))
                # print('a_4: {}'.format(predicatedVector))
            # elif index == (testSetSize-1):
                # print('Predicted: {} : Label: {}'.format(predicatedVector, labelVector))
                # print('a_2: {}'.format(a_2))
                # print('a_3: {}'.format(a_3))
                # print('a_4: {}'.format(predicatedVector))


            if predicatedVector[0][0] >= 0.5:
                predicatedVector[0][0] = 1
            else:
                predicatedVector[0][0] = 0;

            if predicatedVector[0][1] >= 0.5:
                predicatedVector[0][1] = 1;
            else:
                predicatedVector[0][1] = 0

            if ((predicatedVector == labelVector).sum() == 10):
                _accuracy += 1;

            # break;

            index+=1;

        return _accuracy/int(testSetSize);

init()

nnIterator()

# print(test())







