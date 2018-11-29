import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import load_model
from sklearn.externals import joblib 
import datetime

maxSequenceLength = 397
model = None
circle = [[0.0,0.0],[0.0,-3.0],[0.0,-4.0],[-1.0,-3.0],[-2.0,-3.0],[-3.0,-2.0],[-3.0,-3.0],[-3.0,-2.0],[-2.0,-1.0],[-3.0,0.0],[-2.0,0.0],[-4.0,0.0],[-3.0,0.0],[-6.0,2.0],[-2.0,1.0],[-3.0,3.0],[0.0,1.0],[-1.0,2.0],[0.0,2.0],[0.0,5.0],[0.0,2.0],[0.0,3.0],[2.0,4.0],[2.0,3.0],[2.0,3.0],[2.0,1.0],[2.0,0.0],[5.0,1.0],[2.0,0.0],[2.0,0.0],[2.0,0.0],[4.0,0.0],[3.0,0.0],[2.0,0.0],[1.0,-1.0],[1.0,-1.0],[2.0,-2.0],[1.0,-1.0],[1.0,-2.0],[1.0,-1.0],[1.0,-2.0],[1.0,-1.0],[2.0,-2.0],[1.0,-1.0]]
rect = [[0.0,0.0],[1.0,3.0],[0.0,4.0],[3.0,2.0],[0.0,1.0],[0.0,2.0],[0.0,6.0],[0.0,2.0],[0.0,2.0],[0.0,1.0],[0.0,1.0],[0.0,2.0],[1.0,4.0],[0.0,2.0],[3.0,4.0],[0.0,3.0],[0.0,2.0],[0.0,3.0],[0.0,3.0],[0.0,1.0],[0.0,2.0],[0.0,1.0],[0.0,2.0],[0.0,3.0],[0.0,7.0],[0.0,2.0],[0.0,3.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,2.0],[0.0,2.0],[-1.0,2.0],[0.0,2.0],[-1.0,4.0],[-1.0,0.0],[-1.0,3.0],[0.0,3.0],[0.0,3.0],[0.0,4.0],[-2.0,0.0],[-1.0,0.0],[-1.0,0.0],[-4.0,-2.0],[-2.0,-1.0],[-4.0,0.0],[-1.0,0.0],[-2.0,0.0],[-2.0,0.0],[-2.0,0.0],[-2.0,0.0],[-2.0,-2.0],[-4.0,-1.0],[-2.0,0.0],[-1.0,0.0],[-2.0,0.0],[-4.0,0.0],[-2.0,-1.0],[-1.0,-2.0],[-4.0,0.0],[-5.0,-3.0],[-4.0,0.0],[-1.0,-2.0],[-4.0,-1.0],[-4.0,-2.0],[-5.0,-1.0],[-5.0,-2.0],[-1.0,-1.0],[-5.0,-1.0],[-4.0,0.0],[-2.0,-1.0],[-2.0,-1.0],[-4.0,0.0],[-2.0,0.0],[0.0,-1.0],[0.0,-4.0],[1.0,-2.0],[0.0,-4.0],[0.0,-2.0],[0.0,-4.0],[1.0,-2.0],[0.0,-2.0],[2.0,-4.0],[1.0,-3.0],[0.0,-1.0],[4.0,-4.0],[0.0,-3.0],[1.0,-2.0],[2.0,-2.0],[0.0,-1.0],[2.0,-1.0],[2.0,-2.0],[2.0,-2.0],[0.0,-1.0],[0.0,-1.0],[0.0,-2.0],[0.0,-2.0],[0.0,-2.0],[0.0,-2.0],[0.0,-4.0],[0.0,-1.0],[0.0,-2.0],[0.0,-2.0],[1.0,-1.0],[1.0,-2.0],[2.0,-1.0],[0.0,-4.0],[2.0,-2.0],[0.0,-1.0],[0.0,-1.0],[1.0,-2.0],[1.0,-1.0],[2.0,-1.0],[0.0,-2.0],[2.0,-1.0],[1.0,-1.0],[0.0,-4.0],[1.0,0.0],[4.0,0.0],[1.0,0.0],[2.0,0.0],[3.0,0.0],[2.0,0.0],[3.0,0.0],[2.0,0.0],[4.0,0.0],[4.0,0.0],[3.0,0.0],[4.0,1.0],[3.0,0.0],[4.0,0.0],[1.0,0.0],[2.0,0.0],[4.0,0.0],[2.0,0.0],[4.0,0.0],[4.0,2.0],[2.0,0.0],[4.0,0.0],[5.0,0.0],[2.0,0.0],[4.0,0.0],[4.0,2.0],[1.0,0.0],[2.0,2.0],[0.0,2.0],[0.0,1.0],[0.0,4.0],[0.0,2.0],[0.0,4.0],[0.0,2.0],[0.0,2.0],[0.0,1.0],[0.0,2.0],[0.0,1.0]]
line = [[0.0,0.0],[11.0,5.0],[8.0,5.0],[13.0,6.0],[9.0,3.0],[19.9,9.0],[12.0,5.0],[22.9,10.9],[10.0,6.0],[19.9,12.9],[12.0,7.0],[19.9,10.0],[13.0,7.0],[15.9,9.0],[15.9,9.0],[14.0,7.0],[14.0,8.0]]
circle_label = [1,0,0]
rect_label = [0,0,1]
line_label = [0,1,0]
testshapes = [("Circle", circle), ("Rectangle", rect), ("Line", line)]

def preProcessInput(datapoint):
    if(len(datapoint) > maxSequenceLength):
        return np.array([datapoint[0:maxSequenceLength]])
    else:
        for i in range(len(datapoint), maxSequenceLength):
            datapoint.append([0,0])
        return np.array([datapoint])

def loadModel():
    try:
        model = load_model('models/shapedetection_model.h5')
    except Exception as e:
        print(e)
        exit(0)
    return model

def predict(model, testshapes):

    sum_cross_entropy = 0
    for shape in testshapes:
        print("Start prediction...")
        X_test = preProcessInput(shape[1])
        X_test = np.reshape(X_test, (1, maxSequenceLength, 2))
        start = datetime.datetime.now()
        pred = model.predict(X_test)[0]
        end = datetime.datetime.now()
        delta = end - start

        print("Speed (in ms): " + str(delta.total_seconds() * 1000))

        # Calculate cross entropy
        if(shape[0] == "Circle"):
            sum_cross_entropy = sum_cross_entropy + -(math.log(pred[0])*circle_label[0] + math.log(pred[1])*circle_label[1] + math.log(pred[2])*circle_label[2])
        if(shape[0] == "Rectangle"):
            sum_cross_entropy = sum_cross_entropy + -(math.log(pred[0])*rect_label[0] + math.log(pred[1])*rect_label[1] + math.log(pred[2])*rect_label[2])
        if(shape[0] == "Line"):
            sum_cross_entropy = sum_cross_entropy + -(math.log(pred[0])*line_label[0] + math.log(pred[1])*line_label[1] + math.log(pred[2])*line_label[2])

        # Print prediction
        if(pred[0] > pred[1] and pred[0] > pred[2]):
            print("Ground truth: "+shape[0]+". Prediction: Circle. Score: " + str(pred[0]))
        if(pred[1] > pred[0] and pred[1] > pred[2]):
            print("Ground truth: "+shape[0]+". Prediction: Line. Score: " + str(pred[1]))
        if(pred[2] > pred[0] and pred[2] > pred[1]):
            print("Ground truth: "+shape[0]+". Prediction: Rectangle. Score: " + str(pred[2]))

    average_cross_entropy = sum_cross_entropy / len(testshapes)
    print("ACE: " + str(average_cross_entropy))

def main():
    model = loadModel()
    predict(model, testshapes)
  
if __name__== "__main__":
  main()
