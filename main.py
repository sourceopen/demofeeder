# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json

import requests
import matplotlib.pyplot as plt
import pandas as pd
import tkinter
from tkinter import *
from tkinter import messagebox

import numpy as np
import ctypes


def train_model(trainingDataFileName, title, endpoint):
    #plt.xlabel('Sample No.')
    plt.ylabel('Value')

    # print("OSLO Block-wise prediction actual numbers...")
    print(title)
    plt.title(title+" yard occupancy prediction")

    dataframe = pd.read_csv(trainingDataFileName, header=None)
    data = dataframe.values
    X, y = data[:, :-1], data[:, -1:]

    req = requests.post(endpoint, json=data.tolist());
    if req.status_code == 201:
        jsonOfRetValues = json.loads(req.text)
        mape = jsonOfRetValues["mape"]
        rmse = jsonOfRetValues["rmse"]
        successMessage = "The model was created with RMSE = "+str(rmse)+" and MAPE = "+str(mape)
        ctypes.windll.user32.MessageBoxW(0, successMessage, "Success!", 1)
    elif req.status_code == 400:
        ctypes.windll.user32.MessageBoxW(0, "The model could not be created!", "Error!", 1)
        return;

def predict(referenceDataFile, inputForPredictionFile, title, endpoint):
    #plt.xlabel('Sample No.')
    plt.ylabel('Value')

    print(title)
    #plt.title('% Actual(red) Vs Prediction(blue)' + title)
    plt.title(title + " yard occupancy prediction")

    dataframe = pd.read_csv(referenceDataFile, header=None)
    actualData = dataframe.values
    X_actuals, y_actuals = actualData[:, :-1], (actualData[:, -1:]).tolist()

    dataframe = pd.read_csv(inputForPredictionFile, header=None)
    data = dataframe.values
    X, y = data[:, :-1], data[:, -1:]

    req = requests.post(endpoint, json=data.tolist());

    predictionsResp = json.loads(req.text)

    plt.xlim([0, 10])
    plt.ylim([0, 100])

    x = list(range(0, len(y_actuals)))
    plt.plot(x, y_actuals, 'r', label='Actuals')
    plt.plot(x, predictionsResp["predictions"], 'b', label='Predictions')

    plt.plot(x, y_actuals, 'xr')
    plt.plot(x, predictionsResp["predictions"], 'xb')

    plt.legend()
    mapeText = "MAPE = " + "{:.5f}".format(predictionsResp["mape"])
    rmseText = "RMSE = " + "{:.5f}".format(predictionsResp["rmse"])
    plt.text(-1.2, -5.5, mapeText+"\n"+rmseText, bbox=dict(facecolor='blue', alpha=0.5))


    for i in range(0,10):
        plt.annotate(i,(x[i], y_actuals[i][0]),textcoords="offset points", xytext=(0, 10), ha='center', color='r');
        plt.annotate(i, (x[i], predictionsResp["predictions"][i]), textcoords="offset points", xytext=(0, -20), ha='center', color='b');

    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.xaxis.set_ticks([])

    d = {'Number':range(1,11),'Actual':y_actuals, 'Prediction':predictionsResp["predictions"]}
    df = pd.DataFrame(d)
    table = plt.table(cellText=df.values, colLabels=df.columns, loc='bottom', cellLoc='center')

    plt.tight_layout()
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')



    plt.show()


if __name__ == '__main__':
    train_model('./resources/total_yard_percentage_scope_GEB.csv', 'Full yard occupancy percentage - GEBZE',
                'http://localhost:5007/api/CreateModelWithTrainingData?consumerId=FULL_YARD_PREDICTION_PERCENTAGE')

    predict('./resources/test_scope_GEB_with_labels.csv', './resources/test_scope_GEB_without_labels.csv',
            'GEBZE',
            'http://localhost:5007/api/PredictUsingModel?consumerId=FULL_YARD_PREDICTION_PERCENTAGE')

    train_model('./resources/total_yard_percentage_scope_OPA.csv', 'Full yard occupancy percentage - OPA',
                'http://localhost:5007/api/CreateModelWithTrainingData?consumerId=FULL_YARD_PREDICTION_PERCENTAGE')

    predict('./resources/test_scope_OPA_with_labels.csv', './resources/test_scope_OPA_without_labels.csv',
            'OPA',
            'http://localhost:5007/api/PredictUsingModel?consumerId=FULL_YARD_PREDICTION_PERCENTAGE')

    train_model('./resources/total_yard_percentage_scope_GEM.csv', 'Full yard occupancy percentage - GEMLIK',
                'http://localhost:5007/api/CreateModelWithTrainingData?consumerId=FULL_YARD_PREDICTION_PERCENTAGE')

    predict('./resources/test_scope_GEM_with_labels.csv', './resources/test_scope_GEM_without_labels.csv',
            'GEMLIK',
            'http://localhost:5007/api/PredictUsingModel?consumerId=FULL_YARD_PREDICTION_PERCENTAGE')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
