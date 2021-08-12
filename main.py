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

from matplotlib.table import Table

def train_model_1(trainingDataFileName, title, endpoint):
    # plt.xlabel('Sample No.')
    # plt.ylabel('Value')

    # print("OSLO Block-wise prediction actual numbers...")
    # print(title)
    # plt.title(title + " yard occupancy prediction")

    dataframe = pd.read_csv(trainingDataFileName, header=None)
    dataframe = dataframe.drop(columns=dataframe.columns[4])
    data = dataframe.values
    X, y = data[:, :-1], data[:, -1:]

    req = requests.post("http://localhost:5007/api/CreateModelWithTrainingData?consumerId=FULL_YARD_PREDICTION_PERCENTAGE_TEST", json=data.tolist());
    if req.status_code == 201:
        jsonOfRetValues = json.loads(req.text)
        mape = jsonOfRetValues["mape"]
        rmse = jsonOfRetValues["rmse"]
        accuracy = jsonOfRetValues["percentageAccuracy"]



        #mape_3pt = "{:.3f}".format(mape)

    successMessage = ""
    if(rmse != None):
        successMessage = "The model was created with RMSE = " + "{:.3f}".format(rmse)
        ctypes.windll.user32.MessageBoxW(0, successMessage, "Success!", 1)
    if (accuracy != None):
        successMessage = "The model was created with Accuracy = " + str("{:.3f}".format(accuracy)) + "%"
        ctypes.windll.user32.MessageBoxW(0, successMessage, "Success!", 1)
    elif req.status_code == 400:
        ctypes.windll.user32.MessageBoxW(0, "The model could not be created!", "Error!", 1)
        return;


def predictYardOccpancy_1(referenceDataFile, inputForPredictionFile, title, endpoint):
    # plt.xlabel('Sample No.')
    plt.ylabel('Value')

    print(title)
    # plt.title('% Actual(red) Vs Prediction(blue)' + title)
    plt.title(title + " yard occupancy prediction")

    dataframe = pd.read_csv(referenceDataFile, header=None)
    dataframe = dataframe.drop(columns=dataframe.columns[4])
    actualData = dataframe.values
    X_actuals, y_actuals = actualData[:, :-1], (actualData[:, -1:]).tolist()

    dataframe = pd.read_csv(inputForPredictionFile, header=None)
    dataframe = dataframe.drop(columns=dataframe.columns[4])
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
    #mapeText = "MAPE = " + "{:.3f}".format(predictionsResp["mape"])
    #rmseText = "RMSE of model = " + "{:.3f}".format(predictionsResp["rmse"])
    #plt.text(-1.2, -5.5, rmseText , bbox=dict(facecolor='cyan', alpha=0.5))

    for i in range(0, 10):
        plt.annotate(i, (x[i], y_actuals[i][0]), textcoords="offset points", xytext=(0, 10), ha='center', color='r');
        plt.annotate(i, (x[i], predictionsResp["predictions"][i]), textcoords="offset points", xytext=(0, -20),
                     ha='center', color='b');

    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.xaxis.set_ticks([])

    d = {'Number': range(1, 11), 'Actual': y_actuals, 'Prediction': predictionsResp["predictions"]}
    df = pd.DataFrame(d)
    table = plt.table(cellText=df.values, colLabels=df.columns, loc='bottom', cellLoc='center')

    plt.tight_layout()
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.show()



def train_model(trainingDataFileName, title, endpoint):
    # plt.xlabel('Sample No.')
    # plt.ylabel('Value')

    # print("OSLO Block-wise prediction actual numbers...")
    # print(title)
    # plt.title(title + " yard occupancy prediction")

    dataframe = pd.read_csv(trainingDataFileName, header=None)
    data = dataframe.values
    X, y = data[:, :-1], data[:, -1:]

    req = requests.post(endpoint, json=data.tolist());
    if req.status_code == 201:
        jsonOfRetValues = json.loads(req.text)
        mape = jsonOfRetValues["mape"]
        rmse = jsonOfRetValues["rmse"]
        accuracy = jsonOfRetValues["percentageAccuracy"]



        #mape_3pt = "{:.3f}".format(mape)

    successMessage = ""
    if(rmse != None):
        successMessage = "The model was created with RMSE = " + "{:.3f}".format(rmse)
        ctypes.windll.user32.MessageBoxW(0, successMessage, "Success!", 1)
    if (accuracy != None):
        successMessage = "The model was created with Accuracy = " + str("{:.3f}".format(accuracy)) + "%"
        ctypes.windll.user32.MessageBoxW(0, successMessage, "Success!", 1)
    elif req.status_code == 400:
        ctypes.windll.user32.MessageBoxW(0, "The model could not be created!", "Error!", 1)
        return;


def predictYardOccpancy(referenceDataFile, inputForPredictionFile, title, endpoint):
    # plt.xlabel('Sample No.')
    plt.ylabel('Value')

    print(title)
    # plt.title('% Actual(red) Vs Prediction(blue)' + title)
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
    #mapeText = "MAPE = " + "{:.3f}".format(predictionsResp["mape"])
    #rmseText = "RMSE of model = " + "{:.3f}".format(predictionsResp["rmse"])
    #plt.text(-1.2, -5.5, rmseText , bbox=dict(facecolor='cyan', alpha=0.5))

    for i in range(0, 10):
        plt.annotate(i, (x[i], y_actuals[i][0]), textcoords="offset points", xytext=(0, 10), ha='center', color='r');
        plt.annotate(i, (x[i], predictionsResp["predictions"][i]), textcoords="offset points", xytext=(0, -20),
                     ha='center', color='b');

    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.xaxis.set_ticks([])

    d = {'Number': range(1, 11), 'Actual': y_actuals, 'Prediction': predictionsResp["predictions"]}
    df = pd.DataFrame(d)
    table = plt.table(cellText=df.values, colLabels=df.columns, loc='bottom', cellLoc='center')

    plt.tight_layout()
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.show()


def predictBinary(referenceDataFile, inputForPredictionFile, title, endpoint):
    fig, ax = plt.subplots()

    dataframe = pd.read_csv(referenceDataFile, header=None)
    actualData = dataframe.values
    X_actuals, y_actuals = actualData[:, :-1], (actualData[:, -1:]).tolist()

    dataframe = pd.read_csv(inputForPredictionFile, header=None)
    data = dataframe.values
    X, y = data[:, :-1], data[:, -1:]

    req = requests.post(endpoint, json=data.tolist());

    predictionsResp = json.loads(req.text)

    roll = (dataframe.iloc[:, 0]).tolist()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    #dataframe.columns = ['StudentID', 'Quiz I', 'Quiz II', 'Quiz III', 'Quiz IV']
    dataframe.columns = ['Quiz I', 'Quiz II', 'Quiz III', 'Quiz IV']
    dataframe['Actuals'] = y_actuals
    dataframe['Predictions'] = predictionsResp["predictions"]

    t = ax.table(cellText=dataframe.values, colLabels=dataframe.columns, loc='center',cellLoc='center')
    t.auto_set_font_size(False)
    t.set_fontsize(24)



    for key, val in t.get_celld().items():
        # call get_text() on Cell and again on returned Text
        # print(f'{key}\t{val.get_text().get_text()}')
        if (key[0] != 0 and key[1] != 4 and key[1] != 5):
            print(f'{key}\t{val.get_text().get_text()}')
            if (int(val.get_text().get_text()) <= 50):
                val.set_color('mistyrose')
            else:
                val.set_color('gainsboro')
        try:
            if key[1] == 5:
                if(val.get_text().get_text() == str(0.0)):
                        t[key[0],key[1]].set_color('r')
                        t[key[0], key[1]-1].set_color('r')
                #if int(float(val.get_text().get_text())) == 0:
                 #   val.set_color('r')
        except ValueError:
            pass

    fig.tight_layout()
    plt.tight_layout()
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    mapeText = "MAPE = " + "{:.3f}".format(predictionsResp["mape"])
    #rmseText = "RMSE = " + "{:.3f}".format(predictionsResp["rmse"])
    #plt.text(-1.2, -15.5, rmseText, bbox=dict(facecolor='cyan', alpha=0.5))

    #ax.text(0.08, 0.95, rmseText ,transform=ax.transAxes, fontsize=14,
     #       verticalalignment='top', bbox=dict(facecolor='cyan', alpha=0.5))
    plt.show()


def train_GEBZE():
    train_model('./resources/total_yard_percentage_scope_GEB.csv', 'Full yard occupancy percentage - GEBZE',
                'http://localhost:5007/api/CreateModelWithTrainingData?consumerId=FULL_YARD_PREDICTION_PERCENTAGE_GEBZE')

def predict_GEBZE():
    predictYardOccpancy('./resources/test_scope_GEB_with_labels.csv', './resources/test_scope_GEB_without_labels.csv',
            'GEBZE',
            'http://localhost:5007/api/PredictUsingModel?consumerId=FULL_YARD_PREDICTION_PERCENTAGE_GEBZE')

def train_OSLO():
    train_model('./resources/total_yard_percentage_scope_OPA.csv', 'Full yard occupancy percentage - OPA',
                'http://localhost:5007/api/CreateModelWithTrainingData?consumerId=FULL_YARD_PREDICTION_PERCENTAGE_OSLO')

def predict_OSLO():
    predictYardOccpancy('./resources/test_scope_OPA_with_labels.csv', './resources/test_scope_OPA_without_labels.csv',
                        'OPA',
                        'http://localhost:5007/api/PredictUsingModel?consumerId=FULL_YARD_PREDICTION_PERCENTAGE_OSLO')

def train_OSLO_1():
    train_model_1('./resources/total_yard_percentage_scope_OPA_1.csv', 'Full yard occupancy percentage - OPA',
                'http://localhost:5007/api/CreateModelWithTrainingData?consumerId=FULL_YARD_PREDICTION_PERCENTAGE_TEST')

def predict_OSLO_1():
    predictYardOccpancy_1('./resources/test_scope_OPA_with_labels_1.csv', './resources/test_scope_OPA_without_labels_1.csv',
                        'OPA',
                        'http://localhost:5007/api/PredictUsingModel?consumerId=FULL_YARD_PREDICTION_PERCENTAGE_TEST')


def train_GEMLIK():
    train_model('./resources/total_yard_percentage_scope_GEM.csv', 'Full yard occupancy percentage - GEMLIK',
                'http://localhost:5007/api/CreateModelWithTrainingData?consumerId=FULL_YARD_PREDICTION_PERCENTAGE_GEMLIK')

def predict_GEMLIK():
    predictYardOccpancy('./resources/test_scope_GEM_with_labels.csv', './resources/test_scope_GEM_without_labels.csv',
                        'GEMLIK',
                        'http://localhost:5007/api/PredictUsingModel?consumerId=FULL_YARD_PREDICTION_PERCENTAGE_GEMLIK')

def train_STUDENT():
    train_model('./resources/training_test_scores_2.csv', 'Test scores',
                'http://localhost:5007/api/CreateModelWithTrainingData?consumerId=TEST_SCORES')

def predict_STUDENT():
    predictBinary('./resources/ans_test_scores_2.csv', './resources/test_test_scores_2.csv',
                  'TestScores',
                  'http://localhost:5007/api/PredictUsingModel?consumerId=TEST_SCORES')

if __name__ == '__main__':
    root = Tk()
    root.geometry('700x450')

    xx = 0
    yy = 70

    btn_train_gebze = Button(root, text='Gebze Yard Occupancy %', bd='5', width=25,
                 command=lambda:train_GEBZE())
    btn_train_gebze.place(x=100+xx, y=130+yy)

    btn_train_opa = Button(root, text='Oslo Yard Occupancy %', bd='5', width=25,
                 command=lambda:train_OSLO())
    btn_train_opa.place(x=100+xx, y=180+yy)

    btn_train_gemlik = Button(root, text='Gemlik Yard Occupancy %', bd='5', width=25,
                 command=lambda: train_GEMLIK())
    btn_train_gemlik.place(x=100+xx, y=230+yy)

   # btn_train_student = Button(root, text='Student Evaluation', bd='5', width=25,
     #            command=lambda: train_STUDENT())
   # btn_train_student.place(x=100+xx, y=280+yy)

    btn_predict_gebze = Button(root, text='Gebze Yard Occupancy %', bd='5', width=25,
                 command=lambda:predict_GEBZE())
    btn_predict_gebze.place(x=400+xx, y=130+yy)

    btn_predict_opa = Button(root, text='Oslo Yard Occupancy %', bd='5', width=25,
                 command=lambda:predict_OSLO())
    btn_predict_opa.place(x=400+xx, y=180+yy)

    btn_predict_gemlik = Button(root, text='Gemlik Yard Occupancy %', bd='5', width=25,
                 command=lambda:predict_GEMLIK())
    btn_predict_gemlik.place(x=400+xx, y=230+yy)

   # btn_predict_student = Button(root, text='Predict Student', bd='5', width=25,
     #            command=lambda:predict_STUDENT())
   # btn_predict_student.place(x=400+xx, y=280+yy)

    root.title("ML Engine Demo Consumer")

    heading_label = Label(root, text="ML Engine Demo Consumer",font=("Verdana Bold", 25))
    heading_label.place(x=120+xx,y=-40+yy)

    training_label = Label(root, text="Train",font=("Verdana Bold", 20))
    training_label.place(x=140+xx,y=50+yy)

    predict_label = Label(root, text="Predict", font=("Verdana Bold", 20))
    predict_label.place(x=440+xx, y=50+yy)

    root.mainloop()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
