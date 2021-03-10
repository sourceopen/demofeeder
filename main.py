# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json

import requests
import matplotlib.pyplot as plt
import pandas as pd

def train_model(filename, desc, req):
    plt.xlabel('Sample No.')
    plt.ylabel('Value')

    # print("OSLO Block-wise prediction actual numbers...")
    print(desc)
    plt.title('% Actual(red) Vs Predicted(blue)' + desc)

    dataframe = pd.read_csv(filename, header=None)
    data = dataframe.values
    X, y = data[:, :-1], data[:, -1:]

    t = requests.post(req, json=data.tolist());

    jsonOfRetValues = json.loads(t.text)

    plt.xlim([0, 10])
    plt.ylim([0, 100])

    x = list(range(0, len(jsonOfRetValues["actuals"])))
    plt.plot(x, jsonOfRetValues["actuals"], 'r')
    plt.plot(x, jsonOfRetValues["predictions"], 'b')

    plt.show()

def predict(testFileName, title, endpoint):
    plt.xlabel('Sample No.')
    plt.ylabel('Value')

    # print("OSLO Block-wise prediction actual numbers...")
    print(title)
    plt.title('% Actual(red) Vs Predicted(blue)' + title)

    dataframe = pd.read_csv(testFileName, header=None)
    data = dataframe.values
    X, y = data[:, :-1], data[:, -1:]

    t = requests.post(endpoint, json=data.tolist());

    jsonOfRetValues = json.loads(t.text)

    plt.xlim([0, 10])
    plt.ylim([0, 100])

    x = list(range(0, len(jsonOfRetValues["actuals"])))
    plt.plot(x, jsonOfRetValues["actuals"], 'r')
    plt.plot(x, jsonOfRetValues["predictions"], 'b')

    plt.show()


if __name__ == '__main__':
    train_model('./resources/total_yard_percentage_scope_4.csv', 'Full yard occupancy percentage - GEBZE',
                'http://localhost:5009/api/CreateModelWithTrainingData?consumerId=FULL_YARD_PREDICTION_PERCENTAGE')
    train_model('./resources/total_yard_percentage_scope_2.csv', 'Full yard occupancy percentage - OPA',
                'http://localhost:5009/api/CreateModelWithTrainingData?consumerId=FULL_YARD_PREDICTION_PERCENTAGE')
    train_model('./resources/total_yard_percentage_scope_5.csv', 'Full yard occupancy percentage - GEMLIK',
                 'http://localhost:5009/api/CreateModelWithTrainingData?consumerId=FULL_YARD_PREDICTION_PERCENTAGE')

    predict('./resources/test_scope_2.csv', 'Test Full yard occupancy percentage - OPA',
            'http://localhost:5009/api/PredictUsingModel?consumerId=FULL_YARD_PREDICTION_PERCENTAGE')

    predict('./resources/test_scope_4.csv', 'Test Full yard occupancy percentage - GEBZE',
            'http://localhost:5009/api/PredictUsingModel?consumerId=FULL_YARD_PREDICTION_PERCENTAGE')

    predict('./resources/test_scope_5.csv', 'Test Full yard occupancy percentage - GEMLIK',
                'http://localhost:5009/api/PredictUsingModel?consumerId=FULL_YARD_PREDICTION_PERCENTAGE')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
