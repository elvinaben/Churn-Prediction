import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import xgboost as xgb
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve
import pickle

class DataHandler: 
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None        
        self.input_df = None
        self.output_df = None

    def loadData(self):
        self.data = pd.read_csv(self.file_path)
        
    def createInputOutput(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)


class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.createModel()
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5

    def dropColumns(self):
        self.input_data = self.input_data.drop(['Unnamed: 0', 'id', 'CustomerId', 'Surname'], axis=1)
        
    def splitData(self, test_size=0.2, random_state=26):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data, test_size=test_size, random_state=random_state)    

    def checkOutlierWithBoxplot(self, column):
        boxplot = self.x_train.boxplot(column=[column]) 
        plt.show()

    def findMedianFromTrainColumn(self, column):
        return np.median(self.x_train[column])    
    
    def fillingNAWithNumbers(self, columns, number):
        self.x_train[columns].fillna(number, inplace=True)
        self.x_test[columns].fillna(number, inplace=True)
    
    def binaryEncoding(self, encode):
        self.x_train.replace(encode, inplace=True)
        self.x_test.replace(encode, inplace=True)

    def oneHotEncodingGeography(self):
        columns = ['Geography']
        def encode_data(data):
            encoded_data = pd.get_dummies(data, columns=columns, dtype=int)
            encoded_data['Geography_Other'] = 0
            return encoded_data

        x_train_encoded = encode_data(self.x_train)
        x_test_encoded = encode_data(self.x_test)

        missing_cols = set(x_train_encoded.columns) - set(x_test_encoded.columns)
        for col in missing_cols:
            x_test_encoded[col] = 0

        x_test_encoded['Geography_Other'] = 0

        for col in x_test_encoded.columns:
            if col not in x_train_encoded.columns:
                x_test_encoded['Geography_Other'] += x_test_encoded[col]
                x_test_encoded.drop(columns=[col], inplace=True)

        self.x_train = x_train_encoded
        self.x_test = x_test_encoded
    
    def createModel(self):
        base_estimator = DecisionTreeClassifier(max_depth = 4)
        self.xg = xgb.XGBClassifier(estimator = base_estimator, n_estimators = 80, random_state=100)

    def trainModel(self):
        self.xg.fit(self.x_train, self.y_train)

    def makePrediction(self):
        self.y_predict = self.xg.predict(self.x_test)
     
    def createReport(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict, target_names=['0', '1']))
    
    def roc_auc(self):
        roc_auc = roc_auc_score(self.y_test, self.y_predict)
        print('AUC:', roc_auc)

    # def print(self):
    #     print(self.x_train.columns)

file_path = 'data_D.csv'  
data_handler = DataHandler(file_path)
data_handler.loadData()
data_handler.createInputOutput('churn')

input_df = data_handler.input_df
output_df = data_handler.output_df

model_handler = ModelHandler(input_df, output_df)
model_handler.dropColumns()
model_handler.splitData()

model_handler.checkOutlierWithBoxplot('CreditScore')

CreditScore_replace_na = model_handler.findMedianFromTrainColumn('CreditScore')
model_handler.fillingNAWithNumbers('CreditScore', CreditScore_replace_na)

encode = {"Gender": {"Male" : 1,"Female" : 0}}
model_handler.binaryEncoding(encode)

model_handler.oneHotEncodingGeography()   

# model_handler.print()

model_handler.trainModel()
model_handler.makePrediction()
model_handler.createReport()
model_handler.roc_auc()