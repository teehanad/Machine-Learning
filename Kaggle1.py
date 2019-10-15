import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression


#Import the dataset
#Print some useful information about the data to see what we are dealing with
dataset = pd.read_csv("tcd ml 2019-20 income prediction training (with labels).csv")
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
dataset.info()
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print(dataset.head(10))
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print(dataset.describe())
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print(dataset.shape)
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
#Clean up null values in data
dataset.isnull().any()
dataset = dataset.fillna(method='ffill')


#Import the test data
testDataset = pd.read_csv("tcd ml 2019-20 income prediction test (without labels).csv")
testDataset = testDataset.fillna(method='ffill')

#Set the columns of the training data to the ones that we want to test
X_train = dataset[["Year of Record", "Age", "Size of City", "Wears Glasses", "Body Height [cm]"]].values
y_train = dataset["Income in EUR"].values
X_test = testDataset[["Year of Record", "Age", "Size of City", "Wears Glasses", "Body Height [cm]"]].values

#Train the data for Linear Regression 
regressor = LinearRegression()  
regressor.fit(X_train, y_train)

#Predict y values (Income)
y_pred = regressor.predict(X_test)

#Output vlaues to a csv
output = pd.read_csv("tcd ml 2019-20 income prediction submission file.csv")
output["Income"] = y_pred
output.to_csv("tcd ml 2019-20 income prediction submission file.csv", index=False)