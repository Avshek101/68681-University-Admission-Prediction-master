#Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Read the dataset with pandas
dataset=pd.read_csv(r'C:\Users\saairaam prasad\Desktop\remote internship-2020\project\Admission_Predict_Ver1.1.csv')
dataset.head()
#Drop unwanted Columns
dataset.drop(['Serial No.'],axis=1,inplace=True)
dataset

x=dataset.iloc[:,:7]
x
x=dataset.iloc[:,:7].values

y=dataset.iloc[:,-1:]
y.values

#Label Encoding
from sklearn.preprocessing import LabelEncoder

def object_to_int(dataframe_series):
    if dataframe_series.dtype=='object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series
dataset = dataset.apply(lambda x: object_to_int(x))
dataset.head()

#Split the dataset into Train,Test, Split
from sklearn.model_selection import train_test_split
x_test,x_train,y_test,y_train=train_test_split(x,y,test_size=0.2,random_state=0)

#Build a model Unsing Linear Regression

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
#Prediction
reg.fit(x_train,y_train)
y_predict =reg.predict(x_test)

y_predict
#save the file
import pickle
pickle.dump(reg,open('concrete.pkl','wb'))
model=pickle.load(open('concrete.pkl','rb'))