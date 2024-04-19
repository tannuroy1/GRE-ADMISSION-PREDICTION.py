import pandas as pd
df=pd.read_csv("E:\Admission_Predict.csv")
import numpy as np

df.head()

df.info()

df.isnull().sum()

df.isnull().sum().sum()

df.describe()

df.shape

df.duplicated().sum()

df.drop(columns=['Serial No.'],inplace=True)

df.head()

x=df.iloc[:,0:-1]
#y=all the rows with -1 column
y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

x_train

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

scaler

x_train_scaled=scaler.fit_transform(x_train)
x_train_scaled=scaler.fit_transform(x_test)

x_train_scaled

import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(7,activation='relu',input_dim=7))
model.add(Dense(7,activation='relu'))

model.add(Dense(1,activation='linear'))

model.summary()

model.compile(loss='mean_squared_error',optimizer='Adam')

history=model.fit(x_train_scaled,y_train,epochs=100,validation_split=0.2)

y_pred=model.predict(x_train_scaled)

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

model.save("weight.h5")
