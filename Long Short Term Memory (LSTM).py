#!/usr/bin/env python
# coding: utf-8

# In[90]:


# Program that Use artificial recurrent neural network LSTM 
# to predict closing price of any stock
# Credit to https://www.youtube.com/watch?v=QIUxPv5PJOY Computer Science Youtube Tutorial
# Author: Ellis Sentoso


# In[14]:


import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[67]:


# Change arg 1 to any stock symbol
# Pick any desired starting date and ending date.
df = web.DataReader('ADA-USD', data_source = 'yahoo', start = '2019-01-01', end = '2021-12-30')
df


# In[68]:


df.shape


# In[69]:


# Visualization
plt.figure(figsize = (16, 8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize = 20)
plt.ylabel('Close Price USD ($)', fontsize=20)
plt.show()


# In[70]:


data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset) * 0.80) # training data is 80% of the data
training_data_len


# In[71]:


# scaling the data
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data


# In[72]:


# Training the data and create the scaled training dataset
train_data = scaled_data[0:training_data_len,:]
#Split the data into xtrain and ytrain datasets
xtrain = []
ytrain = []
for i in range(60, len(train_data)):
    xtrain.append(train_data[i-60:i, 0])
    ytrain.append(train_data[i, 0])
    if i <= 60:
        print(xtrain)
        print(ytrain)
        print()


# In[73]:


# Cover xtrain & ytrain to arrays.
xtrain, ytrain = np.array(xtrain), np.array(ytrain)


# In[74]:


# Reshape the data.
xtrain.shape
xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))
xtrain.shape


# In[75]:


# LSTM Modeling
mod = Sequential()
mod.add(LSTM(50, return_sequences = True, input_shape = (xtrain.shape[1],1)))
mod.add(LSTM(50, return_sequences = False))
mod.add(Dense(25))
mod.add(Dense(1))


# In[76]:


# Compile the model.
mod.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[77]:


# Traint the model.
mod.fit(xtrain,ytrain,batch_size=1,epochs=1)


# In[78]:


#create test dataset and array containing the scales values 
test_data = scaled_data[training_data_len-60: , :]
#create the datasets xtest and ytest.
xtest = []
ytest = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    xtest.append(test_data[i-60:i,0])


# In[79]:


# Convert data to array.
xtest = np.array(xtest)


# In[80]:


# Reshape data.
xtest = np.reshape(xtest, (xtest.shape[0],xtest.shape[1],1))


# In[81]:


# Predicted price by the model
predictions = mod.predict(xtest)
predictions = scaler.inverse_transform(predictions)


# In[82]:


# RMSE
rmse = np.sqrt(np.mean(predictions - ytest)**2)
rmse


# In[83]:


# Visualization
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title("Modeling")
plt.xlabel('Date',fontsize =20)
plt.ylabel('Close Price USD', fontsize =20)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', "Value", "Predictions"], loc = 'upper right')


# In[84]:


valid


# In[88]:


# Get the quote
stock = web.DataReader('ADA-USD', data_source = 'yahoo', start = '2019-01-01', end = '2021-12-30')
newdf = stock.filter(['Close'])
#get 60 day closing price and convert it to an array
day60 = newdf[-60:].values
scaledday60 = scaler.transform(day60)
xtest = []
xtest.append(scaledday60)
# convert xtest to an array
xtest = np.array(xtest)
xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))
# Predicted scaled price
pred_scaled = mod.predict(xtest)
# revert scaling
pred_scaled = scaler.inverse_transform(pred_scaled)
print(pred_scaled)


# In[89]:


stock2 = web.DataReader('ADA-USD', data_source = 'yahoo', start = '2021-12-30', end = '2021-12-30')
print(stock2['Close'])


# In[ ]:





# In[ ]:




