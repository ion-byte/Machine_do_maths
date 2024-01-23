#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""In this code block, data from cars is evaluated. This code is designed to use linear regression with multiple 
feature variables (Engine Size, Cylinders, Fuel Consumption) used to predict a target variable (Co2 Emissions) """

#libraries needed for the analysis

import matplotlib.pyplot as plt                      
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#pandas feature used to view the first five rows of the data set that contains 1068

df = pd.read_csv("FuelConsumptionCo2.csv")          
# take a look at the dataset
df.head()


# In[3]:


#Using pandas to create a new data frame for the analysis and viewing the first 9 rows

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)                        


# In[4]:


#Full data set plot using two variables to have a general look at the spread of the data.

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='green')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# In[5]:


#Here the shaped data frame is split randomly into 80% training data (used to train the model!) and 20% testing data.

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# In[6]:


#Plotting the same two variable scatter for the training data that will be used to compare to the full set in the
#graph above

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# In[7]:


#Training data regression - creating the model - and generating the 'weights' of the feature variables 
#which are 'engine size', 'cylinders' and 'fuel consumption' against the real numbers for the target variable
#of 'Co2 emissions'

from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)


# In[8]:


#Final Step: Testing the model created in the code block above with the 20% of data left for testing.  This uses the weights calculated
#for the feature variables to predict the 'Co2 emissions' target variable - and then calculate how accurate the training
#model is at predicting the actual test numbers.

y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])

#The lower the residual sum of squares value statistically, the better a model fits the data.
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))


# In[ ]:




