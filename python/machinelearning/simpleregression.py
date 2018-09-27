import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

from sklearn import linear_model as linreg
from sklearn.metrics import r2_score

#
# Read in File
#
df = pd.read_csv("simpleregressionCO2.csv")

#
# Preprocessing
#
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]

#
# Diagrams
#
#viz.hist()
#plt.show()
# plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("FUELCONSUMPTION_COMB")
# plt.ylabel("Emission")
# plt.show()
# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()

#
# Simple Regression Model
#
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
linregmodel = linreg.LinearRegression()
trainx = np.asanyarray(train[['ENGINESIZE']])
trainy = np.asanyarray(train[['CO2EMISSIONS']])
linregmodel.fit(trainx, trainy)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(trainx, linregmodel.coef_[0][0]*trainx + linregmodel.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
# plt.show()

#
# Evaluation
#
testx = np.asanyarray(test[['ENGINESIZE']])
testy = np.asanyarray(test[['CO2EMISSIONS']])
testy_ = linregmodel.predict(testx)
print ("Mean absolute: %.2f" % np.mean(np.absolute(testy_ - testy)))
print ("MSE: %.2f" % np.mean((testy_ - testy) ** 2))
print ("R2: %.2f" % r2_score(testy_, testy))

# Did you know? When it comes to Machine Learning, you will likely be working with large datasets. As a business, where can you host your data? IBM is offering a unique opportunity for businesses, with 10 Tb of IBM Cloud Object Storage: Sign up now for free
# As it is for the advertisement. Font size can be larger?
# Evaluation: Root Mean Square Error (RMSE) Haven't been introduced in video