import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

df = pd.read_csv("polyregressionGDP.csv")

x_data, y_data = (df["Year"].values, df["Value"].values)
def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y

beta_1 = 0.10
beta_2 = 1990.0

#logistic function
Y_pred = sigmoid(x_data, beta_1 , beta_2)

#plot initial prediction against datapoints
# plt.plot(x_data, Y_pred*15000000000000.)
# plt.plot(x_data, y_data, 'ro')

# Normalization
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)
print x_data
print max(x_data)
print xdata


popt, pcov = curve_fit(sigmoid, xdata, ydata)
print popt[0]
print pcov[1]
x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
# plt.show()

## Problem
# Logarithmic
# /home/jupyterlab/conda/lib/python3.6/site-packages/ipykernel_launcher.py:3: RuntimeWarning: invalid value encountered in log
#  This is separate from the ipykernel package so we can avoid doing imports until
# Works well on local

# Choose Model: Logistic ? Expotential ?