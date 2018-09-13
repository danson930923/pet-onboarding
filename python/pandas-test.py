import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

# Read csv
df = pd.read_csv(open("./titanic.csv","rb"))

# Formatting Data
for i in df.index:
    if df.at[i, 'PClass'] == "1st":
        df.at[i, 'PClass']  = 1
    elif df.at[i, 'PClass']  == "2nd":
        df.at[i, 'PClass']  = 2
    elif df.at[i, 'PClass']  == "3rd":
        df.at[i, 'PClass']  = 3

# Replace missing records
df.dropna(subset=["PClass"], inplace = True)

age_mean = df["Age"].mean()
df["Age"].replace(np.nan, age_mean, inplace = True)

# Normalization
Age_zScore = (df["Age"]-df["Age"].mean())/df["Age"].std()

# Get Dummy
df = pd.get_dummies(df, columns=["Sex"])

#Descriptive Statistic
# print df.describe()

boxplot = sns.boxplot(x = "Survived", y= "Age", data = df)
# plt.show()

#P-value
df_anova = df[["PClass", "Survived"]]
grouped_anova = df_anova.groupby(["PClass"])
anova_three_one = stats.f_oneway(grouped_anova.get_group(3)["Survived"], grouped_anova.get_group(1)["Survived"])
#print anova_three_one

#Correlation
df_corr = df.corr()
heatmap = sns.heatmap(data = df_corr)
#plt.show()

#Linear Regression
LinReg = LinearRegression()
LinReg.fit(df[["PClass"]], df["Survived"])
Result = LinReg.predict(df[["PClass"]])

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="Age", y="Survived", data=df)
plt.ylim(0,)
plt.show()