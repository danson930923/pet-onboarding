import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

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

#boxplot = sns.boxplot(x = "Survived", y= "Age", data = df)
# plt.show()

#P-value
df_anova = df[["PClass", "Survived"]]
grouped_anova = df_anova.groupby(["PClass"])
anova_three_one = stats.f_oneway(grouped_anova.get_group(3)["Survived"], grouped_anova.get_group(1)["Survived"])
#print anova_three_one

#Correlation
df_corr = df.corr()
#heatmap = sns.heatmap(data = df_corr)
#plt.show()

# #Linear Regression
# LinReg = LinearRegression()
# LinReg.fit(df[["Age"]], df["Survived"])
# Result = LinReg.predict(df[["Age"]])

# width = 12
# height = 10
# plt.figure(figsize=(width, height))
# #sns.regplot(x="Age", y="Survived", data=df)
# #plt.ylim(0,)
# #plt.show()

# #Poly Regression
# def PlotPolly(model,independent_variable,dependent_variabble, Name):
#     x_new = np.linspace(15, 55, 100)
#     y_new = model(x_new)

#     plt.plot(independent_variable,dependent_variabble,'.', x_new, y_new, '-')
#     plt.title('Age ~ Survived')
#     ax = plt.gca()
#     ax.set_facecolor((0.898, 0.898, 0.898))
#     fig = plt.gcf()
#     plt.xlabel(Name)
#     plt.ylabel('Survived')

#     plt.show()
# x = df['Age']
# y = df['Survived']
# f = np.polyfit(x, y, 5)
# p = np.poly1d(f)
# # PlotPolly(p,x,y, 'Age')

# #Distribute Plot
# plt.figure(figsize=(width, height))


# ax1 = sns.distplot(df['Survived'], hist=False, color="r", label="Actual Value")
# sns.distplot(Result, hist=False, color="b", label="Fitted Values" , ax=ax1)

# plt.title('Actual vs Fitted Values for Price')
# plt.xlabel('Age')
# plt.ylabel('Survived')

# #Model Evaluation
# Y = df['Survived']
# X = df.drop('Survived', axis=1)
# Age_train, Age_test, Survived_train, Survived_test = train_test_split(X, Y, test_size = 0.4, random_state = 1)
# Rcross=cross_val_score(LinReg,X[['PClass']], Y,cv=3)
# # print Rcross