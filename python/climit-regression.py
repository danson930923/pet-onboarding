import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

#
# Read csv
#
df = pd.read_csv(open("./weather.csv","rb"))

#
# Replace missing records
#
def Remove_nan(df):
    nan_in_df = {}
    for col in df:
        nan_in_df[col] = df[col].isna().sum()

    # # Too many NAN in the column, not worth for the classification
    remain_nan_in_df = []
    for col in nan_in_df:
        if nan_in_df[col] > (len(nan_in_df)*0.2):
            df.drop(col, axis=1, inplace=True)
        else:
            remain_nan_in_df.append(col)

    for col in remain_nan_in_df:
        if col in (df.select_dtypes(include=['float64']).columns) or col in (df.select_dtypes(include=['int64']).columns):
            col_mean = df[col].mean()
            df[col].replace(np.nan, col_mean, inplace=True)
        else:
            df.dropna(subset=[col], inplace=True)
    return df

df = Remove_nan(df)

#
# Correlation
#
def Get_Correlation(df):
    df_corr = df.corr()
    print df_corr
    heatmap = sns.heatmap(data = df_corr)
    plt.show()

#df = Get_Correlation(df)

#
# Humidity
#
def Show_Boxplot_Figure(df, col):
    boxplot = sns.boxplot(y=df[col])
    plt.show()

#Show_Boxplot_Figure(df, "Humidity")

width = 12
height = 10
#
# Linear Regression - Order 1
#
def Get_Linear_Regression(x_axis, y_axis, df):
    linReg = LinearRegression()
    linReg.fit(df[[x_axis]], df[y_axis])
    result = linReg.predict(df[[x_axis]])
    return result

linResult = Get_Linear_Regression("Interval Rain", "Humidity", df)

def Show_Linear_Regression_Figure(x_axis, y_axis, df):
    Get_Linear_Regression(df)

    plt.figure(figsize=(width, height))
    sns.regplot(x=x_axis, y=y_axis, data=df)
    plt.ylim(0,120)
    plt.show()

#Show_Linear_Regression_Figure("Interval Rain", "Humidity", df)

#
# Poly Regression
#
def Get_Poly_Regression(x_axis, y_axis, order, df):
    independent_variable = df[x_axis]
    dependent_variable = df[y_axis]
    f = np.polyfit(independent_variable, dependent_variable, order)
    model = np.poly1d(f)
    return model

polyResult = Get_Poly_Regression('Interval Rain', 'Humidity', 8, df)

def Show_Poly_Regression_Figure(x_axis, y_axis, order, df):
    Get_Poly_Regression(x_axis, y_axis, order, df)

    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable,dependent_variable,'.', x_new, y_new, '-')
    plt.title(x_axis + " ~ " + y_axis)
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    plt.show()

#Show_Poly_Regression_Figure('Interval Rain', 'Humidity', 3, df)

#
# Distribute Plot
#
def Show_Distribute_Figure(x_axis, y_axis, order, df):
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(df[y_axis], hist=False, color="r", label="Actual Value")
    sns.distplot(Get_Poly_Regression(x_axis, y_axis, order, df), hist=False, color="b", label="Fitted Values" , ax=ax1)

    plt.title('Actual vs Fitted Values for Humidity')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.show()

#Show_Distribute_Figure('Interval Rain', 'Humidity', 8, df)

#
# Model Evaluation
#
def Print_Model_Evaluation_Score(x_axis, y_axis, df):
    linReg = LinearRegression()
    linReg.fit(df[[x_axis]], df[y_axis])
    Y = df[y_axis]
    X = df.drop(y_axis, axis=1)
    Rain_train, Rain_test, Hum_train, Hum_test = train_test_split(X, Y, test_size = 0.4, random_state = 1)
    Rcross=cross_val_score(linReg,X[[x_axis]], Y,cv=3)
    print Rcross

#Print_Model_Evaluation_Score("Interval Rain", "Humidity", df)

#
# Ridge Regression
#
def Show_Ridge_Regression_Rsquare_Result(df):
    Y = df['Humidity']
    X = df.drop('Humidity', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4, random_state = 1)
    pr=PolynomialFeatures(degree=2)
    x_train_pr=pr.fit_transform(x_train[['Interval Rain', 'Wind Direction', 'Wind Speed', 'Maximum Wind Speed', 'Solar Radiation', 'Battery Life']])
    x_test_pr=pr.fit_transform(x_test[['Interval Rain', 'Wind Direction', 'Wind Speed', 'Maximum Wind Speed', 'Solar Radiation', 'Battery Life']])
    RidgeModel=Ridge(alpha=0.1)
    RidgeModel.fit(x_train_pr,y_train)
    yhat=RidgeModel.predict(x_test_pr)

    Rsqu_test=[]
    Rsqu_train=[]
    dummy1=[]
    ALFA=5000*np.array(range(0,10000))
    for alfa in ALFA:
        RidgeModel=Ridge(alpha=alfa) 
        RidgeModel.fit(x_train_pr,y_train)
        Rsqu_test.append(RidgeModel.score(x_test_pr,y_test))
        Rsqu_train.append(RidgeModel.score(x_train_pr,y_train))

    plt.figure(figsize=(width, height))

    plt.plot(ALFA,Rsqu_test,label='validation data  ')
    plt.plot(ALFA,Rsqu_train,'r',label='training Data ')
    plt.xlabel('alpha')
    plt.ylabel('R^2')
    plt.legend()
    plt.show()

#Show_Ridge_Regression_Rsquare_Result(df)

#
# Grid Search
#
def Grid_Search_Score(df):
    Y = df['Humidity']
    X = df.drop('Humidity', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4, random_state = 1)
    parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000,10000,100000,100000]}]
    RR=Ridge()
    Grid1 = GridSearchCV(RR, parameters1,cv=4)
    Grid1.fit(x_train[['Interval Rain', 'Wind Direction', 'Wind Speed', 'Maximum Wind Speed', 'Solar Radiation', 'Battery Life']],y_train)
    BestRR=Grid1.best_estimator_
    print BestRR.score(x_test[['Interval Rain', 'Wind Direction', 'Wind Speed', 'Maximum Wind Speed', 'Solar Radiation', 'Battery Life']],y_test)

Grid_Search_Score(df)
