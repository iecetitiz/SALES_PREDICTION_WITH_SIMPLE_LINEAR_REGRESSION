# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 22:16:09 2023

@author: iecet
"""

######################################################
# Sales Prediction with Linear Regression
######################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


# Simple Linear Regression 

df = pd.read_csv("advertising.csv")


X = df[["TV"]] 
y = df[["sales"]]

#Model

reg_model = LinearRegression().fit(X, y)

b = reg_model.intercept_[0] #bias
w = reg_model.coef_[0][0] #coefficient/weight

#Prediction Example
#for tv = 150
#sales = b + w * 150

sales_prediction = b + w * 150 

#Model Visualization
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")

g.set_title(f"Model Equatopm: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Number of Sales")
g.set_xlabel("TV Expenditure")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()


# Prediction Performance Evaluation

#MSE
y_pred = reg_model.predict(X)
mse = mean_squared_error(y, y_pred)


#RMSE
rmse = np.sqrt(mean_squared_error(y, y_pred))

#MAE
mae = mean_absolute_error(y, y_pred)


#R-Square
rs= reg_model.score(X, y)





