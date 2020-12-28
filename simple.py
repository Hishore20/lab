import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("Mydata1.csv")
data["TV"].describe()
data.boxplot()




plt.scatter(data["Sales"],data["TV"])




data.corr()
x=data.iloc[:,0:3].values
y=data.iloc[:,3].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train)
print(y_train)
from sklearn.linear_model import LinearRegression
r=LinearRegression()
r.fit(x_train,y_train)
r.coef_
r.intercept_
y_pred=r.predict(x_test)
print(y_pred)
from sklearn.metrics import r2_score
r.score(x_train,y_train)
