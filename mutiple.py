import pandas as pd
import matplotlib.pyplot as plt
d=pd.read_csv("Mydata1.csv")
d.head()
d.corr()



x=d[['TV','Radio','Newspaper']]
y=d['Sales']
print(x)
print(y)



from sklearn import linear_model
reg=linear_model.LinearRegression()
reg.fit(x,y)
pred=reg.predict([[38.2,3.7,13.8]])
print(pred)
print(reg.coef_)
print(reg.intercept_)
from sklearn.metrics import r2_score
reg.score(x,y)
