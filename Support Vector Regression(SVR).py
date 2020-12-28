import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('Mall_Customers.csv')
data=data.head(7)
print(data)
x=data.iloc[:,3:4].values
y=data.iloc[:,4:].values
print(x,end=" ")
print(y,end=" ")



from sklearn.preprocessing import StandardScaler
st_x=StandardScaler()
st_y=StandardScaler()
X=st_x.fit_transform(x)
Y=st_y.fit_transform(y)
print(y)


fig=plt.figure()
f=fig.add_axes([0,0,1,1])
f.scatter(X,Y,color="black")
from sklearn.svm import SVR
regressor=SVR(kernel="rbf")
regressor.fit(X,Y)
plt.scatter(X,Y,color="black")
plt.plot(X,regressor.predict(X),color="blue")
