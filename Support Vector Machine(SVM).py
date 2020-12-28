import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
data=pd.read_csv("1.csv")
print(data)
x=data.iloc[:,2:4].values
y=data.iloc[:,-1:].values
print(x)
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
from sklearn.svm import SVC
sv=SVC(kernel='linear')
sv.fit(x_train,y_train)
y_pred=sv.predict(x_test)
print(y_pred)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
