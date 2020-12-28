import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from scipy import stats
Data=pd.read_csv("http://download.mlcc.google.com/mledu-datasets/california_housing_train.csv",sep=",",nrows=20)
print(Data)
Data.describe()
d=np.mean(Data)
print(d)
d=np.median(Data)
print(d)
d=stats.mode(Data)
print(d)
print(Data.cov())
print(Data.corr())
df=pd.DataFrame(Data)
df.plot()
x=df['housing_median_age'].head()
Data=df['total_rooms'].head()
plt.bar(x,Data);
x=df['total_bedrooms']
y=df['total_rooms']
plt.xlabel('total_bedrooms'); plt.ylabel('total_rooms');
plt.scatter(x,y)
Data=['housing_median_age','total_rooms','total_bedrooms','population','households']
data=[15,5612,1283,1015,472]
fig=plt.figure(figsize=(10,7))
plt.pie(data,labels=Data)
plt.show()
df.plot.box()
plt.boxplot(df['total_rooms'])
plt.show()




//2nd



import pandas as pd 
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
data=pd.read_csv("data.csv")
data.head() 
data.tail() 
print(data)
m1=data['Age'].mean() 
print(m1)
m2=data['Age'].median() 
print(m2)
m3=data['Age'].mode() 
print(m3)
x= data.iloc[:,:-1].values 
y= data.iloc[:,3].values
imputer= SimpleImputer(missing_values =np.nan, strategy='mean')
imputer= imputer.fit(x[:, 1:3])
x[:, 1:3]= imputer.transform(x[:, 1:3]) 
print(x)
print(y)
label_encoder_x=LabelEncoder()
x[:, 0]= label_encoder_x.fit_transform(x[:, 0]) 
labelencoder_Y =LabelEncoder()
y= labelencoder_Y.fit_transform(y) 
print(x)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)
print(x_train) 
print(x_test)
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train) 
x_test=st_x.transform(x_test) 
print(x_train)
print(x_test)
