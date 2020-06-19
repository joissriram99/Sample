import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt

df=pd.read_csv('C:/Users/User/Desktop/Datasets/Dataset/Train.csv')
df_te=pd.read_csv('C:/Users/User/Desktop/Datasets/Dataset/Test.csv')

X=df.iloc[:,1:-1].values
Y=df.iloc[:,-1].values
'''
X_t=df_te.iloc[:,1:].values
'''
from sklearn.impute import SimpleImputer
sc=SimpleImputer(missing_values=np.nan,strategy='mean')
X[:,[1,7,12,14,16,18]]=sc.fit_transform(X[:,[1,7,12,14,16,18]])
'''
X_t[:,[1,7,12,14,16,18]]=sc.fit_transform(X_t[:,[1,7,12,14,16,18]])
'''

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
cs=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[2,4,5,6,13])],remainder='passthrough')
X=cs.fit_transform(X)
'''
X_t=cs.fit_transform(X_t)
'''
le=LabelEncoder()

X[:,31]=le.fit_transform(X[:,31])
X[:,33]=le.fit_transform(X[:,33])
'''
X_t[:,31]=le.fit_transform(X_t[:,31])
X_t[:,33]=le.fit_transform(X_t[:,33])
'''
from sklearn.preprocessing import StandardScaler
se=StandardScaler()
X=se.fit_transform(X)
'''
X_t=se.transform(X_t)
'''


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X,Y)

y_pred=reg.predict(X)
print(y_pred)

from sklearn.metrics import r2_score
ac=r2_score(Y,y_pred)
print(ac*100)