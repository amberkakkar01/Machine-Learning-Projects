#Logistic Regression
import pandas as pd
import numpy as np
'''column_name = ['Preg','Gluc','Skin','Insulin','BMI','Fun','Age','CLass']
df pd.read_csv('diabetes.csv',names=column_name)'''

df = pd.read_csv('diabetes.csv')
pd.set_option('display.max_columns',10,'display.width',1000)
print(df.head())
print(df.shape)

#Checking for null values
print(df.isna().sum())
print(df.describe())

import matplotlib.pyplot as plt
plt.figure(1)
df.hist()
plt.figure(2)
df.plot(kind ='density',subplots = True, layout =(3,3),sharex = False)
plt.figure(3)
df.plot(kind ='box',subplots = True, layout =(3,3),sharex = False)

#Multivaint Correlation
corr_mat = df.corr()
print(corr_mat)

fig = plt.figure(4)
ax = fig.add_subplot(111)
cax = ax.matshow(corr_mat, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(9)
ax.set_xticks(ticks)
ax.set_yticks(ticks)

plt.figure(5)
from pandas.plotting import scatter_matrix
scatter_matrix(df)
plt.show()

X = df.iloc[:,:-1].values
Y = df.iloc[:,-1].values

#traing & testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=7)

#model creation
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix
print("Accuracy Score:",accuracy_score(Y_test,Y_pred))
confusion_mat = confusion_matrix(Y_test,Y_pred)
print(confusion_mat)