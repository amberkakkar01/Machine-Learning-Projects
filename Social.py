#Logistic Regression
import pandas as pd
import numpy as np
df = pd.read_csv('Social_Network_Ads.csv')
print(df.head())
print(df.shape)
#Checking for null values
print(df.isna().sum())
print(df.describe())

#Checking for outliars
import matplotlib.pyplot as plt
plt.figure(1)
plt.boxplot([df['Age']])
plt.figure(2)
plt.boxplot([df['EstimatedSalary']])
#plt.show()

#display total no of zero and one
class_count = df.groupby('Purchased').size()
print(class_count)
#or try this
len(df[df['Purchased'] == 0])

#Making two array aar1: age,salary :0 ; aar2:age,salary:1
arr1 = df[df['Purchased']==0].iloc[:,2:4]
arr2 = df[df['Purchased']==1].iloc[:,2:4]
#or use this
arr11 = df[df['Purchased'] == 1][['Age', 'EstimatedSalary']]
arr22 = df[df['Purchased'] == 0][['Age', 'EstimatedSalary']]

#Showing Clustring
plt.figure(3)
plt.scatter(arr1['Age'],arr1['EstimatedSalary'], color = 'r',marker='*')
plt.scatter(arr2['Age'],arr2['EstimatedSalary'], color = 'k',marker='+')
plt.xticks(np.arange(10,100,10))
plt.xlabel("Age")
plt.ylabel('Salary')
plt.show()

X = df.iloc[:, 2:4].values
Y = df.iloc[:,-1].values

#traing & testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=7)

#model creation
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,Y_pred))

from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(Y_test,Y_pred)
print(confusion_mat)