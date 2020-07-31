#Load the dataset
import pandas as pd
df = pd.read_csv('Salary_Data.csv')
print(df)
print(df.shape)
#outliars
print(df.isna().sum())
#Genral info
print(df.describe())

#X = df['YearExperience'].values.reshape(-1,1)
X = df.iloc[:,0:1].values
Y = df.iloc[:,-1:].values
print("Feature Data",X)
print("Target Data",Y)

import matplotlib.pyplot as plt
plt.xlabel("Year of experience")
plt.ylabel("Salary")
plt.title("Salary Predication")
plt.scatter(X,Y)
#plt.show()

#training
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)
plt.scatter(X_train,Y_train,color ='g',marker='*',label='Training')
plt.scatter(X_test,Y_test,color ='r',marker='+',label='Testing')
plt.legend()
#plt.show()

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y_train)
print("Y-intercept:",model.intercept_)
print("Slope:",model.coef_)
print("Model Score:",model.score(X_test,Y_test))
Y_pred = model.predict(X)

plt.plot(X,Y_pred,color='k',label='Best fit line')
plt.legend()
plt.show()

x_years_input = eval(input("Enter no of yr for prediction"))
predicated_value = model.predict([[x_years_input]])
print(predicated_value)