#Load the dataset
import pandas as pd
pd.set_option('display.max_columns',10,'display.width',1000)
df = pd.read_csv('50_Startups.csv')
print(df)
df2 = df
print("--"*50)
#Encoding form sklearn
#ONEHOT Encoding
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
df3 = pd.DataFrame(onehotencoder.fit_transform(df2[['State']]).toarray())
#print(df3)

merged_data = pd.concat([df2,df3],axis=1)
#print(merged_data)
#print('%%'*30)
merged_data.pop('State')
merged_data.pop(2)
#print(merged_data)
df = merged_data.rename(columns={0:'Califorina',1:'Florida'})
print(df)

X = df.iloc[:,[0,1,2,4,5]].values
Y = df.iloc[:,3:4].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y_train)
print("Slope:",model.coef_)
print("Y-intercept:",model.intercept_)
print("Model Score:",model.score(X_test,Y_test))

#Predicting accuary
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print("Mean Squared Error",mean_squared_error(Y_test,model.predict(X_test)))
print(np.sqrt(mean_squared_error(Y_test,model.predict(X_test))))
print(mean_absolute_error(Y_test,model.predict(X_test)))
print("R2 Score",r2_score(Y_test,model.predict(X_test)))

