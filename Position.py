#Adding Linear Reg + Poly Reg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('Position_Salaries.csv')
print(df)

X = df.iloc[:,1:2].values
Y = df.iloc[:,-1].values
print(X)
print(Y)

plt.title("Positon")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.scatter(X,Y,color ='b',label='Actual')


from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(X,Y)
predict_y1 = model1.predict(X)
plt.plot(X,predict_y1,color='r',label ='Predicated-1')
#plt.legend()
#plt.show()
print("Accuracy by model1:",model1.score(X,Y))

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4,include_bias=False)
X_poly = poly.fit_transform(X)
#poly.fit(X_poly)
model2 = LinearRegression()
model2.fit(X_poly,Y)
predict_y2 = model2.predict(X_poly)
print("Accurancy by model2:",model2.score(X_poly,Y))
plt.plot(X,predict_y2,color='g',label='Predicated')
plt.legend()
plt.show()

#Predicting accuary
'''import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print("Mean Squared Error",mean_squared_error(Y_test,model.predict(X_test)))
print(np.sqrt(mean_squared_error(Y_test,model.predict(X_test))))
print(mean_absolute_error(Y_test,model.predict(X_test)))
print("R2 Score",r2_score(Y_test,model.predict(X_test)))'''