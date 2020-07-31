import numpy as np
ran = np.random.RandomState(7)
x = 10*ran.rand(50)
print(x)
y = np.sin(x)+0.1*ran.randn(50)
print(y)

import  matplotlib.pyplot as plt
plt.title("some Dummy data")
plt.xlabel('X-Axis')
plt.ylabel("Y-Axis")
plt.scatter(x,y,label='Actual')

X = x.reshape(-1,1)
Y = y.reshape(-1,1)
#Finding power 6
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=6)
data = poly.fit_transform(X)
X_poly = poly.fit_transform(X)
poly.fit(X_poly,Y)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_poly,Y)
predicted_y = model.predict(X_poly)
plt.scatter(X,predicted_y,color='r',marker='*',label='predicted')
plt.legend()
plt.show()
