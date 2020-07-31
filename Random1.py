import numpy as np
import matplotlib.pyplot as plt
ran = np.random.RandomState(7)
x = 10*ran.randn(50)
print(x)
y = 2*x+3 + ran.randn(50)
print(y)
y[10] = 15
y[5] = 130
y[3] =105
plt.figure(1)
plt.boxplot(y)
plt.figure(2)
plt.scatter(x,y)
plt.title("SOne Dommy Data")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
#plt.show()
X = x.reshape(-1,1)
Y = x.reshape(-1,1)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,Y)
Y_pred = model.predict(X)
plt.plot(X,Y_pred,color='r')
plt.show()
