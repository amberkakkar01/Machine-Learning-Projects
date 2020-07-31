import numpy as np
import matplotlib.pyplot as plt

x = np.array([150,200,250,300,350,400,600])
#print(x)
#print(x.shape)
x = x.reshape(-1,1)
#print(x)
#print(x.shape)
y = np.array([6450,7450,8450,9450,11450,15450,18450])
y = y.reshape(-1,1)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)
b0 = model.intercept_
b1 = model.coef_
print(b0)
print(b1)
plt.title("Prediction")
plt.xlabel("feature")
plt.ylabel("target")
plt.xticks(x)
plt.xticks(np.arange(0,25,2))
plt.scatter(x,y,color ='r',label='Actual')

#predict y for x
pred_y = model.predict(x)
print(pred_y)
plt.plot(x,pred_y,color ='g',label='Predicated')
plt.show()

#preict
x_input = eval(input("Enter a no to predict:"))
x_input = np.array([x_input],ndmin=2)
print(x_input, x_input.ndim)
predicted_y = model.predict(x_input)
print("Pred",predicted_y)
