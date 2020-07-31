import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_excel('data.xlsx')
print(df)
x = df.iloc[:,:1].values
y = df.iloc[:,1:].values
#print(x)
#print(y)
plt.title("House Prediction")
plt.xlabel("Square_feet")
plt.ylabel("Price")
plt.scatter(x,y,color ='r')
plt.show()

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)
y_pred = model.predict(x)
plt.plot(x,y_pred,color='g')

print('b0:',model.intercept_)
print('b1:',model.coef_)
print('Model Score:',model.score(x,y))

#preict
x_input = [[1000]]
predicted_y = model.predict(x_input)
print("Pred",predicted_y)
#plotting graph
plt.scatter(x_input,predicted_y,color='k')
plt.show()
