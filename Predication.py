import numpy as np
import matplotlib.pyplot as plt

x = np.array([0,1,2,3,4,5,6,7,8,9])
y = np.array([3,5,7,9,11,13,15,17,19,21])
n = len(x)
sum_of_x = np.sum(x)
print(sum_of_x)
sum_of_y = np.sum(y)
print(sum_of_y)
sum_of_xy = np.sum(x*y)
print(sum_of_xy)
sum_of_x_square = np.sum(x*x)
print(sum_of_x_square)
print('%'*50)
b0 =  (sum_of_y*sum_of_x_square - sum_of_x*sum_of_xy)/(n * sum_of_x_square - sum_of_x*sum_of_x)
b1 = (n*sum_of_xy - sum_of_x*sum_of_y)/(n * sum_of_x_square - sum_of_x*sum_of_x)
print(b0)
print(b1)
#plot a graph
plt.title("Manual Calc")
plt.xlabel('X-Axis')
plt.ylabel("Y-Axis")
plt.scatter(x,y,marker='*',color='r')
plt.xticks(x)
plt.yticks(np.arange(0,25,2))
#best fit line
y_pred = b0 + b1*x
plt.plot(x,y_pred,color='k')
plt.show()
#predication
x_input = eval(input("Enter a no for predication"))
predicted_y = b0+b1*x_input
print("Predicted value", predicted_y)