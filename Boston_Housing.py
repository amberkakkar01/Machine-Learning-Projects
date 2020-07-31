import pandas as pd
pd.set_option('display.max_columns',10,'display.width',1000)
from sklearn.datasets import load_boston
boston_dataset = load_boston()
print(boston_dataset)

X = boston_dataset.data
Y = boston_dataset.target
print("----"*30)
print(X)
print('----'*30)
print(Y)
columns = boston_dataset.feature_names
print(columns)

boston_df = pd.DataFrame(boston_dataset.data)
print(boston_df.head())
print("%"*300)
boston_df.columns = boston_dataset.feature_names
print(boston_df.head())
print("Shape of Dataset:",boston_df.shape)
Y = pd.DataFrame(boston_dataset.target)
import matplotlib.pyplot as plt
plt.boxplot([boston_df['CRIM'],boston_df['ZN']])
#plt.show()

from scipy import stats
import numpy as np
z = np.abs (stats.zscore(boston_df))
print(z)
print(np.where(z>3))#values which are greater than 3 in z
print(z[55][1])
#deleting some outliar
boston_df_new = boston_df[(z<3).all (axis=1)]

print("Old Data",boston_df.shape)
print("New Data",boston_df_new.shape)

import matplotlib.pyplot as plt
plt.boxplot([boston_df['CRIM'],boston_df_new['CRIM'],boston_df['ZN'],boston_df_new['ZN']])
plt.show()

