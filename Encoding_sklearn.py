#Load the dataset
import pandas as pd
pd.set_option('display.max_columns',10,'display.width',1000)
df = pd.read_csv('50_Startups.csv')
print(df)
df2 = df
print("--"*50)
#Encoding form sklearn
'''
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df2.State = labelencoder.fit_transform(df2.State)
print(df2)'''

#ONEHOT Encoding
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
df3 = pd.DataFrame(onehotencoder.fit_transform(df2[['State']]).toarray())
print(df3)

merged_data = pd.concat([df2,df3],axis=1)
print(merged_data)
print('%%'*30)
merged_data.pop('State')
merged_data.pop(2)
print(merged_data)
df = merged_data.rename(columns={0:'Califorina',1:'Florida'})
print(df)

