#Load the dataset
import pandas as pd
pd.set_option('display.max_columns',10,'display.width',1000)
df = pd.read_csv('50_Startups.csv')
print(df)
#Encoding
dummies_data = pd.get_dummies(df.State)
print(dummies_data)
#Merging
merged_data = pd.concat([df,dummies_data],axis = 1)
print(merged_data)
#Removing State column
merged_data.pop('State')
print(merged_data)
merged_data.pop('New York')
print(merged_data)