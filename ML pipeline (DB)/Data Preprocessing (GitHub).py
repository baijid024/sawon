#Short Summary
LabelEncoder → টেক্সট কে সংখ্যা বানায় (যেমন “Out” = 0, “Not_Out” = 1)।
StandardScaler → Numerical কলামকে normalize করে (mean = 0, std = 1)।
MinMaxScaler → Numerical কলামকে 0 থেকে 1 এর মধ্যে আনে।

#Short Summary
1. Why we write
df = df[(df['runs'] >= lower_bound) & (df['runs'] <= upper_bound)]
👉 Here df[...] is row filtering.
Inside the brackets [...], we create a Boolean condition (True or False for every row).

Example:
df['runs'] >= lower_bound
gives something like: [True, True, False, True, ...].
Then we combine conditions with & and pass that list of True/False to df[...].
➡️ If you don’t put df[...], only the condition will be created, not the filtered DataFrame.
That’s why we must write df[...] → it tells pandas: “select only those rows from df where condition is True.”

2. Why we write
df[col] = label_enc.fit_transform(df[col])
👉 Here LabelEncoder needs the values of that column.
df[col] → gives the Series of values (e.g., ["Dhoni", "Kohli", "Rohit"]).
fit_transform converts them into numbers (e.g., [0,1,2]).
Then we assign it back to the same column → df[col] = ....
➡️ If you just wrote col, it would only be the string name of the column, not the data.
Example:
col = "batsmanName"
df[col] → [Dhoni, Kohli, Rohit] (the actual values)
col → "batsmanName" (just text, useless for encoding)
That’s why we need df[col].
3. Why we write
df_scaled[df_scaled.select_dtypes(include=["int64", "float64"]).columns] = \
    scaler.fit_transform(df_scaled.select_dtypes(include=["int64", "float64"]))
👉 This line means:
First, we select only numeric columns:
df_scaled.select_dtypes(include=["int64", "float64"])
Then we scale those columns with scaler.fit_transform().
Finally, we assign the scaled values back into those same columns inside df_scaled[...].

➡️ If we don’t write df_scaled[...] =, then the scaled values would be calculated but not stored.
We want to replace the original numeric values with scaled values in df_scaled.
That’s why we must use df_scaled[...] = ....

✅ Summary (Best Answer):
df[...] → always needed when you want to select rows/columns of the DataFrame, otherwise you’re just working with text or conditions.
df[col] → gives you the actual data in the column, not just the name.
df_scaled[...] = ... → ensures the transformed/scaled values are stored back in the DataFrame.

#1.Import Libraries:
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
#2.Load Dataset
file_path='/content/batting_summary.csv'
df=pd.read_csv(file_path)
#3.Basic Info
print(df.info())
print(df.head())
#4.Checking for Missing Values:
print(df.isnuull().sum())
#5.Fix Data Types
df['SR']=pd.to_numeric(df['SR'],errors='coerce')
print(df['SR'].dtype)
#6.Handle Missing Values:
for col in df.select_dtypes(include=['int64','float64']).columns:
    df[col].fillna(df[col].mean(),inplace=True)
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0],inplace=True)
df['SR'].fillna(df['SR'].mean(),inplace=True)
#7.Encode Categorical variables:
label_enc=LabelEncoder()
for col in ['teamInnings','batsmanName','Out/Not_Out','match','matchID']:
    df[col]=label_enc.fit_transform(df[col])
#8.Feature Scaling:
scaler=StandardScaler()
df_scaled=df.copy()
df_scaled[['runs','balls','4s','6s','SR']]=scaler.fit_transform(df_scaled[['runs','balls','4s','6s','SR']])
minmax=MinMaxScaler()
df_minmax=df.copy()
df_minmax[['runs','bslls','4s','6s','SR']]=minmax.fit_transform(df_minmax[['runs','balls','4s','6s','SR']])
print(df_scaled.head())
print(df_minmax.head())
#9.Train_test_split:
x=df.drop(columns=['runs'])
y=df['runs']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train.shape)
print(y_train.shape)

