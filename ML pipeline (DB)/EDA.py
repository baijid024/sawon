#1.Import libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Set plot style:
sns.set(style='whitegrid',palette='muted',font_scale=1.1)
#2.Load Dataset:
df=pd.read_csv("")
display(df)

#3.Basic Dataset Information:
print('Shape of dataset:'df.shape) #Rows & Columns
print('\nColumn names:',df.columns.tolist())
print('\nData Types:\n',df.dtypes)
print('\nMissing values:\n',df.isnull().sum())
print('\nDataset Info:\n')
print(df.info()) #Data types
print('\nStatistical Summary (Numerical Columns):\n',df.describe())

#4.Data cleaning:
#Convert SR to numeric-
df['SR']=pd.to_numeric(df['SR'],errors='coerce')
print(df['SR'].dtype)

#Handle missing values(if any)-
df.fillna(df.mode().iloc[0],inplace=True)

#Remove duplicates-
df.drop_duplicates(inplace=True)

#5.Univariate Analysis:
#Runs Distribution-
plt.figure(figsize=(8,5))
sns.histplot(df['runs'],kde=True,bins=30,color='blue')
plt.title('Distribution of Runs')
plt.show()

#Balls Distribution-
plt.figure(figsize=(8,5))
sns.histplot(df['balls'],kde=True,bins=30,color='green')
plt.title('Distribution of Balls Faced')
plt.show()

#strike Rate Distribution-
plt.figure(figsize=(8,5))
sns.histplot(df['SR'],kde=True,bins=30,color='red')
plt.title('Distribution of Strike Rate')
plt.show()

#Out vs Not out count-
plt.figure(figsize=(6,4))
sns.countplot(x='Out/Not_out',data=df,palette='Set2')
plt.title('Out vs Not Out')
plt.show()

#6.Bivariate Analysis:
#Runs vs Balls-
plt.figure(figsize=(8,5))
sns.scatterplot(x='balls',y='runs',hue='Out/Not_Out',data=df)
plt.title('Runs vs Balls')
plt.show()

#Batting Position vs Runs-
plt.figure(figsize=(10,5))
sns.boxplot(x='battingPos',y='runs',data=df,palette='Set3')
plt.title('Runs by batting position')
plt.show()

#Team Innings vs Average Runs-
plt.figure(figsize=(12,6))
sns.barplot(x='teamInnings',y='runs',data=df,estimator=np.mean,palette='coolwarm')
plt.xticks(rotation=90)
plt.title('Average Runs per Team Innigs')
plt.show()

2D matrix visualize করে, কিন্তু ভিতরে multiple variables compare করা হয়।
#7.Correlation Analysis:
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True,annot=True,cmap='coolwarm'))
plt.title('correlation Heatmap')
plt.show()

#8.Top player Analysis:
df.groupby('batsmanName')['runs'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,5))
sns.barplot=(x=top_batsman.values,y=top_batsman.index,plaette='Blues_r'
plt.title('Top 10 Batsman by Runs')
plt.xlabel('Total Runs')
plt.ylabel('Batsman')
plt.show()

#short summary:
| Type       | How Many Variables? | Goal                        | Example                           |
| ---------- | ------------------- | --------------------------- | --------------------------------- |
| Univariate | 1 (single column)   | Distribution / Pattern বোঝা | Histogram of `runs`               |
| Bivariate  | 2 (two columns)     | Relation বোঝা               | Scatter plot of `runs` vs `balls` |

df.corr(numeric_only=True) → ডেটার numeric column গুলোর মধ্যে correlation coefficient বের করে।
correlation = -1 থেকে +1 এর মধ্যে হয়।
+1 = একে অপরের সাথে strongly positive relation।
-1 = strongly negative relation।
0 = কোনো relation নেই
Heatmap দেখে তুমি বুঝবে কোন numeric feature কোনটার সাথে বেশি strongly related।

df.groupby('batsmanName')['runs'].sum()
প্রতিটা batsman-এর total runs বের করে।

Correlation heatmap → কোন numeric column কোনটার সাথে relation strong সেটা বোঝার জন্য।
Top Player barplot → সবচেয়ে বেশি রান করা ব্যাটসম্যানদের rank wise দেখানোর জন্য।