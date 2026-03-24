#Theory:
Data Processing মানে হলো তথ্যকে সাজানো, বিশ্লেষণ করা এবং ফলাফল বের করার একটি ধাপ-ধাপ প্রক্রিয়া।
Data Processing = Raw Data → Process → Useful Information
ধরো তুমি একটা দোকানের বিক্রির হিসাব রাখতে চাও।
প্রতিদিন বিক্রির ডেটা সংগ্রহ করা হলো Raw Data।
ডুপ্লিকেট বা ভুল এন্ট্রি মুছে ফেলা হলো Cleaning।
সেই ডেটা সফটওয়্যারে দিয়ে রিপোর্ট বানানো হলো Processing।
রিপোর্ট থেকে মাসের মোট বিক্রি দেখা গেল → এটিই Information।

#Generate Test Datasets for Machine Learning:
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
x,y=make_moons(n_samples=200,shuffle=True,noise=0.1,random_state=42)
plt.scatter(x[:,0],x[:,1],c=y)
plt.show()

from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
x,y=make_circles(n_samples=500,shuffle=True,noise=0.15,random_state=42)
plt.scatter(x[:,0],x[:,1],c=y)
plt.show()

#Multi-class classification:
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
x,y=make_blobs(n_samples=500,centers=3,n_features=2,random_state=23)
plt.scatter(x[:,0],x[:,1],c=y)
plt.show()

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
x,y=make_classification(n_samples=100,n_features=2,n_redundant=0,n_informative=2,n_repeated=0,n_classes=3,n_clusters_per_class=1)
plt.scatter(x[:,0],x[:,1],c=y)
plt.show()

from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
x,y=make_regression(n_samples=50,n_features=1,noise=20,random_state=42)
plt.scatter(x,y)
plt.show()
#Multilabel feature using make_sparse_uncorrelated()
from sklearn.datasets import make_sparse_uncorrelated
import matplotlib.pyplot as plt
x,y=make_sparse_uncorrelated(n_samples=100,n_feature=4,random_state=23)
plt.figure(figsize(12,10))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.scatter(x[:,i],y)
    plt.xlabel('x'+str(i+1))
    plt.ylabel('y')
 plt.show()
 
 #Create Test Datasets Using Sklearn:
 from sklearn.datasets import make_blobs
 from matplotlib import pyplot as plt
 from matplotlib import style 
 style.use('fivethirtyeight')
 x,y=make_blobs(n_sample=100,centers=3,cluster_std=1,n_features=2)
 plt.scatter(x[:,0],x[:,1],s=40,color='g')
 plt.xlabel('X')
 plt.ylabel('Y')
 plt.show()
 plt.clf()
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
x,y=make_moons(n_samples=1000,noise=0.1)
plt.scatter(x[:,0],x[:,1],s=40,color='g')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
plt.clf()
##plt.clf() → current figure clear করে দেয়।
মানে আগের plot remove হয়ে যায় → পরের plot এর জন্য fresh figure।

#Steps in Data Preprocessing:
step 1:
import pandas as pd
import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

step 2:
df=pd.read_csv("/content/diabetes.csv")
print(df.head())

step 3:
df.info()
#There is no Null Values in the datasets
df.isnull().sum()

#Check the outliers:
fig,axs= plt.subplots(9,1,dpi=95,figsize=(7,17))
i=0
for col in df.columns:
    axs[i].boxplot(df[col],vert=False)
    axs[i].set_ylabel(col)
    i+=1
plt.show()

#step 5:
corr=df.corr()
plt.figure(dpi=130)
sns.heatmap(corr,annot=True,fmt='.2f')
plt.show()
corr['Outcome'].sort_values(ascending=False)

plt.pie(df.Outcome.value_counts(),labels=['Diabetes','Not Diabetes'],autopct='%.f',shadow=True)
plt.title('Outcome')
plt.show()

dropna() হলো pandas function যা missing values (NaN) থাকা rows বা columns remove করে।
df2.dropna(subset=['Embarked'], axis=0, inplace=True) 

#axis=0 → row drop করা হবে & axis=1 → column drop করা হবে

df3=df2.fillna(df2.Age.mean())
df3.isnull().sum()
df1=df.drop(columns=['Name','Ticket'])
df1.shape

#Using isnull():
import numpy as np
import pandas as pd
df={'First Score':[100,90,np.nan,95],
     'Second Score':[30,45,56,np.nan],
      'Third Score':[np.nan,40,80,98]}
d=pd.DataFrame(df)
display(d)
mv=d.isnull().sum()
print(mv)

#Using isna():
import pandas as pd
import numpy as np
data={'Name':['Amit','Sita',np.nan,'Raj'],
       'Age':[25,np.nan,22,28]}
df=pd.Dataframe(data)
print(df.isna())

#Using notnull:
import pandas as pd
import numpy as np
data= {'First Score':[100,90,np.nan,95],
        'Second Score':[30,45,56,np.nan],
        'Third Score':[np.nan,40,80,98]}
df=pd.DataFrame(data)
print(df.notnull())

#Using fillna()
import pandas as pd
import numpy as np
data= {'First Score':[100,90,np.nan,95],
        'Second Score':[30,45,56,np.nan],
        'Third Score':[np.nan,40,80,98]}
df=pd.DataFrame(data)
df.fillna(0)
df.fillna(method='bfill')

#Using dropna()--Remove rows that contain at least one missing value.
import pandas as pd
import numpy as np
dict = {'First Score': [100, 90, np.nan, 95],
        'Second Score': [30, np.nan, 45, 56],
        'Third Score': [52, 40, 80, 98],
        'Fourth Score': [np.nan, np.nan, np.nan, 65]}
df = pd.DataFrame(dict)
df.dropna()

import pandas as pd
import numpy as np
dict = {'First Score': [100, np.nan, np.nan, 95],
        'Second Score': [30, np.nan, 45, 56],
        'Third Score': [52, np.nan, 80, 98],
        'Fourth Score': [np.nan, np.nan, np.nan, 65]}
df = pd.DataFrame(dict)
df.dropna(how='all')
dict = {'First Score': [100, np.nan, np.nan, 95],
        'Second Score': [30, np.nan, 45, 56],
        'Third Score': [52, np.nan, 80, 98],
        'Fourth Score': [60, 67, 68, 65]}
df = pd.DataFrame(dict)
df.dropna(axis=1)

#Absolute Maximum Scaling():
import numpy as np
import pandas as pd
data=pd.read_csv('/content/diabetes.csv')
print(data.head())
max_vals=np.max(np.abs(data))
print(max_vals)

#Min_max Scaling: (MinMaxScaler() → ডেটাকে 0–1 range এ আনার জন্য scaler বানায়।)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaled_data=scaler.fit_transform(df)
scaled_df=pd.DataFrame(scaled_data,columns=df.columns)
scaled_df.head()

#Normalizer()-এটি row-wise normalization করে।
from sklearn.preprocessing import Normalizer()
scaler=Normalizer()
scaled_data=scaler.fit_transform(df)
scaled_df=pd.DataFrame(scaled_data,columns=df.columns)
print(scaled_data.head())

#StandardScaler()
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_data=scaler.fit_transform(df)
scaled_df=pd.DataFrame(scaled_data,columns=df.columns)
print(scaled_df.head())

#Robust Scaling():
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
scaled_data=scaler.fit_transform(df)
scaled_df=pd.DataFrame(scaled_data,columns=df.columns)
print(scaled_df.head())

MinMaxScaler → সবকিছু 0–1 এ আনে (outlier sensitive)।
StandardScaler → mean=0, std=1 এ আনে (outlier sensitive)।
RobustScaler → median & IQR ব্যবহার করে scale করে (outlier resistant)।

#One Hot Encoding কী?
Machine Learning model সরাসরি categorical data (যেমন string → "Red", "Blue", "Green") বুঝতে পারে না।
তাই categorical values কে numerical form এ রূপান্তর করতে হয়।
One Hot Encoding হলো একটা process, যেখানে প্রতিটি category আলাদা binary column (0/1) এ convert হয়।

#SMOTE = Minority grow (bigger)
#NearMiss = Majority shrink (smaller)