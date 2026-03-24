#Pandas Series---
import numpy as np
import pandas as pd
ser=pd.Series()
print('Pandas Series:',ser)
data=np.array(['g','e','e','k','s'])
ser=pd.Series(data)
print(ser)

import pandas as pd
data=[1,2,3,4]
ser=pd.Series(data)
print(ser)

import pandas as pd
ser=pd.Series()
print(ser)

import numpy as np
import pandas as pd
ser=pd.Series()
print('Pandas Series:',ser)
data=np.array(['g','e','e','k','s'])
ser=pd.Series(data)
print(ser)

import pandas as pd
data_list=['g','e','e','k','s']
ser=pd.Series(data_list)
print(ser)

import pandas as pd
data_dict={'Geeks':10,'for':20,'geeks':30}
ser=pd.Series(data_dict)
print(ser)

import numpy as np
import pandas as pd
data=pd.Series(np.linspace(1,10,5))
print(data)

import pandas as pd
data=pd.Series(range(5,15))
print(dat)

import pandas as pd
data=np.Series(range(1,20,3),index=[x for x in 'abcdefg'])
print(data)

import numpy as np
import pandas as pd
data=np.array(['g','e','e','k','s','f', 'o','r','g','e','e','k','s'])
ser=pd.Series(data)
print(ser[:5])

import pandas as pd
import numpy as np
data = np.array(['g','e','e','k','s','f', 'o','r','g','e','e','k','s'])
ser = pd.Series(data,index=[10,11,12,13,14,15,16,17,18,19,20,21,22])
print(ser[16])

import pandas as pd
df=pd.read_csv('nba.csv')
ser=pd.Series(df['Name']) ## called series from CSV
data=ser.head(10)
data

import pandas as pd
ser1=pd.Series([1,2,3],index=['A','B','C'])
ser2=pd.Series([4,5,6],index=['A','B','C'])
df_sum=ser1.add(ser2)
print(df_sum)

#Pandas DataFrame
import pandas as pd
df= pd.DataFrame()
print(df)
lst=['Geeks','For','Geeks','is','portal','for','Geeks']
df=pd.DataFrame(lst)
print(df)

data={'Name':['Tom','nick','krish','jack'],
'Age':[20,21,19,18]}
df=pd.DataFrame(data)
print(df)

import pandas as pd
name=['aparna','pankaj','sudhir','Geeku']
deg=['MBA','BCA','M.Tech','MBA']
scr=[90,40,80,98]
dic={'FName':name,'degree':deg,'scr':scr}
df=pd.DataFrame(dic)
print(df)

import pandas as pd
lst=[['tom','reacher',25],['krish','pete',30],['nick','wilson',26]]
df=pd.DataFrame(lst)
df['Age']=df['Age'].astype(float)
print(df)

##List of dictoinaries:
#Using from_records()
data=[{'Geeks':'dataframe','For':'using','geeks':'list'},{'Geeks':10,'For':20,'geeks':30}]
df=pd.DataFrame.from_records(data,index=['1','2'])
print(df)
#using pd.DataFrame()
data=[{'Geeks':'dataframe','For':'using','geeks':'list'},{'Geeks':10,'For':20,'geeks':30}]
df=pd.DataFrame(data)
print(df)
#Using from_dict()
data=[{'Geeks':'dataframe','For':'using','geeks':'list'},{'Geeks':10,'For':20,'geeks':30}]
df=pd.DataFrame.from_dict(data)
print(df)
#Using pd.jsonn_normalize()
data=[{'Geeks':'dataframe','For':'using','geeks':'list'},{'Geeks':10,'For':20,'geeks':30}]
df=pd.json_normalize(data)
print(df)

#Check for NaN in Pandas DataFrame:
import numpy as np
import pandas as pd
num={'Integers':[10,15,30,40,55,np.nan,75,np.nan,90,150,np.nan]}
df=pd.DataFrame(num,columns=['Integers'])
check_nan=df['Integers'].isnull().values.any()
print(check_nan)
#Pandas json_normalize Function:
import pandas as pd
data=[{'Roll no':1,'Student':{'first_name':'Ram','last_name':'Kumar'}},{'student':{'English':'95','Math':'88'}
{'Roll no':2,'Student':{'first_name':'Joseph','English':'90','Science':'82'}},
{'Roll no': 3, 'Student':{'first_name':'abinaya','last_name':'devi'},'student':{'English':'91','Math':'98'}},]
df= pd.json_normalize(data)
print(df)

#Basic Method:
import pandas as pd
data = {'Name':['Jai','Princi','Gaurav','Anuj'],
        'Age':[27,24,22,32],
        'Address':['Delhi','Kanpur','Allahabad','Kannauj'],
        'Qualification':['Msc','MA','MCA','Phd']}
df =pd.DataFrame(data)
df[df.columns[1:4]]
import pandas as pd
data = {'Name':['Jai','Princi','Gaurav','Anuj'],
        'Age':[27,24,22,32],
        'Address':['Delhi','Kanpur','Allahabad','Kannauj'],
        'Qualification':['Msc','MA','MCA','Phd']}
df =pd.DataFrame(data)
df[['Name','Qualification']]
#loc()--- Label-based(inclusive)
import pandas as pd
data = {'Name':['Jai','Princi','Gaurav','Anuj'],
        'Age':[27,24,22,32],
        'Address':['Delhi','Kanpur','Allahabad','Kannauj'],
        'Qualification':['Msc','MA','MCA','Phd']}
df =pd.DataFrame(data)
df.loc[1:3,['Name','Qualification']]

#iloc--- Position-based(exclusive)
df=pd.DataFrame({'A':[12,4,5,None,1],
                 'B':[7,2,54,3,None],'C':[20,16,11,3,8],'D':[14,3,None,2,6]})
index_=['Row_1','Row_2','Row_3','Row_4','Row_5',]
df.index=index_
print('Original DataFrame: ')
print(df)

alternate_rows=df.iloc[::2,:]
print('Alternate rows from DataFrame: ')
print(alternate_rows)
alternate_columns=df.iloc[:,::2]
print('Alternate columns from DataFrame: ')
print(alternate_columns)
import pandas as pd
df=pd.read_csv('/content/nba.csv',index_col='Name')
df
first=df.loc['Avery Bradley']
second=df.loc['R.J. Hunter']
print(first,'\n\n\n',second)

df=pd.read_csv('/content/nba.csv',index_col='Name')
row2=data.iloc[3]
print(row2)

#df.isnull():
import pandas as pd
import numpy as np
dict={'First Score':[100,90,np.nan,95],
      'Second Score':[30,45,56,np.nan],
      'Third Score':[np.nan,40,80,98]}
df= pd.DataFrame(dict)
df.isnull()

#df.fillna():
import pandas as pd
import numpy as np
dict={'First Score':[100,90,np.nan,95],
      'Second Score':[30,45,56,np.nan],
      'Third Score':[np.nan,40,80,98]}
df= pd.DataFrame(dict)
df.fillna(0)

#df.dropna():
import pandas as pd
import numpy as np
dict = {'First Score':[100, 90, np.nan, 95],
        'Second Score': [30, np.nan, 45, 56],
        'Third Score':[52, 40, 80, 98],
        'Fourth Score':[np.nan, np.nan, np.nan, 65]}
df = pd.DataFrame(dict)
df.dropna()

#interpolate----
import pandas as pd
df = pd.DataFrame({"A":[12, 4, 5, None, 1],
                   "B":[None, 2, 54, 3, None],
                   "C":[20, 16, None, 3, 8],
                   "D":[14, 3, None, None, 6]})
df.interpolate(method='linear',limit_direction='backward',limit=1)

#thresh():
import pandas as pd
import numpy as np
df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [None, 5, None]})
print(df.dropna(thresh=2))

#subset()--
import pandas as pd
import numpy as np
df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [None, 5, None]})
print(df.dropna(subset=['A']))
import numpy as np
import pandas as pd
data=np.array([[1,2,3],[4,5,6],[7,8,9]])
df=pd.DataFrame(data,columns=['A','B','C'])
print(df)
#Pandas DataFrame index:
import pandas as pd
data = {'Name': ['John', 'Alice', 'Bob', 'Eve', 'Charlie'],
        'Age': [25, 30, 22, 35, 28],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'Salary': [50000, 55000, 40000, 70000, 48000]}
df = pd.DataFrame(data)
print(df.index)
df_with_index=df.set_index('Name')
print(df_with_index)
import pandas as pd
data=pd.read_csv('employees.csv')
data.set_index(['First Name','Gender'],inplace=True,append=True,drop=False)
data.head()

#.at[]
value=data.at['Avery Bradley','Age']
print(value)
#Query()
result=data.query('Age>25 and College =='Duke'')
print(resul) 
#Filter Pandas --
import pandas as pd
dataFrame = pd.DataFrame({'Name': [' RACHEL  ', ' MONICA  ', ' PHOEBE  ',
                                   '  ROSS    ', 'CHANDLER', ' JOEY    '],
                          
                          'Age': [30, 35, 37, 33, 34, 30],
                          
                          'Salary': [100000, 93000, 88000, 120000, 94000, 95000],
                          
                          'JOB': ['DESIGNER', 'CHEF', 'MASUS', 'PALENTOLOGY',
                                  'IT', 'ARTIST']})

display(dataFrame.loc[(dataFrame['Salary']>=10000)& (dataFrame['Age']<40)]& (dataFrame['JOB'].str.stratswith('D')), ['Name','JOB']])
display(dataFrame.loc[(dataFrame['Salary']>=10000)& (dataFrame['Age']<40)&(dataFrame['JOB'].str.startswith('D')),['Name','JOB']])

#Using.concat()
import pandas as pd
data1 = {'Name': ['Jai', 'Princi', 'Gaurav', 'Anuj'],
         'Age': [27, 24, 22, 32],
         'Address': ['Nagpur', 'Kanpur', 'Allahabad', 'Kannuaj'],
         'Qualification': ['Msc', 'MA', 'MCA', 'Phd']}
data2 = {'Name': ['Abhi', 'Ayushi', 'Dhiraj', 'Hitesh'],
         'Age': [17, 14, 12, 52],
         'Address': ['Nagpur', 'Kanpur', 'Allahabad', 'Kannuaj'],
         'Qualification': ['Btech', 'B.A', 'Bcom', 'B.hons']}
df=pd.DataFrame(data1,index=[0,1,2,3])
df1=pd.DataFrame(data2,index=[4,5,6,7])
print(df, '\n\n', df1)
frames=[df,df1]
res=pd.concat(frames)
res
import pandas as pd
data1 = {'Name':['Jai', 'Princi', 'Gaurav', 'Anuj'],
        'Age':[27, 24, 22, 32],
        'Address':['Nagpur', 'Kanpur', 'Allahabad', 'Kannuaj'],
        'Qualification':['Msc', 'MA', 'MCA', 'Phd']}
df=pd.DataFrame(data1,index=[0,1,2,3])
ser=pd.Series([1000,2000,3000,4000],name='Salary')
res=pd.concat([df,ser],axis=1)
res

#How to sort pandas DataFrame?
import pandas as pd
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'Score': [85, 90, 95, 80]
df=pd.DataFrame(data)
x=df.sort_values(by='Age')
print(x)
import pandas as pd
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'Score': [85, 90, 95, 80]}
df=pd.DataFrame(data)
sorted_values=df.sort_values(by='Age',ascending=False)
print(sorted_values)
import pandas as pd
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'Score': [85, 90, 95, 80]}
df=pd.DataFrame(data)
sorted_values=df.sort_values(by=['Name','Age'])
print(sorted_values)
import pandas as pd
data_with_nan = {"Name": ["Alice", "Bob", "Charlie", "David"],"Age": [28, 22, None, 22]}
df_nan = pd.DataFrame(data_with_nan)
sorted_df=df_nan.sort_values(by='Age',na_position='first')
print(sorted_df)
import pandas as pd
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'Score': [85, 90, 95, 80]}
df = pd.DataFrame(data)
sorted_df=df.sort_index(ascending=False)
print(sorted_df)
import pandas as pd
data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Age": [28, 22, 25, 22, 28],
    "Score": [85, 90, 95, 80, 88]
}
df=pd.DataFrame(data)
sorted_df=df.sort_values(by='Age',kind='quicksort')
print(sorted_df)
import pandas as pd
data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Age": [28, 22, 25, 22, 28],
    "Score": [85, 90, 95, 80, 88]
}
df=pd.DataFrame(data)
sorted_df= df.sort_values(by='Age',kind='margesort')
print(sorted_df)
import pandas as pd
data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Age": [28, 22, 25, 22, 28],
    "Score": [85, 90, 95, 80, 88]
}
df=pd.DataFrame(data)
sorted_df=df.sort_values(by='Age',kind='heapsort')
print(sorted_df)
import pandas as pd
data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Age": [28, 22, 25, 22, 28],
    "Score": [85, 90, 95, 80, 88]
}
df = pd.DataFrame(data)
sorted_df=df.sort_values(by='Name',key=lambda col:col.str.lower())
print(sorted_df)

#pivot table()
import pandas as pd
df = pd.DataFrame({'Product': ['Carrots', 'Broccoli', 'Banana', 'Banana',
                               'Beans', 'Orange', 'Broccoli', 'Banana'],
                   'Category': ['Vegetable', 'Vegetable', 'Fruit', 'Fruit',
                                'Vegetable', 'Fruit', 'Vegetable', 'Fruit'],
                   'Quantity': [8, 5, 3, 4, 5, 9, 11, 8],
                   'Amount': [270, 239, 617, 384, 626, 610, 62, 90]})
df
pivot=df.pivot_table(index=['Product'],values=['Amount'],aggfunc='sum')
print(pivot)

import pandas as pd
df = pd.DataFrame({'Product': ['Carrots', 'Broccoli', 'Banana', 'Banana',
                               'Beans', 'Orange', 'Broccoli', 'Banana'],
                   'Category': ['Vegetable', 'Vegetable', 'Fruit', 'Fruit',
                                'Vegetable', 'Fruit', 'Vegetable', 'Fruit'],
                   'Quantity': [8, 5, 3, 4, 5, 9, 11, 8],
                   'Amount': [270, 239, 617, 384, 626, 610, 62, 90]})
df
pivot=df.pivot_table(index=['Category'],values=['Amount'],aggfunc=('median','mean','min'))
print(pivot)

import pandas as pd
data=pd.Series([10,20,30,40],index=['a','b','c','d'])
print(data.index)
data.index=['w','y','z','x']
print(data)
import pandas as pd
ser=['New York','Chicago']
ind=['City 1','City 2']
df=pd.Series(data=ser,index=ind)
df.reset_index(drop=True,inplace=True)
print(df)

#DataFrame to CSV
import pandas as pd
data={'Name':['Amit','Cody','Drew'],"Age":[20,21,25]}
df=pd.DataFrame(data)
df
df.to_csv('Baijid.csv',index=False,header=True)
new_df=pd.read_csv('Baijid.csv')
new_df

#Duplicates:
import pandas as pd
data={'Name':['Alice','Bob','Alice','David'],'Age':[25,30,25,40],'City':['NY','LA','NY','Chicago']}
df=pd.DataFrame(data)
df_cleaned=df.drop_duplicates()
print(df_cleaned)

import pandas as pd
data=pd.DataFrame({'Name':['Alice','Bob','Alice','David'],
                   'Age':[25,30,25,40],
                   'City':['NY','LA','NY','Chicago']})
data
df_cleaned=data.drop_duplicates(subset='Name')
print(df_cleaned)

import numpy as np
import pandas as pd
df=pd.DataFrame({'FirstName': ['Vipul','Ashish','Milan'],'Gender':['','',''],'Age':[0,0,0]})
df['Department']=np.nan
display(df)
df.dropna(how='all',axis=1,inplace=True)
display(df)

import numpy as np
import pandas as pd

df = pd.DataFrame({'FirstName': ['Vipul', 'Ashish', 'Milan'],
                            "Gender": ["", "", ""],
                            "Age": [0, 0, 0]})
df['Department']=np.nan     (## new col a add howa)
display(df)
nan_value=float('NaN')
df.replace("",nan_value,inplace=True)
df.dropna(how='all',axis=1,inplace=True)
display(df)


######Data preprocessing--
display(df.head())
display(df.tail())
#Printing the column names of the DataFrame--
print(list(df.columns))
#Summary of Data Frame
df.info()
#Descriptive Statistical Measures of a DataFrame--
df.describe()
https://www.geeksforgeeks.org/python/data-processing-with-pandas/

