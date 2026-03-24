**PANDAS tutorial**

import pandas as pd

import numpy as np



df = pd.DataFrame(\[\[1,2,3],\[4,5,6],\[7,8,9],\[10,11,12],columns=\['A','B','C'],index=\['x','y','z','zz'])

df

df.head()

df.tail()

df.shape

df.columns

df.size

df.index.tolist()

df.info()

df.describe()

df.nunique()

df\['A'].unique()



coffee = pd.read\_csv('/content/coffee.csv')

coffee



print(coffee)

display(coffee)

coffee.head(5)

coffee.tail(2)

coffee.sample(5)



loc-> label(name based)

coffee.loc\[\[0,1,5]]

coffee.loc\[5:9,\['Day':'Units Sold']]



iloc-> integer number(name based)

coffee.iloc\[:,\[0,2]]



coffee.index = coffee\['Day']

coffee.loc\['Monday':'Wednesday']



coffee.loc\[1:3,\['Units sold']] =10

coffee.Day

coffee\['Day']

coffee.at\[0,'Units Solid']

coffee.iat\[3,1]

coffee.sort\_values(\['Units Sold'],ascending=False)

coffee.sort\_values(\['Units Sold','Coffee type'], ascending=\[0,1])

bios.head()

bios.loc\[bios\['height\_cm}>215]

bios.loc\[bios\['height\_cm]>215,\['name','height\_cm']]

bios\[bios\['height\_cm']>215]\['name','height\_cm']]

bios\[(bios\['height\_cm]>215) \& (bios\['born\_country']=='USA')]

bios\[bios\['name'].str.contains('Keith",case=False)

bios\[bios\['name'].str.contains('Keith|patrick,case = False)

bios\[bios\['born\_country'].isin(\['USA',"FRA']) \& (bios\['name'].str.stratswith('Keith'))]

bios.query('born\_country == "USA" and born\_city =="Seattle"')

coffee\['price']=4.99

coffee\['new\_price']=np.where(coffee\['Coffee Type']=='Espresso',3.99,5)

coffee.drop(columns=\['price'],inplace=True)

coffee=coffee\[\['Day','Cofffe Type','Units Sold','new\_price']]

coffee\['revenue']=coffee\['Units Sold']\*coffee\['new\_price']

coffee.rename(columns={'new\_price':'price'},inplace= True)

bios\_new=bios.copy()



bios\_new\['first\_name']=bios\_new\['name'].str.split('').str\[0]

bios.query('first\_name=="Keith"')

