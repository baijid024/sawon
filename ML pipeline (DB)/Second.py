#Line plot
import pandas as pd
import matplotlib.pyplot as plt
data=pd.DataFrame({'Name':['ANSH','SAHIL','JAYAN','ANURAG'],'AGE':[21,23,20,24]})
data
plt.plot(df.index,df['Age'])
plt.xlabel('Index')
plt.ylabel('Age')
plt.title('Age Line Plot')
plt.show()

#Scatter Plot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.DataFrame({'Name:'['ANSH','SAHIL','JAYAN','ANURAG'],'Age':[21,23,20,24]})
data
sns.scatter(x=data.index,y=data['Age'],data=data)
plt.show()

#Box plot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = {'Name': ['ANSH', 'SAHIL', 'JAYAN', 'ANURAG'], 'Age': [21, 23, 20, 45]}
df = pd.DataFrame(data)
sns.boxplot(y='Age',data=df)
plt.show()

#violin plot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = {'Name': ['ANSH', 'SAHIL', 'JAYAN', 'ANURAG'], 'Age': [21, 23, 20, 45]}
df = pd.DataFrame(data)
sns.violinplot(y='Age',data=df)
plt.show()
import seaborn
seaborn.set(style='whitegrid')
fmri=seaborn.load_dataset('fmri')
seaborn.violinplort(x='timepoint',y='signal',hue='region',data=fmri)
import seaborn
seaborn.set(style='whitegrid')
fmri=seaborn.load_dataset('tips')
seaborn.violinplot(x=fmri['total_bill'])

#Swarm plot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data={'Name':['ANSH','SAHIL','JAYAN','ANURAG'],'Age':[21,23,20,24]}
df=pd.DataFrame(data)
sns.swarmplot(x=df.index,y='Age',data=df)
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
data=sns.load_dataset('tips')
sns.swarmplot(x='day',y='total_bill',data=data,size=5)
plt.show()
 import seaborn as sns
 import matplotlib.pyplot as plt
 data=sns.load_dataset('tips')
 sns.swarmplot(y='day',x='total_bill',hue='size',orient='h',data=data)
 plt.show()
 
 #Bar plot()
 import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data={'Name':['ANSH','SAHIL','JAYAN','ANURAG'],'Age':[21,23,20,24]}
df=pd.DataFrame(data)
sns.boxplot(x='Name',y='Age',data=df)
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
data=sns.load_dataset('titanic')
sns.barplot(x='class',y='fare',data=data)
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
data=sns.load_dataset('titanic')
sns.barplot(x='class',y='fare',hue='sex',data=data)
plt.show()
#point plot()
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
data={'Name':['ANSH','SAHIL','JAYAN','ANURAG'],'Age':[21,23,20,24]}
df=pd.DataFrame(data)
sns.pointplot(x='Name',y='Age',data=df)
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
data=sns.load_dataset('tips')
sns.pointplot(x='size',y='total_bill',linestyles='-.',markers='^',hue='sex',data=data)
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
data=sns.load_dataset('tips')
sns.countplot(x='sex',hue='smoker',data=data)
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
df=sns.load_dataset('tips')
sns.countplot(y='sex',data=df,palette='Set2')
plt.show()
##KDE
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['Species']=iris.target
df['Species']=df['Species'].map({0:'Setosa',1:'Versicolor',2:'Virginica'})
sns.kdeplot(data=df[df['Species']=='Virginica'],x='sepal length(cm)',fill=True,label='Virginica')
plt.legend()
plt.show()
#Pair plot()
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
custom_palette=sns.color_palette('husl',8)
sns.set_palette(custom_palette)
data=sns.load_dataset('iris')
sns.pairplot(data,hue='species')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
data=sns.load_dataset('tips')
data.head()
sns.pairplot(data)
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
df=sns.load_dataset('tips')
selected_vars=['total_bill','tip']
sns.pairplot(data=df,vars=selected_vars)
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
df=sns.load_dataset('tips')
sns.pairplot(df,hue='size')
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
data=sns.load_dataset('tips')
sns.pairplot(data,diag_kind='kde')
plt.show()
g=sns.pairplot(df,hue='day')
g.fig.suptitle('Pairplot of tips Dataset',y=1.02)
g.set(xticks=[],yticks=[])
plt.show()

#Pairplot with Custom Palette by sex
import seaborn as sns
import matplotlib.pyplot as plt
df=sns.load_dataset('tips')
custom_palette={'Male':'lightblue','Female':'pink'}
sns.pairplot(df,hue='sex',palette=custom_palette)
plt.show()
#Joint Plots
import seaborn as sns
import matplotlib.pyplot as plt
data=sns.load_dataset('tips')
sns.jointplot(x='total_bill',y='tip',data=data,kind='scatter',color='#008B8B')
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
iris=sns.load_dataset('iris')
g=sns.JointGrid(data=iris,x='sepal_length',y='sepal_width',hue='species')
g=g.plot(sns.scatterplot,sns,histplot)
plt.show()

#sns.distplot()----( hist+kde)
import seaborn as sns
import matplotlib.pyplot as plt
tips=sns.load_dataset('tips')
plot=sns.FacetGrid(tips,col='time',row='sex')
plot.map(sns.scatter,'total_bill','tip')
plt.show()