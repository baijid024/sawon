#Line Chart:

import matplotlib.pyplot as plt
x=[10,20,30,40]
y=[20,25,35,55]
plt.plot(x,y)
plt.title('Line Chart')
plt.ylabel('Y-Axis')
plt.xlabel('X-Axis')
plt.show()

#Bar Chart:

import matplotlib.pyplot as plt
x=['Thur','Fri','Sat','Sun']
y=[170,120,250,190]
plt.bar(x,y)
plt.title('Bar Chart')
plt.ylabel('Total Bill')
plt.xlabel('Day')
plt.show()

#Histogram

import matplotlib.pyplot as plt
x=[7,8,9,10,12,12,12,13,14,14,15,16,16,17,18,18,19,20,20,21,22,23,24,25,25,26,28,30,32,35,36,38,40,42,44,48,50]
plt.hist(x,bins=10,color='pink')
plt.title('Histogram')
plt.ylabel('Frequency')
plt.xlabel('Total Bill')
plt.show()

#Scatter Plot

import matplotlib as plt
x=['Thur','Fri','Sat','Sun','Thur','Fri','Sat','Sun']
y=[170,120,250,190,160,130,240,200]
plt.scatter(x,y)
plt.title('Scatter Plot')
plt.xlabel('Day')
plt.ylabel('Total Bill')
plt.show()

#Pie chart

import matplotlib.pyplot as plt
import pandas as pd
cars = ['AUDI','BMW','FORD','TESLA','JAGUAR']
data = [23,10,35,15,12]
plt.pie(data,labels=cars)
plt.title('pie chart')
plt.show()

#Box plot
import matpoltlib.pyplot as plt
data =[[10,12,14,15,18,20,22],[8,9,11,13,17,19,21],[14,16,18,20,23,25,27]]
plt.boxplot(data)
plt.title('Box Plot')
plt.xlabel('Groups')
plt.ylabel('Values')
plt.show()

#Heatmap
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)
data= np.random.rand(10,10)
plt.imshow(data,cmap='viridis',interpolation='nearest')
plt.colorbar()
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Heatmap')
plt.show()

#customizing Line chart:
import matplotlib.pyplot as plt
x=[10,20,30,40]
y=[20,25,35,55]
plt.plot(x,y,color='green',linewidth=3,marker='o',markersize=15,linestyle='--')
plt.title('Customizing Line Chart')
plt.ylabel('Y-Axis')
plt.xlabel('X-Axis')
plt.show()

#customizing Bar Chart:
import matplotlib.pyplot as plt
x=['Thur','Fri','Sat','Sun']
y=[170,120,250,190]
plt.bar(x,y,color='green',edgecolor='black',linewidth=2)
plt.title('Customizing Bar chart')
plt.xlabel('Day')
plt.ylabel('Total Bill')
plt.show()

#Customizing Histogram Plot:
import matplotlib.pyplot as plt
x=[7, 8, 9, 10, 10, 12, 12, 12, 13, 14, 14, 15, 16, 16, 17,18, 18, 19, 20, 20, 21, 22, 23, 24, 25, 25, 26, 28, 30,32, 35, 36, 38, 40, 42, 44, 48, 50]
plt.hist(x,bins=10,color='green',edgecolor='pink',linestyle='--',alpha=0.5)
plt.title('Customizing Histogram PLot')
plt.xlabel('Total Bill')
plt.ylabel('Frequency')
plt.show()

#customizing Scatter Plot:
import matplotlib.pyplot as plt
x=['Thur','Fri','Sat','Sun','Thur','Fri','Sat','Sun']
y=[170,120,250,190,180,130,260,200]
size=[2,3,4,2,3,2,4,3]
bill=[170,120,250,190,180,130,260,200]
plt.scatter(x,y,c=size,s=bill,marker='D',alpha=0.5)
plt.title('Customizing Scatter Plot')
plt.xlabel('Day')
plt.ylabel('Total Bill')
plt.show()

#Customizing Pie Chart
import matplotlib.pyplot as plt
import pandas as pd
cars=['AUDI','BMW','FORD','TESLA','JAGUR']
data=[23,13,35,15,12]
explode=[0.1,0.5,0,0,0]
colors=['orange','cyan','yellow','grey','green']
plt.pie(data,labels=cars,explode=explode,autopct='%1.2f%%',colors=colors,shadow=True)
plt.show()

#Matplotlib's Core Components: Figure
import matplotlib.pyplot as plt
fig= plt.figure(figsize=(6,4), facecolor ='lightblue')) #### 
ax=fig.add_axes([0.1,0.1,0.8,0.8]) ### 
x=[1,2,3,4]
y=[10,20,15,25]
ax.plot(x,y)
plt.title('Simple Line Plot')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.show()

import matplotlib.pyplot as plt
from matplotlib.figure import Figure 
x=[10,20,30,40]
y=[20,25,35,55]
fig = plt.figure(figsize=(5,4))
ax= fig.add_axes([0.1,0.1,0.8,0.8])
ax.plot(x,y)
ax.plot(y,x)
ax.set_title('Linear Graph')
ax.set_xlabel('X-Axis')
ax.set_ylabel('Y-Axis')
ax.legend(labels=('line 1','line 2'))
plt.show()

#Advanced Techniques for visulizing Subplots
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
x=[10,20,30,40]
y=[20,25,35,55]
fig=plt.figure(figsize=(10,4))
ax1=fig.add_axes([0.1,0.1,0.35,0.8])
ax2=fig.add_axes([0.55,0.1,0.35,0.8])
ax1.plot(x,y)
ax2.plot(y,x)
plt.show()

#Using subplot()
import matplotlib.pyplot as plt
x =[10,20,30,40]
y =[20,25,35,55]
plt.figure()
plt.subplot(121)
plt.plot(x,y)
plt.subplot(122)
plt.plot(y,x)
plt.show()

#Using subplot2grid()
import matplotlib.pyplot as plt
x=[10,20,30,40]
y=[20,25,35,55]
axes1=plt.subplot2grid((7,1),(0,0),rowspan =2, colspan = 1)
axes2=plt.subplot2grid((7,1),(2,0),rowspan =2,colspan=1)
axes1.plot(x,y)
axes2.plot(y,x)
plt.show()

#Line chart in Matplotlib - Python
import matplotlib.pyplot as plt
import numpy as np
x=np.array([1,2,3,4])
y=x*2
plt.plot(x,y)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
x=np.array([1,2,3,4])
y=x*2
plt.title('Any suitable title')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.plot(x,y)
plt.show()

#Multiple Line Charts:
import matplotlib.pyplot as plt
import numpy as np
x=np.array([1,2,3,4])
y=x*2
plt.plot(x,y)
plt.title('Any suitable title')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
plt.figure()
x1=[2,4,6,8]
y1=[3,5,7,9]
plt.plot(x1,y1,'-.')
plt.show()

#Multiple Plots on the Same Axis:
import matplotlib.pyplot as plt
import numpy as np
x=np.array([1,2,3,4])
y=x*2
plt.plot(x,y)
x1=[2,4,6,8]
y1=[3,5,7,9]
plt.plot(x1,y1)
plt.title('Any suitable title')
plt.xlabel('X-axis data')
plt.ylable('Y-axis data')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
x=np.array([1,2,3,4])
y=x*2
plt.plot(x,y)
x1=[2,4,6,8]
y1=[3,5,7,9]
plt.plot(x,y1,'-.')
plt.title('Any suitable title')
plt.xlabel('X-axis Data')
plt.ylabel('Y-axis Data')
plt.fill_between(x,y,y1,color='green',alpha=0.5)
plt.show()

#Bar plot in Matplotlib
import matplotlib.pyplot as plt
import numpy as np
fruits =['Apples','Bananas','Cherries','Dates']
sales=[400,150,300,450]
plt.bar(fruits,sales)
plt.xlabel('Fruits')
plt.ylabel('Sales')
plt.title('Fruit Sales')
plt.show()

#Create a Basic Histogram in Matplotlib:
import matplotlib.pyplot as plt
import numpy as np
data=np.random.randn(1000)
plt.hist(data,bins=30,color='skyblue','edgecolor='black')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Basic Histogram')
plt.show()

#Customized Histogram in Matplotlib with Density Plot:
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
data=np.random.randn(1000)
sns.histplot(data,bins=30,kde=True,color='lightgreen',edgecolor='red')
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('Customized Histogram with Density Plot')
plt.show()

#Matplotlib Scatter:
import matplotlib.pyplot as plt
import numpy as np
x=np.array([12,45,7,32,89,54,23,67,14,91])
y=np.array([99,31,56,19,88,43,61,35,77,25])
plt.scatter(x,y)
plt.title('Basic Scatter Plot')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.grid()
plt.show()

x1=np.array([160,165,170,175,180,185,190,195,200,205])
y1=np.array([55,58,60,62,64,66,68,70,72,74])
x2=np.array([150,155,160,165,170,175,180,195,200,205])
y2=np.array([50,52,54,56,58,60,62,64,66,68])
plt.scatter(x1,y1,color='red',label='Group 1')
plt.scatter(x2,y2,color='pink',label='Group 2')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Comparison of Height vs Weight between two groups')
plt.legend()
plt.show()

x=np.array([3,12,9,20,5,18,22,11,27,16])
y=np.array([95,55,63,77,89,50,41,70,58,83])
a=[20,50,100,200,500,1000,60,90,150,300]
b=['red','green','blue','purple','orange','black','pink','brown','yellow','cyan']
plt.scatter(x,y,s=a,c=b,alpha=0.5,edgecolor='W',linewidth=1)
plt.title('Scatter Plot with Varying Colors and Sizes')
plt.show()

x=[1,2,3,4,5]
y=[2,3,5,7,11]
sizes=[30,80,150,200,300]
plt.scatter(x,y,s=sizes,alpha=0.5,edgecolor='blue',linewidth=2)
plt.title("Bubble Plot Example")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

x=np.random.randint(50,150,100)
y=np.random.randint(50,150,100)
colors=np.random.rand(100)
sizes=20*np.random.randint(10,100,100)
plt.scatter(x,y,c=colors,s=sizes,cmap='viridis',alpha=0.7)
plt.colorbar(label='Color Scale')
plt.title('Scatter Plot with Colormap and Colorbar')
plt.show()

#Box Plot in Python using Matplotlib:
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(10)
d=np.random.normal(100,20,200)
fig=plt.figure(figsize=(10,7))
plt.boxplot(d)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(10)
d_1=np.random.normal(100,10,200)
d_2=np.random.normal(90,20,200)
d_3=np.random.normal(80,30,200)
d_4=np.random.normal(70,40,200)
d=[d_1,d_2,d_3,d_4]
fig=plt.figure(figsize=(10,7))
ax=fig.add_axes([0,0,1,1])
bp=ax.boxplot(d)
plt.show()
#How to Change Line Color in Matplotlib?
plt.xlabel('Population')
plt.ylabel('Year')
plt.title('Shwarma Population')
plt.plot([100,200,300,400,500,600],[1950,1960,1970,1980,1990,2000],'orange')
plt.show()

#Matplotlib.pyplot.legend() in Python
import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[1,4,9,16,25]
plt.plot(x,y)
plt.legend(['single element'])
plt.show()

import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(0,10,1000)
fig,ax=plt.subplots()
ax.plot(x,np.sin(x),'--b',label='Sine')
ax.plot(x,np.cos(x),c='r',label='Cosine')
ax.axis('equal')
leg=ax.legend(loc='lower left')
plt.show()

#Style Plots using Matplotlib:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
data=np.random.randn(50)
plt.style.use('Solarize_Light2')
plt.plot(data)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
with plt.style.context('dark_background'):
    plt.plot(np.sin(np.linspace(0,2*np.pi)),'r-o')
plt.show()
#Matplotlib Subplots
import matplotlib.pyplot as plt
import numpy as np
x1=np.array([1,2,3,4])
y1=np.array([10,20,25,30])
plt.subplot(1,2,1)
plt.plot(x1,y1)
x2=np.array([1,2,3,4])
y2=np.array([30,25,20,10])
plt.subplot(1,2,2)
plt.plot(x2,y2)
plt.show()

#Box plots in subplots:
import matplotlib.pyplot as plt
import pandas as pd
data={'Category':['A','B','C','D'],'Value1':np.random.rand.int(1,10,4),'Value2':np.random.rand.int(1,10,4),'Value3':np.random.rand.int(1,10,4)} 
df=pd.DataFrame(data)
fig,axes=plt.subplots(nrows=1,ncols=3,figsize=(12,4))
df.plot(kind='bar',x='Category',y='Value1',color='Skyblue',ax=axes[0])
df.plot(kind='bar',x='Category',y='Value2',color='lightgreen',ax=axes[1])
df.plot(kind='bar',x='category',y='value3',color='coral',ax=axes[2])
axes[0].set_title('Value1 Distribution')
axes[1].set_title('Value2 Distribution')
axes[2].set_title('Value3 Distribution')
plt.tight_layout()
plt.show()