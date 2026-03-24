#Creating a rank 1 Array:
arr=np.array([1,2,3])
print(arr)
#Creating a rank 2 Array:
arr= np.array([[1,2,3],[4,5,6]])
print(arr)
#Creating a rank 3 Array:
arr=np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(arr)

#Creating an array from tuple:
arr= np.array((1,2,3))
print(arr)

#Accessing the array Index:
import numpy as np
arr=np.array([[-1,2,0,4],[4,-0.5,6,0],[2.6,0,7,8],[3,-7,4,2.0]])
arr2=arr[:2,::2]
print(arr2)

arr3=arr[[1,1,0,3],[3,2,1,0]]
print(arr3)

#Basic Operation:
a=np.array([[1,2],[3,4])
b=np.array([[4,3],[2,1]])
print(a+1)
print(b-2)
print(a.sum())
print(a+b)

#Construction a Datatype Object:
x=np.array([1,2])
print(x.dtype)
x=np.array([1.0,2.0])
print(x.dtype)
x=np.array([1,2],dtype=np.int64)
print(x.dtype)

#Math Operation on DataType array:
import numpy as np
arr1=np.array([[4,7],[2,6],dtype=np.float64)
arr2=np.array([[3,6],[2,8],dtype=np.float64)
Sum=np.add(arr1,arr2)
print(Sum)
sum1=np.sum(arr1)
print(sum1)
sqrt=np.sqrt(arr1)
print(sqrt)
Trans_arr=arr1.T
print(Trans_arr)

#Mathematical function:
import numpy as np
import math
in_array=[0,math.pi/2,np.pi/3,np.pi]
print(in_array)
Sin_values=np.sin(in_array)
print(Sin_values)

import numpy as np
import math
in_arr=[0,math.pi/4,3*np.pi/2,math.pi/6]
print(in_arr)
tan_values=np.tan(in_arr)
print(tan_values)

#Graphical Representation:
import matplotlib.pyplot as plt
import numpy as np
in_arr=np.linspace(0,np.pi,12)
out_arr=np.tan(in_arr)
print('in_array:',in_arr)
print('out_array:',out_arr)
plt.plot(in_arr,out_arr,color='red',marker='o')
plt.title('numpy.tan()')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#arcsin()
import matplotlib.pyplot as plt
import numpy as np
in_arr=np.linspace(-np.pi,np.pi,12)
out_arr1=np.sin(in_arr)
out_arr2=np.arcsin(out_arr1)
print('in_array:',in_arr)
print('out_array:',out_arr1)
print('out_array:',out_arr2)
plt.plot(in_arr,out_arr1,color='red',marker='o')
plt.plot(in_arr,out_arr2,color='pink',marker='*')
plt.title('numpy.arcsin()')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#arctan2()
import numpy as np
arr1=[-1,+1,+1,-1]
arr2=[-1,-1,+1,+1]
ans=np.arctan2(arr2,arr1)*180/np.pi
print('X-coordinates:',arr1)
print('Y-coordinates:',arr2)
print('arctan2(Y,X) in degrees:',ans)

#cos()
import numpy as np
import math
in_arr=[0,math.pi/2,np.pi/3,np.pi]
print('Input array:\n',in_arr)
cos_values=np.cos(in_arr)
print(cos_values)

#degress
import numpy as np
import math
in_arr=[0,math.pi/2,np.pi/3,np.pi]
print(in_arr)
deg_values=np.degrees(in_arr)
print(deg_values)

#rad2deg()
import numpy as np
import math
in_arr=[0,math.pi/2,np.pi/3,np.pi]
print(in_arr)
rad2deg_values=np.rad2deg(in_arr)
print(rad2deg_values)

#hypot 
import numpy as np
leg1=[12,3,4,6]
print(leg1)
leg2=[5,4,3,6]
print(leg2)
result = np.hypot(leg1,leg2)
print(result)

import numpy as np
leg1 =np.random.rand(3,4)
print(leg1)
leg2=np.ones((3,4))
print(leg2)
result=np.hypot(leg1,leg2)
print(result)

#sinh()
import numpy as np
import math 
in_arr=[0,math.pi/2,np.pi/3,np.pi]
print(in_arr)
sinh_values = np.sinh(in_arr)
print(sinh_values)
# Graphical Representation of sinh()
import matplotlib.pyplot as plt
import numpy as np
in_array=np.linspace(-np.pi,np.pi,12)
out_array=np.sinh(in_array)
print('int_arr:\n',in_array)
print('out_arr:\n',out_array)
plt.plot(in_array,out_array,color='red',marker='o')
plt.title('numpy.sinh()')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#arcsinh()
import numpy as np
in_arr =[2,1,10,100]
print('Input array:\n',in_arr)
arcsinh_values=np.arcsinh(in_arr)
print(arcsinh_values)

import matplotlib.pyplot as plt
import numpy as np
in_arr=np.linspace(1,np.pi,18)
out_array1=np.sin(in_arr)
out_array2=np.arcsinh(in_arr)
print('in_array:\n',in_arr)
print('out_array1:\n',out_array1)
print('out_array2:\n',out_array2)
plt.plot(in_arr,out_array1,color='blue',marker='o')
plt.plot(in_arr,out_array2,color='red',marker='*')
plt.title('numpy.arcsinh()')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
#around()
import numpy as np
a=np.array([1.2,3.4,5.7])
res=np.around(a)
print(res)

import numpy as np
a=np.array([1.567,2.345,3.789])
out_array=np.empty_like(a)
np.around(a,decimals=1,out=out_array)
print(out_array)

import numpy as np
a1_zeros = np.zeros((3,3))
a2_ones = np.ones((2,2))
a3_range = np.arange(0,10,2)
print(a1_zeros)
print(a2_ones)
print(a3_range)

import numpy as np
zeros = np.zeros((2,3))
ones = np.ones((3,3))
constant_array = np.full((2,2),7)
range_array=np.arange(0,10,2)
linspace_array=np.linspace(0,1,5)
print(zeros)
print(ones)
print(constant_array)
print(range_array)
print(linspace_array)

import numpy as np
x=np.random.rand(2,3)
y=np.random.randn(2,2)
z=np.random.randint(1,10,size=(2,3))
print(x)
print(y)
print(z)

matrix=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(matrix[1,2])

arr=np.array([10,15,20,25,30])
print(arr[arr>20])

arr=np.array([10,15,20,25,30])
print(arr[(arr>10)&(arr<30)])

arr=np.array([10,20,30,40,50])
indices=[0,2,4]
print(arr[indices])

arr=np.array([1,2,3,4,5])
print(arr[[0,2,4]])

arr=np.array([1,2,3])
print(arr[:,np.newaxis])

###np.floor();(-1.7->-2)-> nicher dike(aro rinattok)####
###np.Ceil();(-1.7->-1)-> uporer dike(aro Dhonattok)####

#numpy.add()
import numpy as geek
in_num1=10
in_num2=15
print('1st Input number:',in_num1)
print('2nd Input number:',in_num2)
out_num=geek.add(in_num1,in_num2)
print('output number after addition:',out_num)

import numpy as np
array1=np.array([9,7,12])
scaler =4
result=np.add(array1,scaler)
print(result)
#np.conj()
import numpy as np
in_arr=np.eye(2)+3j*np.eye(2)
print("Input array:",in_arr)
out_arr=np.conj(in_arr)
print('output conjugated array:',out_arr)

#Key Attributes of Numpy arrays
import numpy as np
arr=np.array([[1,2,3],[4,5,6]])
print(arr.shape)
print(arr.dtype)
print(arr.ndim)

#Operations on NumPy Arrays
import numpy as np
x=np.array([1,2,3])
y=np.array([4,5,6])
print(x+y)
a=np.array([[1,2],[3,4]])
b=np.array([[5,6],[7,8]])
print(np.dot(a,b))

#np.eye()
import numpy as np
rectangular_matrix=np.eye(3,5,k=1)
print(rectangular_matrix)

#order='C'
import numpy as geek
array=geek.arange(10).reshape(5,2)
print(array)
array=geek.arange(4).reshape(2,2)
c=geek.zeros_like(array,dtype='float')
print(c)
array=geek.arange(8)
c=geek.zeros_like(array,dtype='float',order='C')
print(c)

import numpy as np
arr1d=np.array([10,20,30,40,50])
print(arr1d[2])
print(arr1d[-1])
arr2d=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr2d[1,0])
#np.sort()
import numpy as np
a=np.array([[12,15],[10,1]])
arr1=np.sort(a,axis=0)
a=np.array([[10,15],[12,1]])
arr2=np.sort(a,axis=-1)
print(arr2)
a=np.array([[12,15],[10,1]])
arr1=np.sort(a,axis=None)
print(arr1)

import numpy as np
dtypes=[('name','S10'),('grad_year',int),('cgpa',float)]
values=[('Hrithik',2009,8.5),('Ajay',2008,8.7),('Pankaj',2008,7.9),('Aakash',2009,9.0)]
arr=np.array(values,dtype=dtypes)
print(np.sort(arr,order='name'))
print(np.sort(arr,order=['grad_year','cgpa'])

#Stacking
import numpy as np
a= np.array([[1,2],[3,4]])
b= np.array([[5,6],[7,8]])
print(np.vstack((a,b)))
print(np.hstack((a,b)))
c=[5,6]
print(np.column_stack((a,c)))
print(np.concatenate((a,b),1))
#Broadcasting
a=np.array([0.0,10.0,20.0,30.0])
b=np.array([0.0,1.0,2.0])
print(a[:,np.newaxis]+b)

#Working with Datetime
import numpy as np
today=np.datetime64('2017-02-12')
print('Date is:',today)
print('Year is:',np.datetime64(today,'Y')
dates=np.arange('2017-02','2017-03',dtype='datetime64[D]')
print(dates)
print(today in dates)
dur=np.datetime64('2017-05-22)-np.datetime64('2016-05-22')
print(dur)
print(np.timedelta64(dur,'W')
a=np.array(['2017-02-12','2016-10-13','2019-05-22'],dtype='datetime64')
print(np.sort(a))

#Linear algebra in Numpy:
A=np.array([[6,1,1],[4,-2,5],[2,8,7]])
print(np.linalg.matrix_rank(A))
print(np.trace(A))
print(np.linalg.det(A))
print(np.linalg.inv(A))
print(np.linalg.matrix_power(A,3))

a=np.array([[1,2],[3,4]])
b=np.array([8,18])
print(np.linalg.solve(a,b))

a=np.arange(10,1,-2)
print(a)
newarr=a[np.array([3,1,2])]
print(newarr)

a=np.arange(20)
print(a)
print(a[15])
print(a[-8:17:1])
print(a[10:])
#Ellipsis
import numpy as np
b=np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(b[...,1])
#Using purely integer array indexing
import numpy as np
a = np.array([[1 ,2 ],[3 ,4 ],[5 ,6 ]])
print(a[[0 ,1 ,2 ],[0 ,0 ,1]])

#using dict.fromkeys()
a=[1,2,1,1,3,4,3,3,5]
res=list(dict.fromkeys(a))
print(res)
#Using loop():
a=[1,2,1,1,3,4,3,3,5]
res=[]
for item in a:
  if item not in res:
    res.append(item)
print(res)
#Using math.prod()
import math
a=[2,4,8,3]
res=math.prod(a)
print(res)
#Using Statistics
import statistics
a=[2,4,6,8,10]
avg=statistics.mean(a)
print(avg)

#Using Flatten()& Ravel()
import numpy as np
a=np.array([(1,2,3,4),(3,1,4,2)])
print(a)
c=a.flatten()
print(c)
ra=np.array([(1,2,3,4),(3,1,4,2)])
print(ra)
ra1=np.ravel(ra)
print(ra1)




| Framework              | `axis=0`                                    | `axis=1`                                    |
| ---------------------- | ------------------------------------------- | ------------------------------------------- |
| **NumPy**              | Row-wise operation (প্রতিটি column ধরে ধরে) | Column-wise operation (প্রতিটি row ধরে ধরে) |
| **Pandas** (DataFrame) | Row drop/filter (row গুলোতে কাজ)            | Column drop/filter (column গুলোতে কাজ)      |

Tricks-----
import numpy as geek
x=geek.
!...
constant_array=np.full((2,7),7)
constant_array=np.full([2,7],7)  #Both are correct.....
2....
index call korle always eibhabe hobee
np.array([:2,::2])
np.array([indices]) #indices=np.array([1,3,5])
np.array([arr>20])#condition decler korleo eibhabe dite hoy
arr=np.array([1,2,3,4,5])
print(arr[[0,2,4]]) #another type...
3...
Uaing np.newaxis to Add New Dimensions
arr=np.array([1,2,3])
print(arr[:,np.newaxis])
4.
axis=0 → vertical  ↑↓  (columns)
axis=1 → horizontal  ←→  (rows)
axis=None → ignore shape, treat as 1D flat list
5.
out_arr = geek.random.randint(2, 10, (2, 3, 4)) #2-> koyta table hbe, 3->row, 4->col
