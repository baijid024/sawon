###import numpy
import numpy as np
import pandas as pd

###A list of Numpy Data Types
dtypes = pd.DataFrame({
'Type':['int8','uint8','int16','uint16','int32','uint32','int64','uint64','float16','float32','float64','float128','complex64','complex128','bool','object','string_','unicode'],
'Type Code' :['i1','u1','i2','u2','i4','u4','i8','u8','f2','f4 or f','f8 or d','f16 or g','c8','c16','','o','S','U'])
dtypes

arr = np.array([1,2,3],dtype='f4')
print(arr)
print(arr.dtype)

arr=np.array([1+2j,3-2j],dtype=np.complex64)
print(arr)
print(arr.dtype)

arr=np.array([0,1,1],dtype=np.bool)
print(arr)
print(arr.dtype)

###create an array from a Python array
arr=np.array(range(10))
print(arr)

arr=np.array([1,2,3,4,5])
print(arr)

###create an array in a specified data type
arr=np.array([[1,2,3],[4,5,6]],dtype='i2')
print(arr)

###create an aray of evenly spaced values within a specified interval
arr=np.arange(0,20,2)
print(arr)

arr=np.linspace(0,10,20)
print(arr)

arr, step = np.linspace(0,10,20, endpoint=False, retstep=True)
print(arr)
print(step)

arr=np.random.rand(3,3)
print(arr)

###create an array of zeros in a given shape
zeros = np.zeros((2,3), dtype='i4')
print(zeros)

ones = np.ones((2,3))
print(ones)

empty = np.empty((2,3))
print(empty)

p=np.full((2,3),5)
print(p)

arr=[0,1,2]
print(np.repeat(arr,3))

arr = [[1,2],[3,4]]
print(np.repeat(arr,[1,2],axis=0))

identity_matrix = np.eye(3)
print(identity_matrix)

identity_matrix = np.identity(3)
print(identity_matrix)

identity_matrix = np.eye(5,k=1)
print(identify_matrix)

arr = np.diag([1,2,3,4,5])
print(arr)

arr= np.array([[1,2,3],[4,5,6]],dtype=np.int64)
print(arr)

print(np.info(arr))
print(np.info(arr))

print(arr.dtype)
print(arr.shape)
print(arr.size)
print(arr.itemsize)
print(np.info(arr))
print(len(arr))
print(arr.nbytes)
print(arr.ndim)

print(np.random.shuffle(arr))

arr = np.array(range(100)).reshape((10,10))
print(arr[5][5])

arr = np.array(range(100)).reshape((10,10))
arr[1:3,:]=100
arr[:,8:]=100
print(arr)

arr=np.array(range(16)).reshape((4,-1))
print(arr)

arr1 =np.array([[1,2,3,4],[1,2,3,4]])
arr2 =np.array([[5,6,7,8],[5,6,7,8]])
cat=np.concatenate((arr1,arr2),axis=1)
print(cat)
