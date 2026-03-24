#Method 1: Creating a matrix with a List of list
mat=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]
print("Matrix =",mat)
#Method 2: Take Matrix input from user in Python
rows=int(input("rows:"))
cols=int(input('cols:'))
matrix=[]
print('Entries row-wise:')
for i in range(rows):
    row=[]
    for j in range(cols):
        row.append(int(input()))
    matrix.append(row)
print('\n2D matrix is:')
for i in range(rows):
    for j in range(cols):
        print(matrix[i][j],end='')
    print()
    
rows = int(input('rows:'))
col=int(input('cols:'))
matrix=[]
print('Entries row-wise:')
for i in range(rows):
    row=[]
    for j in range(col):
        row.append(int(input))
    matrix.append(row)
print('\n2D matrix is:')
for i in range(rows):
    for j in range(col):
        print(matrix[i][j],end='')
    print()
#Method 3: Create a matrix using list comprehension:
matrix=[[col for col in range(4)]for row in range(4)]
print(matrix)
a=[2,3,4,5]
res=[val**2 for val in a]
print(res)
a=[1,2,3,4,5]
res=[]
for val in a:
    res.append(val*2)
print(res)
a=[1,2,3,4,5]
res=[val*2 for val in a]
print(res)
#Creating a list from a range:
a=[i for i in range(10)]
print(a)
#Using nested loops:
coordinates =[(x,y)for x in range(3) for y in range(3)]
print(coordinates)
#Flattering a list of lists:
mat=[[1,2,3],[4,5,6],[7,8,9]]
res=[y for row in mat y in row]
print(res)
#make a list
a=list((1,2,3,'apple',4.5))
print(a)
#append(),extend(),insert():
a=[]
a.append(10)
print(a)
a.extend([10,20])
print(a)
a.insert(0,9)
print(a)
#Iterating over lists:
a=['apple','banana','cherry']
for item in a:
    print(item)
#Syntax of lambda
lst=list(map(lambda x:x**2,range(1,5)))
print(lst)
#Using zip()
a=[1,2,3,4]
b=[5,6,7,8]
res=[x+y for x,y in zip(a,b) if x+y>10]
print(res)
#Using enumerate():
a=['Python','is','fun!']
b=['Learn','with','GFG']
res=[(i,x,y) for i,(x,y) in enumerate(zip(a,b))]
print(res)

**List comprehension means condition+loop+expression in a one line
**Map() need a function

#Assiging Value in a mtarix
x=[[1,2,3],[4,5,6],[7,8,9]]
x[1][1]= 11 
print(x)

** Properties of Matrix Addition:
    Closure Property: A+B=C
    Commutative Property: A+B=B+A
    Associative Property: A+(B+C)= (A+B)+C
    Additive Identity: A+O=A=O+A
    Additive Inverse: A+B=0
#Exp 1: Addition Using Loops
x=[[1,2,3],[4,5,6],[7,8,9]]
y=[[9,8,7],[6,5,4],[3,2,1]]
res={[0]*3 for _ in range(3)}
for i in range(len(x)):
    for j in range(len(x[0])):
        res[i][j]=x[i][j]+y[i][j]
for r in res:
    print(r)
 #Exp 2:Addition & subtraction with list comprehension
 x=[[1,2,3],[4,5,6],[7,8,9]]
 y=[[9,8,7],[6,5,4],[3,2,1]]
 add=[[x[i][j]+y[i][j]for j in range(len(x[0]))]for i in range(len(x))]
 print("Matrix Addition:")
 for r in add:
     print(r)
sub=[[x[i][j]-y[i][j]for j in range(len(x[0]))]for i in range(len(x)] 
print('\nMatrix Subtraction:')
for r in sub:
    print(r)
#Exp3: Python program to multiply & divide:
x = [[2, 4, 6],
    [8, 10, 12],
    [14, 16, 18]]

y = [[1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]]
mul=[[x[i][j]*y[i][j]for j in range(3)]for i in range(3)]
print("Matrix Multiplication:")
for r in mul:
  print(r)
div=[[x[i][j]//y[i][j]for j in range(3)]for i in range(3)]
print('\nMatrix Division:')
for r in div:
  print(r)
  
 *Transpose of a Matrix: A= A^T
 *Diagonal matrix: the numbers 2, 3, and 5 appear along the diagonal, while all other entries are zeros. 
 *Minor: a22,a12 etc
 *Scalar matrix: constant(k)*identity matrix
 *Orthogonal Matrix: A^T = A-1 Or AA^T = A^TA = I

#Transpose of a matrix:
transpose=[[0]*3 for _ in range(3)]
for i in range(len(x)):
    for j in range(len(x[0])):
        transpose[j][i]=x[i][j]
for r in transpose:
    print(r)

transpose=[[x[j][i]for j in range(len(x))]for i in range(len(x[0]))]
for row in transpose:
    print(row)

#Basic Math operation with numpy:
import numpy as np
x=np.array([[1,2],[4,5]])
y=np.array([[7,8],[9,10]])
print('Addition:\n',np.add(x,y))
print('Subtraction:\n',np.subtract(x,y))
print('Multiplication:\n',np.multiply(x,y))
print('Division:\n',np.divide(x,y))

#Deleting Row with NumPy:
axis=0 → row delete
axis=1 দিলে → column delete হবে
axis=0 মানে row বরাবর কাজ করো, তাই row বাদ যাবে বা aggregate হবে।
axis=1 মানে column বরাবর কাজ করো, তাই column বাদ যাবে বা aggregate হবে।
import numpy as np
a = np.array([[1, 2, 3],
              [4, 5, 6]])

print(np.sum(a, axis=0))  # প্রতিটি column এর যোগফল [5, 7, 9]
print(np.sum(a, axis=1))  # প্রতিটি row এর যোগফল [6, 15]

#Pandas-এও 
axis=0 → কাজ হবে row direction-এ (মানে প্রতিটি column ধরে অপারেশন হবে)।
এটাকে অনেক সময় বলা হয় “index axis”।
axis=1 → কাজ হবে column direction-এ (মানে প্রতিটি row ধরে অপারেশন হবে)।
এটাকে বলা হয় “columns axis”।
Exp:
    import pandas as pd
df = pd.DataFrame({"A":[1,2,3],
                   "B":[4,5,6]})
print(df.sum(axis=0))  
# প্রতিটি column এর sum
# A     6
# B    15
print(df.sum(axis=1))  
# প্রতিটি row এর sum
# 0     5
# 1     7
# 2     9

#Eigenvalue & Eigenvector:
Eigenvector → a direction that doesn’t change under the transformation of the matrix.
Eigenvalue → the factor (stretch or shrink amount) by which the eigenvector is scaled.
-----Used in data science (PCA)

#LU Decomposition:
L: A lower triangular matrix with ones on the diagonal.
U: An upper triangular matrix.
 1st -- AX=B
        A=LU
      So,LUX=B
      Let,(UX)=Y;
      LY=B
Applications of LU Decomposition:
    *Structural Engineering
    *Computer Graphics
    *Robotics
    *Weather Prediction
    *Electrical Engineering
    *Economics and Finance

#QR Decomposition:
      A=QR
      *A is the original matrix
      *Q is orthogonal matrix
      *R is upper triangular matrix
#QR Decompostion using python: Q=orthogonal, R=upper triangular
import numpy as np
arr=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr)
q,r=np.linalg.qr(arr) 
print('\nQ:\n',q)
print('\nR:\n',r)
print(np.allclose(arr,np.dot(q,r)))

#Creating Vectors in NumPy:
import numpy as np
vector1=np.array([1,2,3])
vector2=np.array([[10],[20],[30]])
print('Horizontal Vector:')
print(vector1)
print("------")
print('Vertical Vectors:')
print(vector2)
#Using np.arange(0
import numpy as np
vector = np.arange(1,6)
print('Vector using np.arange():',vector)
#Using np.linspace()
vector=np.linspace(0,1,5)
print(vector)
#Using zeros() & Ones():
vector = np.zeros(5)
print(vector)
vector1=np.ones(6)
print(vector1)
#Vector Dot Product:
import numpy as np
vector1=np.array([5,6,9])
vector2=np.array([1,2,3])
print('First vector:'+str(vector1))
print('Second vector:'+str(vector2))
dot_product=vector1.dot(vector2)
print('Dot Product:'+str(dot_product))
-----------
Vector space: vectors এর set + addition & scalar multiplication rules
Subspace: vector space-এর subset যা নিজেই vector space সব রুলস মানে

#Regression Analysis:
*Types of Supervised Learning in Machine Learning:
    1.Classification/defined Labels(YES/NO,spam/non-spam)
    2.Regression/no Labels defined(predicting house prices,stock prices)
#What is Linear Regression?
Linear Regression
কাজ: কোনো সংখ্যাগত (continuous) মান predict করা।
Example: বাড়ির দাম, মানুষের বয়স, স্টক প্রাইস ইত্যাদি।
Here Independent variable(Input) & Dependent variable(Output)
   Y= mX+c
Solves regression problems

#what is Logistic Regression?
কাজ: কোনো outcome Yes/No / 0/1 / Category হিসেবে predict করা।
Nature: Output সবসময় probability (0 থেকে 1 এর মধ্যে) দেয়।
যেমন: 0.8 → মানে 80% chance Yes।
*Types of Logistic Regression-- Binomial(দুইটা class Yes/No)/Multinomial(একাধিক class Cat/Dog/Horse)/ordinal(Poor < Average < Good < Excellent)
*Solves classification problems
------------------------
#Binomial Logistic Regression:
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X,y=load_breast_cancer(return _X_y=True)
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.20,random_state=23)
clf=LogisticRegression(max_iter=10000,random_state=0)
clf.fit(X_train,y_train)
acc=accuracy_score(y_test,clf.prdict(x_test)*100)
print(f"Logistic Regression model accuracy: {acc:.2f}%")

#Multinomial Logistic Regression:
from sklearn.model_selection import train_test_split
from sklearn import datasets,linear_model,metrics
digits=datasets.load_digits()
x=digits.data
y=digits.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=1)
reg=linear_model.LogisticRegression(max_iter=10000,random_state=0)
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
print(f'Logistic Regression model accuracy:{metrics.accuracy_score(y_test,y_pred)*100:.2f}%')
--------------
#Simple Analogy (Exam Example)
Train data = Student এর পড়ার বই।
Test data (X_test) = Exam question।
Model’s prediction (y_pred) = Student এর answer।
y_test = আসল correct answer (answer sheet)।
accuracy_score(y_test, y_pred) = Student এর কতটা সঠিক উত্তর হয়েছে সেটা মাপা।
-------


**Statistic***
#Average:
import numpy as np
list=[2,3,4,5]
res=np.average(list)
print(res)
#Variance:
Variance is the sum of squares of differences between all numbers and means. 
import numpy as np
list=[2,4,4,4,5,5,7,9]
print(np.var(list))
#Standard Deviation:
Standard Deviation is the square root of variance.
StandardDeviation = variance
import numpy as np
list=[2,4,4,4,5,5,7,9]
print(np.std(list))​

Simple error → হাতে থাকা dataset এ মাপা error। Example: test data তে model accuracy 96% হলো → মানে error = 4% (এটাই simple/test error)
True error → আসল পৃথিবীর (unseen data) error, যেটা আমরা শুধু অনুমান করতে পারি। EXM:আমাদের model test set এ 96% accurate, কিন্তু আসল new patients এর উপর চালালে হয়তো 92% accurate হবে। সেই 8% হলো true error, যেটা আমরা শুধু estimate করতে পারি।
------
| Test / Interval         | কবে ব্যবহার হয়?                                 |
| ----------------------- | ----------------------------------------------- |
| **Confidence Interval** | Population mean এর range অনুমান করতে(আমরা 95% নিশ্চিত population mean এই range এ)            |
| **Z-test**              | বড় sample + σ জানা থাকলে hypothesis test করতে   |
| **t-test**              | ছোট sample + σ অজানা থাকলে hypothesis test করতে |
Confidence Interval (CI) আসলে কী?
আমরা যখন পুরো population থেকে data নিতে পারি না, তখন sample নিই।
সেই sample থেকে population parameter (যেমন population mean μ) কে estimate করি।
উদাহরণ: ১০ লাখ মানুষের গড় উচ্চতা বের করতে চাই। সব measure করা সম্ভব না। তাই ২০০ জনের sample নেই। তাদের sample mean থেকে আমরা CI বের করি → population mean কোথায় থাকতে পারে সেটা range আকারে পাই।
#Hypothesis = একটি ধরনা / অনুমান যা পরীক্ষা করা হয়|

Covariance: দুইটি variable একসাথে কতটা পরিবর্তন করছে তা measure করে।
Correlation: দুই variable কতটা linearly related তা scale-free measure। (+1 positive,-1 negative,0 non-correlation)
-------
#Hypothesis testing
Factory বলে, চকলেটের গড় weight = 50g
তুমি sample নিয়ে পরীক্ষা করছো → Hypothesis Testing দিয়ে দেখবে claim (50g) সত্যি কি না
#Chi-Square Test (χ²-Test):
Categorical data (jaise: Yes/No, Red/Blue/Green) er modhye relation ache ki na check korte.

**Geometry**
#Using np.linalg.norm()
import numpy as np
p1=np.array((1,2,3))
p2=np.array((1,1,1))
d=np.linalg.norm(p1-p2)
print(d)

#Inverse sine can be written in two ways:
sin-1 x
arcsin x
Ekbar try korle → Bernoulli
Koyekbar try kore success count korle → Binomial
Kono fixed time e koto event ghote → Poisson
Shobai jodi shoman chance pay → Uniform
#Easy Tricks:
গুনতে পারলে → Discrete
মাপতে হলে → Continuous