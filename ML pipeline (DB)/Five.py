#Step 1: Loading a Dataset
from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data
y=iris.target
feature_names=iris.feature_names 
target_names=iris.target_names
print("Feature names:",feature_names)
print("Target names:",target_names)
print(type(x))
print(x[:5])

##Imp notes--
Classification (তথ্যকে শ্রেণিতে ভাগ করা)
Regression (সংখ্যাগত মান পূর্বাভাস দেওয়া)
Clustering (ডাটা গ্রুপ করা)
Data preprocessing (ডাটা পরিষ্কার ও প্রস্তুত করা)
Model evaluation (মডেল কেমন কাজ করছে তা পরীক্ষা করা)
##Step 2: Train-Test Split
আমরা dataset-কে দুই ভাগে ভাগ করি —
Training set → মডেলকে শেখানোর জন্য (বেশিরভাগ ডাটা এখানে যায়)
Testing set → মডেলের পারফরম্যান্স যাচাই করার জন্য

#Step 2: Splitting the Dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris=load_iris()
x=iris.data()
y=iris.target()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=1)
print(x_train.shape)

#Step 3: Handling Categorical Data
Label Encoding
প্রতিটি category কে একটি সংখ্যা দেওয়া হয়।
One-Hot Encoding
প্রতিটি category এর জন্য নতুন কলাম তৈরি হয়।

from sklearn.preprocessing import LabelEncoder
categorical_feature=['cat','dog','dog','cat','bird']
encoder=LabelEncoder()
encoded_feature=encoder.fit_transform(categorical_feature)
print('Encoded feature:',encoded_feature)

from sklearn.preprocessing import LabelEncoder
import pandas as pd
data=pd.DataFrame({'Fruit':['Apple','Banana','Orange','Apple','Banana','Orange'],
                        'Price':[1.2,0.5,0.8,1.3,0.9,0.6]})
encoder=LabelEncoder()
data['Fruit_Encoded']=encoder.fit_transform(data['Fruit'])
print(data)

from sklearn.preprocessing import OneHotEncoder
import numpy as np
categorical_feature=['cat','dog','dog','cat','bird']
categorrical_feature=np.array(categorical_feature).reshape(-1,1)
encoder=OneHotEncoder(sparse_output=False)
encoded_feature=encoder.fit_transform(categorical_feature)
print(encoded_feature)
##Mean_coding:
df.groupby(['SubjectName'])['Target'].cout()
df.groupby(['SubjectName'])['Target'].mean()

#Step 4: Training The Model
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression(max_iter=200)
log_reg.fit(x_train,y_train)

##Extra information from chatGPT
#Step 5:Make Predictions
y_prd=log_reg.predict(x_test)
print("Predicted labels:",y_pred[:10])
print('Actual labels:',y_test[:10])
accuracy=log_reg.score(x_test,y_test)
print('\nModel Accuracy:',accuracy)

#Step 6: Evaluating Model Accuray:
from sklearn import matrics
print('Logistic Regression model accuracy:'matrics.accuracy_score(y_test,y_pred))


--------------
#Step 4: Model Evaluation(Advance)

আমরা দুইটি প্রধান জিনিস দেখবো:
Confusion Matrix — কোন ক্লাসের কতগুলো সঠিক ও ভুল পূর্বাভাস হয়েছে সেটা দেখায়।
Classification Report — precision, recall, f1-score ইত্যাদি মেট্রিক্স দেয়।
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
# Prediction
y_pred = model.predict(X_test)
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
# Heatmap দিয়ে Confusion Matrix দেখানো
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
# Classification Report
cr = classification_report(y_test, y_pred, target_names=iris.target_names)
print("Classification Report:\n", cr)
