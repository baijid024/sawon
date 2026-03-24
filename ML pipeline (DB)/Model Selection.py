🔎 Model Selection in Machine Learning
Model selection মানে হলো – একটা নির্দিষ্ট ML problem এর জন্য কোন algorithm / কোন মডেল সবচেয়ে ভালো কাজ করছে সেটা বেছে নেওয়া।

✅ কেন Model Selection দরকার?
একই dataset-এ:
Logistic Regression, Decision Tree, Random Forest, SVM, Neural Network ইত্যাদি ভিন্ন ভিন্নভাবে perform করতে পারে।
সব মডেলেরই strengths & weaknesses আছে।
তাই আমাদের compare করে best performing model নির্বাচন করতে হয়।

✅ কেন দরকার?
ভিন্ন ভিন্ন algorithm ভিন্নভাবে কাজ করে
Logistic Regression → simple, fast, কিন্তু non-linear data handle করতে পারে না।
Decision Tree → complex relation ধরতে পারে, কিন্তু overfit করতে পারে।
Random Forest / XGBoost → powerful, কিন্তু computation heavy।
তাই একেক dataset-এ একেক model ভালো perform করবে।

🔧 Model Selection Process:
Data Split-------
Dataset কে Train, Validation, Test এ ভাগ করা হয়।
Train দিয়ে মডেল শেখানো হয়, Validation দিয়ে tuning করা হয়, আর Test দিয়ে final performance দেখা হয়।

Try Multiple Algorithms---
Example: Logistic Regression, Random Forest, SVM, XGBoost ইত্যাদি।

Hyperparameter Tuning-----
প্রতিটি মডেলের কিছু setting থাকে (যেমন Random Forest → number of trees)।
Validation set বা Cross Validation দিয়ে কোন parameter ভালো কাজ করছে সেটা বের করা হয়।

Evaluation Metrics----
Classification problem হলে → Accuracy, Precision, Recall, F1-score।
Regression problem হলে → MSE, RMSE, R² score।
Business goal অনুযায়ী কোন metric important সেটা দেখে model নির্বাচন করা হয়।

Cross Validation---
Dataset কে বারবার ভাগ করে train-test করা হয় (যেমন K-Fold CV) → যাতে নিশ্চিত হওয়া যায় মডেল generalize করতে পারছে।

Final Model Selection----
যেটা Validation/Test data-তে সবচেয়ে ভালো result দিচ্ছে → সেটাকেই deploy করা হয়।


🔎 Overfitting vs Underfitting
👉 তুমি একটু ভুলভাবে ধরেছো যে accuracy “add” করতে হবে।
আসলে এখানে percentage মানে হলো — আলাদা আলাদা dataset-এ মডেল কতটা ভালো করছে।
উদাহরণ:
Overfitting Model:
Training Accuracy = 95%
Test Accuracy = 60%
👉 মানে model শুধু training data-ই মুখস্থ করেছে, নতুন data (test) এ খারাপ করছে।
⚠️ এখানে 95% + 60% = 155% কোন calculation না। দুইটা independent performance measure।

Generalized Model:
Training Accuracy = 75%
Test Accuracy = 72%
👉 মানে model শিখেছে, কিন্তু মুখস্থ করেনি। নতুন data-তেও প্রায় similar accuracy দিচ্ছে → Generalization ভালো।
✅ লক্ষ্য: Train এবং Test accuracy যেন একই রকম হয়, বড় gap না থাকে।

Key Point
Overfitting: Train high, Test low
Underfitting: Train low, Test low
Good Generalization: Train ~ Test (mid-range)

🔎 Penalty = L1 vs L2
Logistic Regression বা Linear models-এ regularization নামে একটা concept আছে।
L1 penalty (Lasso): অনেক coefficient = 0 করে দেয় (feature selection করে)।
L2 penalty (Ridge): coefficient গুলোকে ছোট করে দেয় (shrink করে), কিন্তু 0 করে না।
👉 এগুলো মডেলকে overfitting থেকে বাঁচায়।

🔎 Metrics (Model Compare করার জন্য)
Classification (Yes/No, High/Low এর মতো problem):
Accuracy = কত % সঠিক predict হলো
Precision = Positive বলেছে তার মধ্যে কয়টা সত্যি Positive
Recall = আসল Positive এর মধ্যে কয়টা ধরা পড়েছে
F1 = Precision আর Recall এর balance
AUC-ROC = মডেল Positive/Negative আলাদা করতে কতটা efficient

Regression (Continuous value যেমন দাম, মার্কস predict):
MSE = Error² এর গড়
RMSE = √MSE (error এর average size)
R² = মডেল কতটা variation explain করতে পারছে

🔎 Validation কেন দরকার?
তুমি ঠিক বলেছো – train/test split করলে তো দেখা যায় মডেল কেমন কাজ করছে। তাহলে extra validation কেন?
Case 1: শুধু Train/Test
Train দিয়ে শেখালে → model train হয়।
Test-এ accuracy মাপলে → bias-free performance পাওয়া যায়।
👉 কিন্তু, যদি তুমি hyperparameter tuning করো (যেমন Random Forest-এ কতটা tree, Logistic Regression-এ penalty), তখন তুমি test dataset বারবার use করবে।
⚠️ এতে test dataset “leak” হয়ে যাবে → আসল final unbiased test possible হবে না।

Case 2: Train/Validation/Test
Train → মডেল শেখানোর জন্য
Validation → tuning/check করার জন্য (কোন parameter best)
Test → একেবারে শেষে, একবারই ব্যবহার করা হবে (final unbiased evaluation)
👉 এভাবে Test data untouched থাকে → আসল generalization check হয়।

✅ In short:
Train/Test = basic split
Train/Validation/Test = যখন tuning দরকার হয়
Cross Validation (CV) = আরও robust (multiple train/test split)

## 1️⃣ Train-Test Split
* Dataset কে **Train** আর **Test** এ ভাগ করা।
* **Train:** মডেল শেখানোর জন্য
* **Test:** মডেলের **final performance** দেখার জন্য (unseen data)
* Problem: যদি Train-Validation/Test না করে হঠাৎ test-এ performance দেখি → মডেল biased হতে পারে।

## 2️⃣ Train-Validation-Test Split
* Dataset কে **Train, Validation, Test** এ ভাগ করা।
* **Validation:**
  * Hyperparameter tuning, model selection করার জন্য
  * Test data **safe** থাকে → final unbiased evaluation
* **Test:** শুধু শেষে মডেল evaluate করতে
💡 উদ্দেশ্য: **Test data leak রোধ করা**, মানে tuning এ test data ব্যবহার না করা।

## 3️⃣ Cross Validation (CV)
* Single validation split কখনো reliable হয় না, যদি dataset ছোট বা unbalanced হয়।
* **K-Fold CV:** Dataset কে K অংশে ভাগ করে, K বার train-test করা হয়।

  * প্রতিবার অন্য অংশ test হয়, বাকি অংশ train।
  * সমস্ত অংশ evaluation এ আসে → model performance stable & unbiased হয়
* **Benefit:**

  * ছোট dataset-এর জন্য ideal
  * Random split e performance fluctuation কমে
  * Overfitting risk detect করতে সহায়তা করে
  
  ------------------------------------------

#import libraries---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
#Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

Answer---
1. Basic Libraries
pandas (pd) → টেবিল আকারে data (DataFrame) নিয়ে কাজ করতে।
numpy (np) → numerical computation (array, matrix calculation ইত্যাদি)।
matplotlib.pyplot (plt) → data visualization (graph, chart)।
seaborn (sns) → matplotlib এর উপর ভিত্তি করে আরও সুন্দর এবং আকর্ষণীয় visualization দেয়।
👉 এই চারটা library কে তুমি বলতে পারো “data analysis আর visualization এর backbone।”
2. Preprocessing Tools
LabelEncoder → categorical data (যেমন “Male”, “Female”) কে সংখ্যায় রূপান্তর করে (0,1)।
StandardScaler → data কে normalize/standardize করে (mean=0, variance=1), যাতে model ভালভাবে train করতে পারে।
👉 একে বলো “ডাটাকে মডেলের উপযোগী করে তোলার কাজ।”
3. Dataset Splitting & Model Selection
train_test_split → dataset কে ভাগ করে (training set + testing set)।
cross_val_score → model এর performance যাচাই করার জন্য cross-validation।
GridSearchCV → best parameters খুঁজে বের করার smart technique (hyperparameter tuning)।
👉 এগুলোকে বলো “training-এর জন্য data ভাগ করা আর সেরা সেটিংস খুঁজে বের করার টুলস।”
4. Performance Metrics
accuracy_score → model কতটুকু ঠিকভাবে predict করেছে তার শতাংশ।
classification_report → precision, recall, F1-score সহ বিস্তারিত রিপোর্ট। Precision → Positive prediction কতটা ঠিক,Recall → Positive class কতটা ঠিক ধরেছে,F1-score → Precision এবং Recall এর harmonic mean,Support → প্রতিটি class এ test samples এর সংখ্যা
confusion_matrix → কোন class-এ model কতটুকু ভুল করেছে, টেবিল আকারে।
👉 একে বলো “মডেলের ফলাফল যাচাইয়ের স্কোরকার্ড।”
5. Different Models (Algorithms)
Logistic Regression → simple but effective classifier।
Decision Tree → data কে tree আকারে ভাগ করে decision নেয়।
Random Forest → অনেকগুলো tree একসাথে vote দেয়।
Gradient Boosting → sequential model, প্রতিবার error কমানোর চেষ্টা করে।
SVC (Support Vector Classifier) → data কে আলাদা boundary দিয়ে classify করে।
KNN (K-Nearest Neighbors) → কাছাকাছি neighbor এর উপর ভিত্তি করে prediction।
Naive Bayes (GaussianNB) → probability ভিত্তিক simple & fast classifier।
XGBoost → খুবই শক্তিশালী এবং fast boosting algorithm, Kaggle competitions এ খুব জনপ্রিয়।
👉 এগুলোকে তুমি বলো “বিভিন্ন রকম classifier অস্ত্রাগার (toolbox)।”

Random Forest-এ confusion কেন হয়?
Random Forest অনেকগুলো tree বানায় (যেমন 100 টি decision tree)। প্রতিটা tree আলাদা vote দেয়। কিন্তু ওই tree গুলোকে class বলা হয় না।
সেগুলোকে বলা হয় estimators (base learners)।
Random Forest শেষ পর্যন্ত target class (যেমন Spam/Not Spam) predict করে।


df=pd.read_csv('')
df

#Preprocessing function:
def preprocess_data(df):
    df_clean=df.copy()
    df_clean['SR']= pd.to_numeric(df_clean['SR'],errors='coerce')
    👉 যদি কোনো value number না হয় (যেমন string থাকে), তাহলে errors='coerce' এর কারণে সেটা NaN হয়ে যাবে।
    df_clean['SR'].fillna(df_clean['SR'].mean(),inplace=True)
    👉 মানে missing data handle করা।
    df_clean['high_performance']=(df_clean['runs']>30).astype(int)
    le=LabelEncoder()
    df_clean['team_encoded']=le.fit_transform(df_clean['teamInnings'])
    df_clean['out_status']=le.fit_transform(df_clean['Out/Not_Out'])
    return df_clean
df_processed=preprocess_data(df)

#Select Features and target:
features = ['battingPos', 'runs', 'balls', '4s', '6s', 'SR', 'team_encoded', 'out_status']
x=df_processed['features']
y=df_processed['high_performance']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 1. Initialize multiple models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}
#(another ekta formate)---- exceptional
for name, model in models.items():
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    print(name, accuracy_score(y_test, y_pred))

# 2. Evaluate models using cross-validation

cv_results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
         Cross-validation সবসময় training set এর মধ্যে করা হয়।
         Test set কখনো CV তে ঢোকানো হয় না।
    cv_results[name] = cv_scores
    print(f"{name:25} | Mean Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# 3. Visualize cross-validation results
plt.figure(figsize=(12, 6))
cv_means = [np.mean(scores) for scores in cv_results.values()]
cv_stds = [np.std(scores) for scores in cv_results.values()]
model_names = list(cv_results.keys())

plt.barh(model_names, cv_means, xerr=cv_stds, alpha=0.7)
plt.xlabel('Accuracy')
plt.title('Model Comparison using 5-Fold Cross-Validation')
plt.xlim(0, 1)
plt.tight_layout()
plt.show()

# 4. Train and evaluate models on test set
test_results = {}
for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    test_results[name] = accuracy
    
    print(f"{name:25} | Test Accuracy: {accuracy:.4f}")

# 5. Visualize test results
plt.figure(figsize=(12, 6))
model_names = list(test_results.keys())
accuracies = list(test_results.values())

plt.barh(model_names, accuracies, alpha=0.7)
plt.xlabel('Accuracy')
plt.title('Model Performance on Test Set')
plt.xlim(0, 1)
for i, v in enumerate(accuracies):
    plt.text(v + 0.01, i, f'{v:.4f}', va='center')
plt.tight_layout()
plt.show()

# 6. Hyperparameter tuning for the best model
# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Perform grid search
grid_search.fit(X_train_scaled, y_train)

# Print best parameters and score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# Evaluate on test set
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
best_accuracy = accuracy_score(y_test, y_pred_best)
print(f"Test Accuracy with Best Model: {best_accuracy:.4f}")

# 7. Feature importance from the best model
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance from Random Forest')
plt.tight_layout()
plt.show()

# 8. Detailed evaluation of the best model
print("Classification Report:")
print(classification_report(y_test, y_pred_best))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low Performance', 'High Performance'],
            yticklabels=['Low Performance', 'High Performance'])
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# 9. Compare performance before and after tuning
# Default Random Forest
default_rf = RandomForestClassifier(random_state=42)
default_rf.fit(X_train_scaled, y_train)
y_pred_default = default_rf.predict(X_test_scaled)
default_accuracy = accuracy_score(y_test, y_pred_default)

print(f"Default Random Forest Accuracy: {default_accuracy:.4f}")
print(f"Tuned Random Forest Accuracy: {best_accuracy:.4f}")
print(f"Improvement: {(best_accuracy - default_accuracy)*100:.2f}%")

# 10. Save the best model
import joblib   #joblib একটি Python library, যা large objects efficiently save/load করতে পারে। Training শেষ করার পরে model এবং scaler কে save করে রাখা, যাতে পরে আবার training না করেও new data prediction করা যায়।” Model → যেটা আমরা GridSearchCV দিয়ে best tune করেছি, সেটা save হবে Scaler → যেটা data scale করতে ব্যবহার করেছি (StandardScaler), সেটা save হবে
# Save the model
joblib.dump(best_model, 'best_cricket_model.pkl')
print("\nBest model saved as 'best_cricket_model.pkl'")

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as 'scaler.pkl'")

# 11. Create a function for making predictions
def predict_performance(batting_pos, runs, balls, fours, sixes, sr, team_encoded, out_status):
    """
    Predict if a batsman will be a high performer
    """
    # Load the model and scaler
    model = joblib.load('best_cricket_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Create feature array
    features = np.array([[batting_pos, runs, balls, fours, sixes, sr, team_encoded, out_status]])
    
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)
    
    return prediction[0], probability[0]

# Example prediction
print("\nExample Prediction:")
pred, prob = predict_performance(3, 45, 30, 5, 2, 150.0, 1, 1)
print(f"Prediction: {'High Performer' if pred == 1 else 'Low Performer'}")
print(f"Probability: {prob[1]*100:.2f}% chance of high performance")