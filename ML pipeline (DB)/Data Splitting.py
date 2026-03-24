import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('batting_summary.csv')

# Data preprocessing
# Handle missing values in SR column
df['SR'] = pd.to_numeric(df['SR'], errors='coerce')
df['SR'].fillna(df['SR'].mean(), inplace=True)

# Create target variable (high performance indicator)
df['high_performance'] = (df['runs'] > 30).astype(int)

# Encode categorical variables
le = LabelEncoder()
df['team_encoded'] = le.fit_transform(df['teamInnings'])
df['out_status'] = le.fit_transform(df['Out/Not_Out'])

# Select features and target
features = ['battingPos', 'runs', 'balls', '4s', '6s', 'SR', 'team_encoded', 'out_status']
X = df[features]
y = df['high_performance']

# Check the distribution of the target variable
print("Target variable distribution:")
print(y.value_counts())
print(f"\nPercentage of high performers: {y.mean()*100:.2f}%")

# 1. Basic Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nBasic Split:")
print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# 2. Train-Validation-Test Split (60% train, 20% validation, 20% test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\nTrain-Validation-Test Split:")
print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# 3. Check distribution in each split
def check_distribution(y_data, dataset_name):
    print(f"{dataset_name}: {len(y_data)} samples, "
          f"{y_data.sum()} high performers ({y_data.mean()*100:.1f}%)")

print("\nDistribution across splits:")
check_distribution(y_train, "Training set")
check_distribution(y_val, "Validation set")
check_distribution(y_test, "Test set")

# 4. Visualize the splits
plt.figure(figsize=(10, 6))

# Target distribution
plt.subplot(1, 2, 1)
split_names = ['Training', 'Validation', 'Test']
split_sizes = [len(y_train), len(y_val), len(y_test)]
plt.bar(split_names, split_sizes, color=['blue', 'orange', 'green'])
plt.title('Dataset Split Sizes')
plt.ylabel('Number of Samples')

# Percentage of high performers
plt.subplot(1, 2, 2)
high_perf_percent = [y_train.mean()*100, y_val.mean()*100, y_test.mean()*100]
plt.bar(split_names, high_perf_percent, color=['blue', 'orange', 'green'])
plt.title('Percentage of High Performers')
plt.ylabel('Percentage (%)')
plt.axhline(y=y.mean()*100, color='red', linestyle='--', label='Overall Average')
plt.legend()
   y.mean() → তোমার target variable (y) এর গড় বের করছে।
    উদাহরণ: যদি y হয় 0 আর 1 (high performer = 1, low performer = 0), তাহলে mean() আসলে high performer এর proportion (percentage) দেয়।

plt.tight_layout()
plt.show()

# 5. Save the splits to CSV files (optional)
X_train.to_csv('X_train.csv', index=False)
X_val.to_csv('X_val.csv', index=False)
X_test.to_csv('X_test.csv', index=False)

y_train.to_csv('y_train.csv', index=False)
y_val.to_csv('y_val.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("\nSplits saved to CSV files: X_train.csv, X_val.csv, X_test.csv, y_train.csv, y_val.csv, y_test.csv")

# 6. Advanced: K-Fold Cross Validation (for more robust evaluation)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Initialize model
model = RandomForestClassifier(random_state=42)

# Perform 5-fold cross validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"\n5-Fold Cross Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# 7. Time-based splitting (if your data has temporal order)
# Assuming matchID represents temporal order
df_sorted = df.sort_values('matchID')
X_sorted = df_sorted[features]
y_sorted = df_sorted['high_performance']

# Time-based split: 70% train, 15% validation, 15% test
train_size = int(0.7 * len(X_sorted))
val_size = int(0.15 * len(X_sorted))

X_train_time = X_sorted.iloc[:train_size]
X_val_time = X_sorted.iloc[train_size:train_size+val_size]
X_test_time = X_sorted.iloc[train_size+val_size:]

y_train_time = y_sorted.iloc[:train_size]
y_val_time = y_sorted.iloc[train_size:train_size+val_size]
y_test_time = y_sorted.iloc[train_size+val_size:]

print(f"\nTime-based Split:")
print(f"Training set: {X_train_time.shape[0]} samples")
print(f"Validation set: {X_val_time.shape[0]} samples")
print(f"Test set: {X_test_time.shape[0]} samples")

# 8. Scaling features (important for many algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed!")
print("Use X_train_scaled, X_val_scaled, X_test_scaled for models that require feature scaling")



###Ensemble মানে কী?
ensemble মানে হলো একাধিক model একসাথে ব্যবহার করে final decision নেয়া।
যেমন ধরো, তুমি একা কোনো exam এর উত্তর দিচ্ছো → ভুল হওয়ার chance বেশি।
কিন্তু যদি ১০ জন মিলে vote দিয়ে উত্তর দেয়, তাহলে final answer বেশি accurate হওয়ার chance থাকে।
👉 Machine Learning এ ensemble methods এর উদাহরণ:
Random Forest → অনেকগুলো Decision Tree বানিয়ে তাদের majority vote নেয়।
Gradient Boosting → একটার পরে একটা ছোট model বানায়, আগের ভুলগুলো ঠিক করে।
তাই RandomForestClassifier হলো একটি ensemble model (কারণ এটি অনেকগুলো decision tree এর ensemble)।

👉 তাই Time-based split মানে হলো ভবিষ্যতের ডেটা আলাদা রাখা আর মডেল কেবল past ডেটা দিয়ে ট্রেন করা।

🔑 সংক্ষেপে:
.shape[0] → কত row আছে = কতগুলো sample
.shape[1] → কত column আছে = কতগুলো feature

Cross Validation (CV) মানে কী?
Cross-validation হলো model performance test করার একটা method যেখানে dataset কে একবারে train-test split না করে, বারবার ভাগ করে model evaluate করা হয়।