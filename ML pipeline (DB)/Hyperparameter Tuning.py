import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('batting_summary.csv')

def preprocess_data(df):
    df_clean = df.copy()
    df_clean['SR'] = pd.to_numeric(df_clean['SR'], errors='coerce')
    df_clean['SR'].fillna(df_clean['SR'].mean(), inplace=True)
    df_clean['high_performance'] = (df_clean['runs'] > 30).astype(int)
    
    le = LabelEncoder()
    df_clean['team_encoded'] = le.fit_transform(df_clean['teamInnings'])
    df_clean['out_status'] = le.fit_transform(df_clean['Out/Not_Out'])
    
    return df_clean

df_processed = preprocess_data(df)

# Features and target
features = ['battingPos', 'runs', 'balls', '4s', '6s', 'SR', 'team_encoded', 'out_status']
X = df_processed[features]
y = df_processed['high_performance']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Data prepared for hyperparameter tuning")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# 1. RANDOM FOREST HYPERPARAMETER TUNING

# Define parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

print("Random Forest Parameter Grid:")
for param, values in rf_param_grid.items():
    print(f"  {param}: {values}")

# Perform Grid Search for Random Forest
rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=rf_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("\nStarting Random Forest Grid Search...")
rf_grid.fit(X_train_scaled, y_train)

print("✅ Random Forest Grid Search completed!")
print(f"Best parameters: {rf_grid.best_params_}")
print(f"Best cross-validation score: {rf_grid.best_score_:.4f}")

# 2. GRADIENT BOOSTING HYPERPARAMETER TUNING

# Define parameter grid for Gradient Boosting
gb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
}

print("Gradient Boosting Parameter Grid:")
for param, values in gb_param_grid.items():
    print(f"  {param}: {values}")

# Perform Randomized Search for Gradient Boosting (faster than GridSearch)
gb_random = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_distributions=gb_param_grid,
    n_iter=50,  # Number of parameter combinations to try
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("\nStarting Gradient Boosting Randomized Search...")
gb_random.fit(X_train_scaled, y_train)

print("✅ Gradient Boosting Randomized Search completed!")
print(f"Best parameters: {gb_random.best_params_}")
print(f"Best cross-validation score: {gb_random.best_score_:.4f}")

# 3. XGBOOST HYPERPARAMETER TUNING

# Define parameter grid for XGBoost
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.1, 1.2]
}

print("XGBoost Parameter Grid:")
for param, values in xgb_param_grid.items():
    print(f"  {param}: {values}")

# Perform Randomized Search for XGBoost
xgb_random = RandomizedSearchCV(
    estimator=XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    param_distributions=xgb_param_grid,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("\nStarting XGBoost Randomized Search...")
xgb_random.fit(X_train_scaled, y_train)

print("✅ XGBoost Randomized Search completed!")
print(f"Best parameters: {xgb_random.best_params_}")
print(f"Best cross-validation score: {xgb_random.best_score_:.4f}")

# 4. LOGISTIC REGRESSION HYPERPARAMETER TUNING

# Define parameter grid for Logistic Regression
lr_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [1000, 2000]
}

print("Logistic Regression Parameter Grid:")
for param, values in lr_param_grid.items():
    print(f"  {param}: {values}")

# Perform Grid Search for Logistic Regression
lr_grid = GridSearchCV(
    estimator=LogisticRegression(random_state=42),
    param_grid=lr_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("\nStarting Logistic Regression Grid Search...")
lr_grid.fit(X_train_scaled, y_train)

print("✅ Logistic Regression Grid Search completed!")
print(f"Best parameters: {lr_grid.best_params_}")
print(f"Best cross-validation score: {lr_grid.best_score_:.4f}")

# 5. SVM HYPERPARAMETER TUNING
print("\n" + "="*60)
print("5. SVM HYPERPARAMETER TUNING")
print("="*60)

# Define parameter grid for SVM
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 1],
    'degree': [2, 3, 4]  # Only for poly kernel
}

print("SVM Parameter Grid:")
for param, values in svm_param_grid.items():
    print(f"  {param}: {values}")

# Perform Randomized Search for SVM
svm_random = RandomizedSearchCV(
    estimator=SVC(random_state=42, probability=True),
    param_distributions=svm_param_grid,
    n_iter=30,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("\nStarting SVM Randomized Search...")
svm_random.fit(X_train_scaled, y_train)

print("✅ SVM Randomized Search completed!")
print(f"Best parameters: {svm_random.best_params_}")
print(f"Best cross-validation score: {svm_random.best_score_:.4f}")

# 6. COMPARE ALL TUNED MODELS
print("\n" + "="*60)
print("6. COMPARE TUNED MODELS PERFORMANCE")
print("="*60)

# Get the best estimators
best_rf = rf_grid.best_estimator_
best_gb = gb_random.best_estimator_
best_xgb = xgb_random.best_estimator_
best_lr = lr_grid.best_estimator_
best_svm = svm_random.best_estimator_

tuned_models = {
    'Random Forest': best_rf,
    'Gradient Boosting': best_gb,
    'XGBoost': best_xgb,
    'Logistic Regression': best_lr,
    'SVM': best_svm
}

# Evaluate on test set
results = []

for name, model in tuned_models.items():
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    results.append({
        'Model': name,
        'Test Accuracy': accuracy,
        'Test AUC': auc_score,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std()
    })

# Create results dataframe
results_df = pd.DataFrame(results).sort_values('Test Accuracy', ascending=False)
print("\nTuned Models Performance Comparison:")
print(results_df.round(4))

# 7. VISUALIZE TUNING RESULTS
# Plot comparison
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Test Accuracy comparison
models = results_df['Model']
test_acc = results_df['Test Accuracy']
ax1.barh(models, test_acc, color='lightblue')
ax1.set_xlabel('Accuracy')
ax1.set_title('Test Accuracy of Tuned Models')
ax1.set_xlim(0, 1)
for i, v in enumerate(test_acc):
    ax1.text(v + 0.01, i, f'{v:.4f}', va='center')

# Test AUC comparison
test_auc = results_df['Test AUC']
ax2.barh(models, test_auc, color='lightgreen')
ax2.set_xlabel('AUC Score')
ax2.set_title('Test AUC Score of Tuned Models')
ax2.set_xlim(0, 1)
for i, v in enumerate(test_auc):
    ax2.text(v + 0.01, i, f'{v:.4f}', va='center')

# CV Mean comparison
cv_mean = results_df['CV Mean']
cv_std = results_df['CV Std']
ax3.barh(models, cv_mean, xerr=cv_std, alpha=0.7, color='orange')
ax3.set_xlabel('Accuracy')
ax3.set_title('Cross-Validation Performance\n(Mean ± Std)')
ax3.set_xlim(0, 1)

# Performance improvement (you might need baseline models for comparison)
ax4.barh(models, test_acc, color='purple', alpha=0.7)
ax4.set_xlabel('Accuracy')
ax4.set_title('Final Model Performance')
ax4.set_xlim(0, 1)
for i, v in enumerate(test_acc):
    ax4.text(v + 0.01, i, f'{v:.4f}', va='center')

plt.tight_layout()
plt.show()

# 8. DETAILED ANALYSIS OF BEST MODEL

# Get the best model
best_model_name = results_df.iloc[0]['Model']
best_model = tuned_models[best_model_name]

print(f"🏆 Best Model: {best_model_name}")
print(f"📊 Test Accuracy: {results_df.iloc[0]['Test Accuracy']:.4f}")
print(f"🎯 Test AUC: {results_df.iloc[0]['Test AUC']:.4f}")

# Detailed evaluation of best model
y_pred_best = best_model.predict(X_test_scaled)
y_pred_proba_best = best_model.predict_proba(X_test_scaled)

print(f"\n📋 Classification Report for {best_model_name}:")
print(classification_report(y_test, y_pred_best))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low Performance', 'High Performance'],
            yticklabels=['Low Performance', 'High Performance'])
plt.title(f'Confusion Matrix - {best_model_name} (Tuned)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# 9. FEATURE IMPORTANCE FOR TREE-BASED MODELS

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("Feature Importance from Best Model:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title(f'Feature Importance - {best_model_name} (Tuned)')
    plt.tight_layout()
    plt.show()

# 10. SAVE THE TUNED MODELS

# Save all tuned models
for name, model in tuned_models.items():
    filename = f'tuned_{name.lower().replace(" ", "_")}_model.pkl'
    joblib.dump(model, filename)
    print(f"✅ {name} saved as: {filename}")

# Save the best model separately
best_model_filename = 'best_tuned_model.pkl'
joblib.dump(best_model, best_model_filename)
print(f"✅ Best model saved as: {best_model_filename}")

# Save the scaler
joblib.dump(scaler, 'tuned_scaler.pkl')
print("✅ Scaler saved")

# Save tuning results
tuning_results = {
    'best_model': best_model_name,
    'best_parameters': str(best_model.get_params()),
    'performance': results_df.to_dict()
}

import json
with open('hyperparameter_tuning_results.json', 'w') as f:
    json.dump(tuning_results, f, indent=2)
print("✅ Tuning results saved to 'hyperparameter_tuning_results.json'")

# 11. COMPARE WITH DEFAULT MODELS
# Create default models
default_models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'Logistic Regression': LogisticRegression(random_state=42),
    'SVM': SVC(random_state=42, probability=True)
}

comparison_results = []

for name in tuned_models.keys():
    # Default model
    default_model = default_models[name]
    default_model.fit(X_train_scaled, y_train)
    default_accuracy = accuracy_score(y_test, default_model.predict(X_test_scaled))
    
    # Tuned model
    tuned_accuracy = results_df[results_df['Model'] == name]['Test Accuracy'].values[0]
    
    improvement = tuned_accuracy - default_accuracy
    
    comparison_results.append({
        'Model': name,
        'Default Accuracy': default_accuracy,
        'Tuned Accuracy': tuned_accuracy,
        'Improvement': improvement
    })

comparison_df = pd.DataFrame(comparison_results).sort_values('Improvement', ascending=False)
print("\nTuned vs Default Models Comparison:")
print(comparison_df.round(4))

# Plot improvement
plt.figure(figsize=(12, 6))
x_pos = np.arange(len(comparison_df))
width = 0.35

plt.bar(x_pos - width/2, comparison_df['Default Accuracy'], width, label='Default', alpha=0.7)
plt.bar(x_pos + width/2, comparison_df['Tuned Accuracy'], width, label='Tuned', alpha=0.7)

plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Hyperparameter Tuning: Default vs Tuned Models')
plt.xticks(x_pos, comparison_df['Model'], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Add improvement values
for i, row in comparison_df.iterrows():
    plt.text(i, max(row['Default Accuracy'], row['Tuned Accuracy']) + 0.01, 
             f"+{row['Improvement']:.4f}", ha='center')

plt.tight_layout()
plt.show()

# 12. FINAL SUMMARY
print("\n" + "="*60)
print("HYPERPARAMETER TUNING SUMMARY")
print("="*60)

best_row = comparison_df.iloc[0]
print(f"🏆 Best Model: {best_row['Model']}")
print(f"📈 Improvement from tuning: +{best_row['Improvement']:.4f}")
print(f"🎯 Final Accuracy: {best_row['Tuned Accuracy']:.4f}")
print(f"📊 Best parameters saved in: 'hyperparameter_tuning_results.json'")
print(f"💾 All tuned models saved as .pkl files")

print(f"\n✅ Hyperparameter tuning completed successfully!")
print(f"🔧 Total models tuned: {len(tuned_models)}")
print(f"📁 Models saved in current directory")