import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve, validation_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.utils import resample
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🚀 STARTING COMPREHENSIVE MODEL VALIDATION")
print("="*60)

# Load the dataset
df = pd.read_csv('batting_summary.csv')

# Preprocessing function
def preprocess_data(df):
    df_clean = df.copy()
    df_clean['SR'] = pd.to_numeric(df_clean['SR'], errors='coerce')
    df_clean['SR'].fillna(df_clean['SR'].mean(), inplace=True)
    df_clean['high_performance'] = (df_clean['runs'] > 30).astype(int)
    
    le = LabelEncoder()
    df_clean['team_encoded'] = le.fit_transform(df_clean['teamInnings'])
    df_clean['out_status'] = le.fit_transform(df_clean['Out/Not_Out'])
    
    return df_clean

# Preprocess data
df_processed = preprocess_data(df)
print("✅ Data preprocessing completed")

# Features and target
features = ['battingPos', 'runs', 'balls', '4s', '6s', 'SR', 'team_encoded', 'out_status']
X = df_processed[features]
y = df_processed['high_performance']

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Target distribution: {y.value_counts().to_dict()}")

# Load the trained model
try:
    model = joblib.load('best_tuned_model.pkl')
    print("✅ Loaded tuned model")
except:
    try:
        model = joblib.load('best_cricket_model.pkl')
        print("✅ Loaded best model")
    except:
        print("❌ No saved model found. Training a new one...")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)

# Load or create scaler
try:
    scaler = joblib.load('tuned_scaler.pkl')
    print("✅ Loaded saved scaler")
except:
    scaler = StandardScaler()
    print("✅ Created new scaler")

# 1. BASIC DATA SPLIT VALIDATION
print("\n" + "="*60)
print("1. BASIC TRAIN-TEST SPLIT VALIDATION")
print("="*60)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model if not already trained
if not hasattr(model, 'feature_importances_'):
    model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

print(f"📊 Basic Split Validation Results:")
print(f"   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print(f"   AUC-ROC:   {auc_roc:.4f}")

# 2. CROSS-VALIDATION
print("\n" + "="*60)
print("2. CROSS-VALIDATION")
print("="*60)

# Perform k-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores_accuracy = cross_val_score(model, scaler.transform(X), y, cv=kfold, scoring='accuracy')
cv_scores_precision = cross_val_score(model, scaler.transform(X), y, cv=kfold, scoring='precision')
cv_scores_recall = cross_val_score(model, scaler.transform(X), y, cv=kfold, scoring='recall')
cv_scores_f1 = cross_val_score(model, scaler.transform(X), y, cv=kfold, scoring='f1')
cv_scores_auc = cross_val_score(model, scaler.transform(X), y, cv=kfold, scoring='roc_auc')

print("📈 Cross-Validation Results (5-fold):")
print(f"   Accuracy:  {cv_scores_accuracy.mean():.4f} (±{cv_scores_accuracy.std():.4f})")
print(f"   Precision: {cv_scores_precision.mean():.4f} (±{cv_scores_precision.std():.4f})")
print(f"   Recall:    {cv_scores_recall.mean():.4f} (±{cv_scores_recall.std():.4f})")
print(f"   F1-Score:  {cv_scores_f1.mean():.4f} (±{cv_scores_f1.std():.4f})")
print(f"   AUC-ROC:   {cv_scores_auc.mean():.4f} (±{cv_scores_auc.std():.4f})")

# Plot CV results
plt.figure(figsize=(12, 6))
cv_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
cv_means = [cv_scores_accuracy.mean(), cv_scores_precision.mean(), 
           cv_scores_recall.mean(), cv_scores_f1.mean(), cv_scores_auc.mean()]
cv_stds = [cv_scores_accuracy.std(), cv_scores_precision.std(), 
          cv_scores_recall.std(), cv_scores_f1.std(), cv_scores_auc.std()]

plt.bar(cv_metrics, cv_means, yerr=cv_stds, capsize=5, alpha=0.7, color='skyblue')
plt.ylabel('Score')
plt.title('Cross-Validation Performance (5-fold)')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)

for i, v in enumerate(cv_means):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# 3. LEARNING CURVE ANALYSIS
print("\n" + "="*60)
print("3. LEARNING CURVE ANALYSIS")
print("="*60)

train_sizes, train_scores, test_scores = learning_curve(
    model, scaler.transform(X), y, cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy', n_jobs=-1, random_state=42
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='red', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='green', label='Cross-validation score')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='red')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='green')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.show()

print("📚 Learning Curve Analysis:")
print(f"   Final training score: {train_scores_mean[-1]:.4f}")
print(f"   Final validation score: {test_scores_mean[-1]:.4f}")
print(f"   Gap: {train_scores_mean[-1] - test_scores_mean[-1]:.4f}")

# 4. VALIDATION CURVE
print("\n" + "="*60)
print("4. VALIDATION CURVE (if applicable)")
print("="*60)

# Example for Random Forest - n_estimators
if hasattr(model, 'n_estimators'):
    param_range = [50, 100, 150, 200, 250]
    train_scores, test_scores = validation_curve(
        model, scaler.transform(X), y, 
        param_name="n_estimators", param_range=param_range,
        cv=5, scoring="accuracy", n_jobs=-1
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_scores_mean, 'o-', color='red', label='Training score')
    plt.plot(param_range, test_scores_mean, 'o-', color='green', label='Cross-validation score')
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='red')
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='green')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy Score')
    plt.title('Validation Curve - n_estimators')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.show()

# 5. BOOTSTRAP VALIDATION
print("\n" + "="*60)
print("5. BOOTSTRAP VALIDATION")
print("="*60)

n_bootstraps = 1000
bootstrap_scores = []

for i in range(n_bootstraps):
    # Create bootstrap sample
    X_bs, y_bs = resample(X_test_scaled, y_test, random_state=i)
    
    # Make predictions
    y_pred_bs = model.predict(X_bs)
    
    # Calculate accuracy
    accuracy_bs = accuracy_score(y_bs, y_pred_bs)
    bootstrap_scores.append(accuracy_bs)

bootstrap_scores = np.array(bootstrap_scores)

# Calculate confidence intervals
confidence_interval = np.percentile(bootstrap_scores, [2.5, 97.5])
bootstrap_mean = np.mean(bootstrap_scores)
bootstrap_std = np.std(bootstrap_scores)

print(f"🔄 Bootstrap Validation Results ({n_bootstraps} iterations):")
print(f"   Mean Accuracy: {bootstrap_mean:.4f}")
print(f"   Standard Deviation: {bootstrap_std:.4f}")
print(f"   95% Confidence Interval: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")

# Plot bootstrap distribution
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_scores, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
plt.axvline(bootstrap_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {bootstrap_mean:.4f}')
plt.axvline(confidence_interval[0], color='orange', linestyle='--', linewidth=2, label='95% CI')
plt.axvline(confidence_interval[1], color='orange', linestyle='--', linewidth=2)
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.title('Bootstrap Accuracy Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 6. TEMPORAL VALIDATION (if matchID represents time)
print("\n" + "="*60)
print("6. TEMPORAL VALIDATION")
print("="*60)

# Sort by matchID (assuming it represents temporal order)
try:
    df_sorted = df_processed.sort_values('matchID')
    X_sorted = df_sorted[features]
    y_sorted = df_sorted['high_performance']
    
    # Time-based split: 80% train, 20% test
    split_point = int(0.8 * len(X_sorted))
    
    X_train_time = X_sorted.iloc[:split_point]
    X_test_time = X_sorted.iloc[split_point:]
    y_train_time = y_sorted.iloc[:split_point]
    y_test_time = y_sorted.iloc[split_point:]
    
    # Scale and train
    X_train_time_scaled = scaler.fit_transform(X_train_time)
    X_test_time_scaled = scaler.transform(X_test_time)
    
    temporal_model = type(model)(**model.get_params())
    temporal_model.fit(X_train_time_scaled, y_train_time)
    
    y_pred_time = temporal_model.predict(X_test_time_scaled)
    temporal_accuracy = accuracy_score(y_test_time, y_pred_time)
    
    print(f"⏰ Temporal Validation Results:")
    print(f"   Training period: First {split_point} matches")
    print(f"   Testing period: Last {len(X_test_time)} matches")
    print(f"   Temporal Accuracy: {temporal_accuracy:.4f}")
    
except Exception as e:
    print(f"⚠️  Temporal validation skipped: {e}")

# 7. FEATURE ABLATION STUDY
print("\n" + "="*60)
print("7. FEATURE ABLATION STUDY")
print("="*60)

# Test model performance by removing one feature at a time
feature_performance = []

for i, feature_to_remove in enumerate(features):
    # Create feature set without one feature
    reduced_features = [f for j, f in enumerate(features) if j != i]
    X_reduced = X[reduced_features]
    
    # Split and scale
    X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler_red = StandardScaler()
    X_train_red_scaled = scaler_red.fit_transform(X_train_red)
    X_test_red_scaled = scaler_red.transform(X_test_red)
    
    # Train and evaluate
    reduced_model = type(model)(**model.get_params())
    reduced_model.fit(X_train_red_scaled, y_train_red)
    y_pred_red = reduced_model.predict(X_test_red_scaled)
    accuracy_red = accuracy_score(y_test_red, y_pred_red)
    
    feature_performance.append({
        'Removed_Feature': feature_to_remove,
        'Accuracy_Without_Feature': accuracy_red,
        'Performance_Drop': accuracy - accuracy_red
    })

feature_ablation_df = pd.DataFrame(feature_performance).sort_values('Performance_Drop', ascending=False)

print("🔍 Feature Ablation Results:")
print(feature_ablation_df.round(4))

# Plot feature importance through ablation
plt.figure(figsize=(10, 6))
sns.barplot(x='Performance_Drop', y='Removed_Feature', data=feature_ablation_df)
plt.xlabel('Performance Drop When Removed')
plt.title('Feature Importance via Ablation Study')
plt.tight_layout()
plt.show()

# 8. STATISTICAL SIGNIFICANCE TESTING
print("\n" + "="*60)
print("8. STATISTICAL SIGNIFICANCE TESTING")
print("="*60)

# Compare with random classifier
random_predictions = np.random.randint(0, 2, size=len(y_test))
random_accuracy = accuracy_score(y_test, random_predictions)

# Perform statistical test (McNemar's test for paired data)
from statsmodels.stats.contingency_tables import mcnemar

# Create contingency table
contingency_table = np.zeros((2, 2))
for i in range(len(y_test)):
    contingency_table[y_test.iloc[i], y_pred[i]] += 1

# McNemar's test
try:
    mcnemar_result = mcnemar(contingency_table, exact=True)
    print(f"📊 Statistical Significance Test:")
    print(f"   Model Accuracy: {accuracy:.4f}")
    print(f"   Random Classifier Accuracy: {random_accuracy:.4f}")
    print(f"   McNemar's test p-value: {mcnemar_result.pvalue:.6f}")
    print(f"   Statistically significant: {mcnemar_result.pvalue < 0.05}")
except:
    print("⚠️  McNemar's test could not be performed")

# 9. CALIBRATION CHECK
print("\n" + "="*60)
print("9. MODEL CALIBRATION CHECK")
print("="*60)

# Check if predicted probabilities are calibrated
from sklearn.calibration import calibration_curve

fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, y_pred_proba, n_bins=10, strategy='quantile'
)

plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
plt.xlabel("Mean Predicted Value")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Plot")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Calculate Brier score
from sklearn.metrics import brier_score_loss
brier_score = brier_score_loss(y_test, y_pred_proba)

print(f"🎯 Calibration Analysis:")
print(f"   Brier Score: {brier_score:.4f} (lower is better)")
print(f"   Perfect calibration would be 0")

# 10. ROBUSTNESS TO NOISE
print("\n" + "="*60)
print("10. ROBUSTNESS TO NOISE")
print("="*60)

# Test model performance with added noise
noise_levels = [0.01, 0.05, 0.1, 0.2]
noise_performance = []

for noise_level in noise_levels:
    # Add Gaussian noise to test features
    X_test_noisy = X_test_scaled + np.random.normal(0, noise_level, X_test_scaled.shape)
    
    # Make predictions
    y_pred_noisy = model.predict(X_test_noisy)
    accuracy_noisy = accuracy_score(y_test, y_pred_noisy)
    
    noise_performance.append({
        'Noise_Level': noise_level,
        'Accuracy': accuracy_noisy,
        'Performance_Drop': accuracy - accuracy_noisy
    })

noise_df = pd.DataFrame(noise_performance)

print("🔊 Noise Robustness Results:")
print(noise_df.round(4))

plt.figure(figsize=(10, 6))
plt.plot(noise_df['Noise_Level'], noise_df['Accuracy'], 'o-', linewidth=2)
plt.axhline(y=accuracy, color='red', linestyle='--', label='Original Accuracy')
plt.xlabel('Noise Level (Standard Deviation)')
plt.ylabel('Accuracy')
plt.title('Model Robustness to Feature Noise')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 11. COMPREHENSIVE VALIDATION REPORT
print("\n" + "="*60)
print("COMPREHENSIVE VALIDATION REPORT")
print("="*60)

validation_report = {
    'basic_split_accuracy': accuracy,
    'cross_validation_mean_accuracy': cv_scores_accuracy.mean(),
    'cross_validation_std': cv_scores_accuracy.std(),
    'bootstrap_mean_accuracy': bootstrap_mean,
    'bootstrap_confidence_interval': confidence_interval.tolist(),
    'brier_score': brier_score,
    'feature_ablation_summary': feature_ablation_df.to_dict(),
    'noise_robustness_summary': noise_df.to_dict()
}

print("📋 Validation Summary:")
print(f"   ✅ Basic Split Accuracy: {accuracy:.4f}")
print(f"   ✅ Cross-Validation Accuracy: {cv_scores_accuracy.mean():.4f} (±{cv_scores_accuracy.std():.4f})")
print(f"   ✅ Bootstrap 95% CI: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")
print(f"   ✅ Model Calibration (Brier Score): {brier_score:.4f}")
print(f"   ✅ Most Important Feature: {feature_ablation_df.iloc[0]['Removed_Feature']}")
print(f"   ✅ Noise Robustness: Maintains {noise_df.iloc[-1]['Accuracy']:.4f} accuracy at 20% noise")

# Save validation results
import json
with open('model_validation_report.json', 'w') as f:
    json.dump(validation_report, f, indent=2)

print("\n" + "="*60)
print("✅ MODEL VALIDATION COMPLETED SUCCESSFULLY!")
print("="*60)
print("📁 Validation report saved as 'model_validation_report.json'")
print("🎯 Key findings:")
print(f"   - Model shows {'GOOD' if cv_scores_accuracy.mean() > 0.7 else 'MODERATE'} generalization")
print(f"   - Cross-validation stability: {'HIGH' if cv_scores_accuracy.std() < 0.05 else 'MODERATE'}")
print(f"   - Calibration: {'GOOD' if brier_score < 0.1 else 'NEEDS IMPROVEMENT'}")
print(f"   - Robustness: {'HIGH' if noise_df.iloc[-1]['Performance_Drop'] < 0.1 else 'MODERATE'}")