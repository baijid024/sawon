import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, learning_curve
import scikitplot as skplt
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the trained model and data
print("Loading trained model and data...")

# Load the model (assuming you've saved it previously)
model = joblib.load('best_cricket_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

# Load the dataset
df = pd.read_csv('batting_summary.csv')

# Preprocessing function (same as training)
def preprocess_data(df):
    df_clean = df.copy()
    df_clean['SR'] = pd.to_numeric(df_clean['SR'], errors='coerce')
    df_clean['SR'].fillna(df_clean['SR'].mean(), inplace=True)
    df_clean['high_performance'] = (df_clean['runs'] > 30).astype(int)
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df_clean['team_encoded'] = le.fit_transform(df_clean['teamInnings'])
    df_clean['out_status'] = le.fit_transform(df_clean['Out/Not_Out'])
    
    return df_clean

# Preprocess data
df_processed = preprocess_data(df)

# Features and target
features = ['battingPos', 'runs', 'balls', '4s', '6s', 'SR', 'team_encoded', 'out_status']
X = df_processed[features]
y = df_processed['high_performance']

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
X_test_scaled = scaler.transform(X_test)

print("✅ Data loaded and prepared for evaluation")
print(f"Test set size: {X_test.shape[0]} samples")

# 1. BASIC PREDICTIONS AND METRICS

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

# Calculate basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba[:, 1])

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc_roc:.4f}")

# 2. CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Detailed confusion matrix analysis
tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")

# Calculate additional metrics
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

print(f"\nSpecificity: {specificity:.4f}")
print(f"False Positive Rate: {false_positive_rate:.4f}")
print(f"False Negative Rate: {false_negative_rate:.4f}")

# Plot confusion matrix
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low Perf', 'High Perf'],
            yticklabels=['Low Perf', 'High Perf'])
plt.title('Confusion Matrix\n(Counts)')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# Normalized confusion matrix
plt.subplot(1, 2, 2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=['Low Perf', 'High Perf'],
            yticklabels=['Low Perf', 'High Perf'])
plt.title('Confusion Matrix\n(Percentages)')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.tight_layout()
plt.show()

# 3. CLASSIFICATION REPORT

print(classification_report(y_test, y_pred, 
                          target_names=['Low Performance', 'High Performance']))

# 4. ROC CURVE AND AUC
# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])

# Find optimal threshold (Youden's J statistic)
youden_j = tpr - fpr
optimal_idx = np.argmax(youden_j)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal Threshold: {optimal_threshold:.4f}")
print(f"At this threshold:")
print(f" - True Positive Rate: {tpr[optimal_idx]:.4f}")
print(f" - False Positive Rate: {fpr[optimal_idx]:.4f}")

# Plot ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {auc_roc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='Random Classifier')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', 
            s=100, label=f'Optimal Threshold ({optimal_threshold:.3f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# 5. PRECISION-RECALL CURVE

precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])
average_precision = average_precision_score(y_test, y_pred_proba[:, 1])

print(f"Average Precision Score: {average_precision:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(recall_vals, precision_vals, color='blue', lw=2, 
         label=f'Precision-Recall curve (AP = {average_precision:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.show()

# 6. PROBABILITY DISTRIBUTION

plt.figure(figsize=(12, 6))

# Probability distribution by actual class
plt.subplot(1, 2, 1)
plt.hist(y_pred_proba[y_test == 0, 1], bins=20, alpha=0.7, 
         label='Actual Low Performers', color='red')
plt.hist(y_pred_proba[y_test == 1, 1], bins=20, alpha=0.7, 
         label='Actual High Performers', color='green')
plt.xlabel('Predicted Probability of High Performance')
plt.ylabel('Frequency')
plt.title('Probability Distribution by Actual Class')
plt.legend()

# Cumulative distribution
plt.subplot(1, 2, 2)
for actual_class in [0, 1]:
    probabilities = y_pred_proba[y_test == actual_class, 1]
    sorted_probs = np.sort(probabilities)
    cumulative = np.arange(len(sorted_probs)) / len(sorted_probs)
    label = 'Low Performers' if actual_class == 0 else 'High Performers'
    plt.plot(sorted_probs, cumulative, label=label, lw=2)
    
plt.xlabel('Predicted Probability of High Performance')
plt.ylabel('Cumulative Proportion')
plt.title('Cumulative Probability Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7. FEATURE IMPORTANCE ANALYSIS

if hasattr(model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("Feature Importance Scores:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

# 8. CROSS-VALIDATION SCORES

# Perform cross-validation
cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='accuracy')
cv_scores_precision = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='precision')
cv_scores_recall = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='recall')
cv_scores_f1 = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='f1')

print(f"Cross-Validation Accuracy:  {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
print(f"Cross-Validation Precision: {cv_scores_precision.mean():.4f} (±{cv_scores_precision.std():.4f})")
print(f"Cross-Validation Recall:    {cv_scores_recall.mean():.4f} (±{cv_scores_recall.std():.4f})")
print(f"Cross-Validation F1-Score:  {cv_scores_f1.mean():.4f} (±{cv_scores_f1.std():.4f})")

# 9. LEARNING CURVE
train_sizes, train_scores, test_scores = learning_curve(
    model, scaler.transform(X), y, cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy', n_jobs=-1
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

# 10. THRESHOLD ANALYSIS
thresholds = np.arange(0.1, 1.0, 0.1)
threshold_results = []

for threshold in thresholds:
    y_pred_thresh = (y_pred_proba[:, 1] >= threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred_thresh)
    precision = precision_score(y_test, y_pred_thresh, zero_division=0)
    recall = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
    
    threshold_results.append({
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })

threshold_df = pd.DataFrame(threshold_results)
print("Performance at Different Thresholds:")
print(threshold_df.round(4))

# Plot threshold analysis
plt.figure(figsize=(12, 8))
plt.plot(threshold_df['threshold'], threshold_df['accuracy'], label='Accuracy', marker='o')
plt.plot(threshold_df['threshold'], threshold_df['precision'], label='Precision', marker='s')
plt.plot(threshold_df['threshold'], threshold_df['recall'], label='Recall', marker='^')
plt.plot(threshold_df['threshold'], threshold_df['f1_score'], label='F1-Score', marker='d')
plt.xlabel('Classification Threshold')
plt.ylabel('Score')
plt.title('Model Performance vs Classification Threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 11. ERROR ANALYSIS
# Create results dataframe
results_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred,
    'probability_high': y_pred_proba[:, 1]
})

# Add features for analysis
for i, feature in enumerate(features):
    results_df[feature] = X_test.iloc[:, i].values

# Identify errors
results_df['correct'] = results_df['actual'] == results_df['predicted']
results_df['error_type'] = 'Correct'
results_df.loc[(results_df['actual'] == 0) & (results_df['predicted'] == 1), 'error_type'] = 'False Positive'
results_df.loc[(results_df['actual'] == 1) & (results_df['predicted'] == 0), 'error_type'] = 'False Negative'

print("Error Distribution:")
print(results_df['error_type'].value_counts())

# Analyze feature patterns in errors
print("\nAverage Feature Values by Prediction Type:")
error_analysis = results_df.groupby('error_type')[features].mean()
print(error_analysis.round(2))


print(f"📊 Overall Accuracy: {accuracy:.4f}")
print(f"🎯 AUC-ROC Score: {auc_roc:.4f}")
print(f"⚖️  F1-Score: {f1:.4f}")
print(f"📈 Average Precision: {average_precision:.4f}")

print(f"\n🔍 Confusion Matrix:")
print(f"   True Positives: {tp}")
print(f"   True Negatives: {tn}")
print(f"   False Positives: {fp}")
print(f"   False Negatives: {fn}")

print(f"\n📋 Classification Report:")
print(f"   Precision: {precision:.4f}")
print(f"   Recall: {recall:.4f}")
print(f"   Specificity: {specificity:.4f}")

print(f"\n🔄 Cross-Validation:")
print(f"   Mean CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

print(f"\n🎯 Recommended Threshold: {optimal_threshold:.4f}")

print("\n" + "="*60)
print("✅ MODEL EVALUATION COMPLETED SUCCESSFULLY!")
print("="*60)

# Save evaluation results
evaluation_results = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'auc_roc': auc_roc,
    'average_precision': average_precision,
    'confusion_matrix': cm,
    'optimal_threshold': optimal_threshold
}

import json
with open('model_evaluation_results.json', 'w') as f:
    # Convert numpy arrays to lists for JSON serialization
    json_ready = {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                 for k, v in evaluation_results.items()}
    json.dump(json_ready, f, indent=2)

print("📁 Evaluation results saved to 'model_evaluation_results.json'")






# 2. CONFUSION MATRIX(Another Method)
ConfusionMatrixDisplay:
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

Alternative Syntax:
Confusion Matrix আগে বানিয়ে তারপর display করা যায়:
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'High'])
disp.plot(cmap='Blues')
plt.show()

## 1) ROC (Receiver Operating Characteristic) Curve
ROC curve হলো একটা গ্রাফ, যেখানে আমরা model-এর **performance** দেখি **সব threshold-এর জন্য**।
* **X-axis:** False Positive Rate (FPR)
  * FPR = কতগুলো negative case ভুল করে positive বলে ধরা হলো
* **Y-axis:** True Positive Rate (TPR) (অর্থাৎ Recall/Sensitivity)
 * TPR = কতগুলো আসল positive case সঠিকভাবে positive হিসেবে ধরা হলো
👉 মানে: ROC curve দেখায়, cutoff (threshold) পরিবর্তন করলে **model-এর sensitivity (Recall) বনাম false alarms (FPR)** কেমন পরিবর্তিত হয়।
## 2) AUC (Area Under Curve)
AUC = ROC curve-এর নিচের এলাকা।
* মান সর্বোচ্চ 1.0 হতে পারে।
* **AUC = 0.5** → model একেবারে random guess (কোনো লাভ নাই)।
* **AUC = 1.0** → perfect classifier।
* যত বেশি AUC, তত বেশি model positive vs negative আলাদা করতে পারছে।
## 3) Threshold (Cutoff)
Model probability দেয় (যেমন `0.2`, `0.6`, `0.85`)।
কিন্তু তোমাকে শেষমেশ **0 অথবা 1** বলতে হবে।
সেটা করার জন্য একটা threshold ঠিক করতে হয়।

* যদি threshold = **0.5** (default):

  * probability ≥ 0.5 → predict = 1
  * probability < 0.5 → predict = 0
* Threshold কমালে → Recall বাড়ে, কিন্তু False Positive-ও বাড়ে।
* Threshold বাড়ালে → Precision বাড়তে পারে, কিন্তু Recall কমে যায়।
### উদাহরণ:
ধরি model output দিলো:
```
Sample1 → 0.8
Sample2 → 0.6
Sample3 → 0.4
Sample4 → 0.2
```* যদি threshold=0.5 → predict: [1,1,0,0]
* যদি threshold=0.3 → predict: [1,1,1,0] (Recall বেশি, কিন্তু ভুল বেশি হতে পারে)
* যদি threshold=0.7 → predict: [1,0,0,0] (Precision বেশি, কিন্তু অনেক positive miss হবে)
## সহজভাবে
* **ROC curve** = performance at all thresholds
* **AUC** = সার্বিকভাবে model কত ভালো positive vs negative আলাদা করতে পারে
* **Threshold** = সেই cutoff probability, যেখানে তুমি বলবে "এখন থেকে এটাকে 1 ধরবো"

ROC Curve → ভালো যখন dataset balanced
Precision–Recall Curve → ভালো যখন dataset imbalanced (rare events)
