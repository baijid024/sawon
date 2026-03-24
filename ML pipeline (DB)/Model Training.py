import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Load the dataset
df = pd.read_csv('batting_summary.csv')

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# Data Preprocessing Function
def preprocess_data(df):
    """Preprocess the cricket batting data"""
    df_clean = df.copy()
    
    # Handle missing values in SR column
    df_clean['SR'] = pd.to_numeric(df_clean['SR'], errors='coerce')
    df_clean['SR'].fillna(df_clean['SR'].mean(), inplace=True)
    
    # Create target variable (high performance indicator)
    df_clean['high_performance'] = (df_clean['runs'] > 30).astype(int)
    
    # Encode categorical variables
    le = LabelEncoder()
    df_clean['team_encoded'] = le.fit_transform(df_clean['teamInnings'])
    df_clean['out_status'] = le.fit_transform(df_clean['Out/Not_Out'])
    
    return df_clean

# Preprocess the data
df_processed = preprocess_data(df)
print(f"\nAfter preprocessing - Shape: {df_processed.shape}")

# Check target distribution
target_distribution = df_processed['high_performance'].value_counts()
print(f"\nTarget Distribution:\n{target_distribution}")
print(f"Percentage of high performers: {target_distribution[1]/len(df_processed)*100:.2f}%")

# Select features and target
features = ['battingPos', 'runs', 'balls', '4s', '6s', 'SR', 'team_encoded', 'out_status']
X = df_processed[features]
y = df_processed['high_performance']

print(f"\nFeatures used: {features}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Maintain the same distribution in both sets
)

print(f"\nData Split:")
print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed!")

# 1. Define Models to Train
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'SVM': SVC(random_state=42, probability=True)
}

# 2. Train All Models


trained_models = {}
training_results = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Store results
    trained_models[model_name] = model
    training_results[model_name] = {
        'accuracy': accuracy,
        'auc_score': auc_score,
        'model': model,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"✅ {model_name} trained successfully!")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   AUC Score: {auc_score:.4f}")

# 3. Compare Model Performance
# Create performance comparison dataframe
performance_df = pd.DataFrame({
    'Model': list(training_results.keys()),
    'Accuracy': [results['accuracy'] for results in training_results.values()],
    'AUC_Score': [results['auc_score'] for results in training_results.values()]
}).sort_values('Accuracy', ascending=False)

print(performance_df)

# 4. Visualize Model Performance
plt.figure(figsize=(12, 6))

# Accuracy comparison
plt.subplot(1, 2, 1)
plt.barh(performance_df['Model'], performance_df['Accuracy'], color='skyblue')
plt.xlabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.xlim(0, 1)
for i, v in enumerate(performance_df['Accuracy']):
    plt.text(v + 0.01, i, f'{v:.4f}', va='center')

# AUC comparison
plt.subplot(1, 2, 2)
plt.barh(performance_df['Model'], performance_df['AUC_Score'], color='lightgreen')
plt.xlabel('AUC Score')
plt.title('Model AUC Score Comparison')
plt.xlim(0, 1)
for i, v in enumerate(performance_df['AUC_Score']):
    plt.text(v + 0.01, i, f'{v:.4f}', va='center')

plt.tight_layout()
plt.show()

# 5. Select the Best Model
best_model_name = performance_df.iloc[0]['Model']
best_model = trained_models[best_model_name]
best_results = training_results[best_model_name]

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Accuracy: {best_results['accuracy']:.4f}")
print(f"   AUC Score: {best_results['auc_score']:.4f}")

# 6. Detailed Evaluation of Best Model
# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, best_results['predictions']))

# Confusion Matrix
cm = confusion_matrix(y_test, best_results['predictions'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low Performance', 'High Performance'],
            yticklabels=['Low Performance', 'High Performance'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, best_results['probabilities'])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {best_results["auc_score"]:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve - {best_model_name}')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# 7. Feature Importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    print("\nFeature Importance:")
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title(f'Feature Importance - {best_model_name}')
    plt.tight_layout()
    plt.show()

# 8. Save the Best Model and Scaler
# Save the best model
best_model_filename = f'best_cricket_model_{best_model_name.replace(" ", "_").lower()}.pkl'
joblib.dump(best_model, best_model_filename)
print(f"✅ Best model saved as: {best_model_filename}")

# Save all models
for model_name, model in trained_models.items():
    filename = f'cricket_model_{model_name.replace(" ", "_").lower()}.pkl'
    joblib.dump(model, filename)
    print(f"✅ {model_name} saved as: {filename}")

# Save the scaler
scaler_filename = 'feature_scaler.pkl'
joblib.dump(scaler, scaler_filename)
print(f"✅ Scaler saved as: {scaler_filename}")

# Save feature names
feature_info = {
    'features': features,
    'feature_indices': {feature: idx for idx, feature in enumerate(features)}
}
joblib.dump(feature_info, 'feature_info.pkl')
print("✅ Feature information saved")

# 9. Create Prediction Function
def predict_batting_performance(batting_data):
    """
    Predict batting performance using the trained model
    
    Parameters:
    batting_data: dict or list with features in order:
        [battingPos, runs, balls, 4s, 6s, SR, team_encoded, out_status]
    """
    try:
        # Load model and scaler
        model = joblib.load(best_model_filename)
        scaler = joblib.load(scaler_filename)
        
        # Convert input to numpy array
        if isinstance(batting_data, dict):
            # If input is dictionary, extract values in correct order
            feature_values = [batting_data[feature] for feature in features]
        else:
            # If input is list, use directly
            feature_values = batting_data
            
        input_array = np.array([feature_values])
        
        # Scale the features
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        return {
            'prediction': 'High Performer' if prediction == 1 else 'Low Performer',
            'probability_high_performance': probability[1],
            'probability_low_performance': probability[0],
            'confidence': max(probability)
        }
        
    except Exception as e:
        return f"Error in prediction: {str(e)}"

# 10. Test the Prediction Function


# Test cases
test_cases = [
    [3, 45, 30, 5, 2, 150.0, 1, 1],  # High performer
    [1, 15, 20, 2, 0, 75.0, 0, 0],   # Low performer
    [5, 60, 40, 8, 3, 150.0, 1, 1]   # High performer
]

for i, test_case in enumerate(test_cases, 1):
    result = predict_batting_performance(test_case)
    print(f"\nTest Case {i}: {test_case}")
    print(f"Prediction: {result['prediction']}")
    print(f"Probability of High Performance: {result['probability_high_performance']:.4f}")
    print(f"Confidence: {result['confidence']:.4f}")

# 11. Training Summary

print(f"✅ Dataset: {df.shape[0]} records, {df.shape[1]} features")
print(f"✅ Target: High Performance (>30 runs)")
print(f"✅ Features used: {len(features)}")
print(f"✅ Best Model: {best_model_name}")
print(f"✅ Best Accuracy: {best_results['accuracy']:.4f}")
print(f"✅ Models saved: {len(trained_models)}")
print(f"✅ Ready for predictions!")

# 12. Additional: Training History (for models that support it)
if hasattr(best_model, 'estimators_'):
    print(f"\nTraining completed with {len(best_model.estimators_)} estimators")

print("\n🎯 Model training completed successfully!")