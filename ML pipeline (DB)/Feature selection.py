import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import (SelectKBest, f_classif, mutual_info_classif, 
                                     RFE, SelectFromModel, VarianceThreshold)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import scipy.stats as stats
from scipy.stats import pearsonr, f_oneway
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the data
df = pd.read_csv('batting_summary.csv')

print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Data Preprocessing
def preprocess_data(df)
    # Create a copy
    data=df.copy()
    
    # Handle missing values in SR column
    data['SR']=pd.to_numeric(data['SR'],errors='coerce')
    data['SR'].fillna(data['SR'].mean(),inplace=True)
    # Create target variable (high performance indicator)
    data['high_performance']=(data['runs']>30).astype(int)
    
    # Encode categorical variables
    le=LabelEncoder()
    data['team_encoded']=le.fit_transform(data['teamInnings'])
    data['out_status']=le.fit_transform(data['Out/Not_Out'])
    
    # Extract match type or tournament info if needed
    data['is_t20i']=data['matchID'].str.contains('T20I').astype(int)
    return data

# Preprocess the data
df_processed = preprocess_data(df)

# Define features for selection
features = ['battingPos', 'runs', 'balls', '4s', '6s', 'SR', 'team_encoded', 'out_status', 'is_t20i']
X=df_processed[features]
Y=df_processed['high_performance']
# Handle any remaining missing values
x=X.fillna(X.mean())

print(f'Features shape:{X.shape}')
print(f'Target distribution:\n {Y.value_counts()}')

# 1. Correlation Analysis

def correlation_analysis(X,Y):
    #calculate correlation matrix
    correlation_matrix = X.corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',center=0,fmt='.2f')
    plt.title('Feature correlation Matrix',fontsize=16,fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    #Correlation with target
    correlation_with_target=X.coorwith(Y)
    print('Correlation with Target (high_performance):')
    print(correlation_with_target.sort_values(ascending=False))
    
    #plot correlation with target
    plt.figure(figsize=(10,6))
    correlation_with_target.sort_values().plot(kind='barh',color='skyblue')
    plt.title('Feature Correlation with Target',fontsize=14,fontweight='bold')
    plt.tight_layout()
    plt.show()

# 2. Statistical Tests (ANOVA and Chi-square)
def statistical_test(X,Y):
    anova_results={}
    continuous_features=['battingPos','runs','balls','4s','6s','SR']
    for feature in continous_features:
        group1=X[feature][Y==0]
        group2=X[feature][Y==1]
        f_stat,p_value=f_oneway(group1,group)
        anova_results[feature]={'F-statistic':f_stat,'p-value':p_value}
    anova_df = pd.DataFrame(anova_results).T
    anova_df['significant']=anova_df['p-value']<0.05
    print('ANOVA Results:')
    print(anova_df.sort_values('F-statistic',ascending=False))
    return anova_df
statistical_results=statistical_test(X,Y)

# 3. Univariate Feature Selection

def univariate_selection(X, y):
    # SelectKBest with ANOVA F-value
    selector_anova = SelectKBest(score_func=f_classif, k='all')
    selector_anova.fit(X, y)
    
    # SelectKBest with Mutual Information
    selector_mi = SelectKBest(score_func=mutual_info_classif, k='all')
    selector_mi.fit(X, y)
    
    # Create results DataFrame
    univariate_results = pd.DataFrame({
        'Feature': X.columns,
        'ANOVA_F_Score': selector_anova.scores_,
        'Mutual_Info_Score': selector_mi.scores_
    }).sort_values('ANOVA_F_Score', ascending=False)
    
    print("Univariate Feature Scores:")
    print(univariate_results)
    
    # Plot univariate feature importance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.barplot(x='ANOVA_F_Score', y='Feature', data=univariate_results, ax=ax1)
    ax1.set_title('ANOVA F-Scores', fontweight='bold')
    ax1.set_xlabel('F-Score')
    
    sns.barplot(x='Mutual_Info_Score', y='Feature', data=univariate_results, ax=ax2)
    ax2.set_title('Mutual Information Scores', fontweight='bold')
    ax2.set_xlabel('MI Score')
    
    plt.tight_layout()
    plt.show()
    
    return univariate_results

univariate_results = univariate_selection(X, y)

# 4. Recursive Feature Elimination (RFE)

def recursive_feature_elimination(X, y):
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use Random Forest for RFE
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    rfe = RFE(estimator=model, n_features_to_select=5)
    rfe.fit(X_scaled, y)
    
    rfe_results = pd.DataFrame({
        'Feature': X.columns,
        'RFE_Ranking': rfe.ranking_,
        'RFE_Selected': rfe.support_
    }).sort_values('RFE_Ranking')
    
    print("RFE Results:")
    print(rfe_results)
    
    # Plot RFE rankings
    plt.figure(figsize=(10, 6))
    sns.barplot(x='RFE_Ranking', y='Feature', data=rfe_results.sort_values('RFE_Ranking'))
    plt.title('RFE Feature Rankings', fontweight='bold')
    plt.xlabel('Ranking (1 = Best)')
    plt.tight_layout()
    plt.show()
    
    return rfe_results

rfe_results = recursive_feature_elimination(X, y)

# 5. Tree-based Feature Importance


def tree_based_importance(X, y):
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_scaled, y)
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'RF_Importance': rf.feature_importances_,
        'GB_Importance': gb.feature_importances_
    }).sort_values('RF_Importance', ascending=False)
    
    print("Tree-based Feature Importance:")
    print(feature_importance)
    
    # Plot feature importance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.barplot(x='RF_Importance', y='Feature', data=feature_importance, ax=ax1)
    ax1.set_title('Random Forest Feature Importance', fontweight='bold')
    
    sns.barplot(x='GB_Importance', y='Feature', data=feature_importance, ax=ax2)
    ax2.set_title('Gradient Boosting Feature Importance', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return feature_importance

tree_importance = tree_based_importance(X, y)

# 6. Lasso Feature Selection

def lasso_selection(X, y):
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use Lasso for feature selection
    lasso = Lasso(alpha=0.01, random_state=42)
    lasso.fit(X_scaled, y)
    
    lasso_results = pd.DataFrame({
        'Feature': X.columns,
        'Lasso_Coefficient': lasso.coef_
    }).sort_values('Lasso_Coefficient', key=abs, ascending=False)
    
    print("Lasso Feature Selection Results:")
    print(lasso_results)
    
    # Plot Lasso coefficients
    plt.figure(figsize=(10, 6))
    coefficients = pd.Series(lasso.coef_, index=X.columns)
    coefficients.sort_values(key=abs, ascending=False).plot(kind='bar')
    plt.title('Lasso Coefficients', fontweight='bold')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.tight_layout()
    plt.show()
    
    return lasso_results

lasso_results = lasso_selection(X, y)

# 7. Variance Threshold
def variance_threshold_selection(X):
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(X)
    
    variance_results = pd.DataFrame({
        'Feature': X.columns,
        'Variance': selector.variances_,
        'Selected': selector.get_support()
    }).sort_values('Variance', ascending=False)
    
    print("Variance Threshold Results:")
    print(variance_results)
    
    # Plot variances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Variance', y='Feature', data=variance_results)
    plt.title('Feature Variances', fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return variance_results

variance_results = variance_threshold_selection(X)

# 8. Comprehensive Feature Ranking

def comprehensive_ranking(X, y, all_results):
    # Create comprehensive ranking table
    comparative_ranking = pd.DataFrame({'Feature': X.columns})
    
    # Add rankings from different methods
    comparative_ranking['Correlation_Rank'] = correlation_results.rank(ascending=False)
    comparative_ranking['ANOVA_Rank'] = all_results['univariate']['ANOVA_F_Score'].rank(ascending=False)
    comparative_ranking['MI_Rank'] = all_results['univariate']['Mutual_Info_Score'].rank(ascending=False)
    comparative_ranking['RFE_Rank'] = all_results['rfe']['RFE_Ranking']
    comparative_ranking['RF_Rank'] = all_results['tree']['RF_Importance'].rank(ascending=False)
    comparative_ranking['Lasso_Rank'] = all_results['lasso']['Lasso_Coefficient'].abs().rank(ascending=False)
    
    # Calculate average rank
    rank_columns = ['Correlation_Rank', 'ANOVA_Rank', 'MI_Rank', 'RFE_Rank', 'RF_Rank', 'Lasso_Rank']
    comparative_ranking['Average_Rank'] = comparative_ranking[rank_columns].mean(axis=1)
    
    comparative_ranking = comparative_ranking.sort_values('Average_Rank')
    
    print("Comprehensive Feature Ranking:")
    print(comparative_ranking)
    
    # Plot comprehensive ranking
    plt.figure(figsize=(12, 8))
    ranking_data = comparative_ranking.set_index('Feature')[rank_columns]
    ranking_data.plot(kind='bar', figsize=(12, 8))
    plt.title('Comprehensive Feature Rankings', fontweight='bold', fontsize=14)
    plt.ylabel('Rank (Lower = Better)')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    return comparative_ranking

# Collect all results
all_results = {
    'correlation': correlation_results,
    'univariate': univariate_results,
    'rfe': rfe_results,
    'tree': tree_importance,
    'lasso': lasso_results,
    'variance': variance_results
}

comprehensive_rank = comprehensive_ranking(X, y, all_results)

# 9. Final Feature Selection and Model Validation

def final_selection_validation(X, y, comprehensive_rank, n_features=5):
    # Select top features
    top_features = comprehensive_rank.nsmallest(n_features, 'Average_Rank')['Feature'].tolist()
    print(f"Top {n_features} recommended features: {top_features}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train with all features
    model_all = RandomForestClassifier(n_estimators=100, random_state=42)
    model_all.fit(X_train_scaled, y_train)
    y_pred_all = model_all.predict(X_test_scaled)
    accuracy_all = accuracy_score(y_test, y_pred_all)
    
    # Cross-validation with all features
    cv_scores_all = cross_val_score(model_all, X_train_scaled, y_train, cv=5)
    
    # Train with selected features
    X_selected = X[top_features]
    X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_sel_scaled = scaler.fit_transform(X_train_sel)
    X_test_sel_scaled = scaler.transform(X_test_sel)
    
    model_sel = RandomForestClassifier(n_estimators=100, random_state=42)
    model_sel.fit(X_train_sel_scaled, y_train_sel)
    y_pred_sel = model_sel.predict(X_test_sel_scaled)
    accuracy_sel = accuracy_score(y_test_sel, y_pred_sel)
    
    # Cross-validation with selected features
    cv_scores_sel = cross_val_score(model_sel, X_train_sel_scaled, y_train_sel, cv=5)
    
    print(f"\nModel Performance Comparison:")
    print(f"Accuracy with all features: {accuracy_all:.4f}")
    print(f"Accuracy with selected features: {accuracy_sel:.4f}")
    print(f"CV Score with all features: {cv_scores_all.mean():.4f} (±{cv_scores_all.std():.4f})")
    print(f"CV Score with selected features: {cv_scores_sel.mean():.4f} (±{cv_scores_sel.std():.4f})")
    print(f"Number of features reduced from {X.shape[1]} to {len(top_features)}")
    
    # Plot performance comparison
    plt.figure(figsize=(10, 6))
    methods = ['All Features', 'Selected Features']
    accuracies = [accuracy_all, accuracy_sel]
    
    plt.bar(methods, accuracies, color=['lightblue', 'lightgreen'])
    plt.title('Model Accuracy Comparison', fontweight='bold')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return top_features, accuracy_all, accuracy_sel

top_features, acc_all, acc_sel = final_selection_validation(X, y, comprehensive_rank)

# 10. Feature Relationships Visualization

def visualize_feature_relationships(X, y, top_features):
    # Create pairplot of top features
    selected_df = X[top_features].copy()
    selected_df['high_performance'] = y
    
    plt.figure(figsize=(12, 10))
    sns.pairplot(selected_df, hue='high_performance', diag_kind='kde', palette='viridis')
    plt.suptitle('Pairplot of Selected Features', y=1.02, fontweight='bold')
    plt.show()
    
    # Boxplots for each selected feature
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_features, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(x='high_performance', y=feature, data=selected_df)
        plt.title(f'{feature} vs Performance', fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Correlation heatmap of selected features
    plt.figure(figsize=(8, 6))
    selected_corr = selected_df.corr()
    sns.heatmap(selected_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation of Selected Features', fontweight='bold')
    plt.tight_layout()
    plt.show()

visualize_feature_relationships(X, y, top_features)

# Summary Report
print("\n" + "="*60)
print("FEATURE SELECTION SUMMARY REPORT")
print("="*60)
print(f"Total features analyzed: {X.shape[1]}")
print(f"Recommended features: {top_features}")
print(f"Most important feature: {comprehensive_rank.iloc[0]['Feature']}")
print(f"Model accuracy maintained: {acc_sel:.4f} (vs {acc_all:.4f} with all features)")
print(f"Feature reduction: {X.shape[1] - len(top_features)} features removed")
print(f"Final feature set represents {len(top_features)/X.shape[1]*100:.1f}% of original features")

# Save results to CSV
comprehensive_rank.to_csv('feature_selection_results.csv', index=False)
print("\nResults saved to 'feature_selection_results.csv'")


👉Machine Learning-এ Ensemble মানে হলো
একটা final prediction বের করতে অনেকগুলো model (বা weak learner) একসাথে ব্যবহার করা।

accuracy_score ব্যবহার করা হয় মডেল কতটা সঠিকভাবে predict করছে তা মাপার জন্য।