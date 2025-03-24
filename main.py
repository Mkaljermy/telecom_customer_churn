# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Set random seed for reproducibility
np.random.seed(42)

# 1. Import Dataset
print("Step 1: Importing the dataset")
df = pd.read_csv('Telecom_Customer_Churn.csv')

# Display basic information about the dataset
print("\nDataset Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)
print("\nFirst few rows:")
print(df.head())

# 2. Exploratory Data Analysis
print("\nStep 2: Exploratory Data Analysis")

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Check class distribution (assuming 'Churn' is the target variable)
if 'Churn' in df.columns:
    target_column = 'Churn'
elif 'churn' in df.columns:
    target_column = 'churn'
else:
    # Try to identify the target column based on binary values
    binary_cols = [col for col in df.columns if df[col].nunique() == 2]
    if binary_cols:
        target_column = binary_cols[0]  # Assume the first binary column is the target
        print(f"\nAssuming '{target_column}' is the target variable.")
    else:
        target_column = None
        print("\nWARNING: Could not identify a target column. Please specify the correct column.")

if target_column:
    print(f"\nClass Distribution ({target_column}):")
    print(df[target_column].value_counts())
    print(f"Churn Rate: {df[target_column].mean() * 100:.2f}%")

# Identify categorical and numerical features
if target_column:
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != target_column]
    numeric_cols = [col for col in df.columns if df[col].dtype != 'object' and col != target_column]
else:
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
    numeric_cols = [col for col in df.columns if df[col].dtype != 'object']

print("\nCategorical columns:", categorical_cols)
print("\nNumerical columns:", numeric_cols)

# 3. Data Preprocessing
print("\nStep 3: Data Preprocessing")

# Handle missing values
# For numerical features, use median imputation
# For categorical features, use most frequent value imputation

# Create preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Prepare data for modeling
if target_column:
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Convert target to numeric if it's categorical
    if y.dtype == 'object':
        print(f"\nConverting {target_column} to numeric...")
        y = y.map({'Yes': 1, 'No': 0}) if 'Yes' in y.unique() else y.map({True: 1, False: 0})
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Check for class imbalance
    print("\nClass distribution in training set:")
    print(y_train.value_counts())
    
    # Handle class imbalance if necessary
    if y_train.value_counts().min() / y_train.value_counts().sum() < 0.25:
        print("\nApplying SMOTE for handling class imbalance...")
        smote = SMOTE(random_state=42)
        
        # First apply preprocessing
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        
        # Then apply SMOTE
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)
        
        print("After SMOTE, class distribution:")
        print(pd.Series(y_train_resampled).value_counts())
        
        # For model training, we'll use the resampled data
        # But preprocessor has already been fit, so we need a different approach
        X_test_preprocessed = preprocessor.transform(X_test)
        
        # 4. Model Development
        print("\nStep 4: Model Development")
        
        # Choose Random Forest as our model
        model = RandomForestClassifier(random_state=42)
        
        # Train the model
        model.fit(X_train_resampled, y_train_resampled)
        
        # Make predictions
        y_pred = model.predict(X_test_preprocessed)
        y_pred_proba = model.predict_proba(X_test_preprocessed)[:, 1]
        
        # Calculate performance metrics
        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Churn', 'Churn'], 
                    yticklabels=['No Churn', 'Churn'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig('roc_curve.png')
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_names = (
                numeric_cols + 
                list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols))
            )
            
            # Get feature importances
            importances = model.feature_importances_
            
            # Sort feature importances in descending order
            indices = np.argsort(importances)[::-1]
            
            # Rearrange feature names so they match the sorted feature importances
            names = [feature_names[i] for i in indices]
            
            # Create plot
            plt.figure(figsize=(12, 8))
            plt.title("Feature Importance")
            plt.bar(range(len(indices)), importances[indices], align='center')
            plt.xticks(range(len(indices)), names, rotation=90)
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            
            print("\nTop 10 most important features:")
            for i in range(min(10, len(names))):
                print(f"{names[i]}: {importances[indices[i]]:.4f}")
        
    else:
        # If no class imbalance, use a simpler pipeline
        # Create the full pipeline with the model
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # 4. Model Development
        print("\nStep 4: Model Development")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate performance metrics
        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Churn', 'Churn'], 
                    yticklabels=['No Churn', 'Churn'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig('roc_curve.png')
        
        # Optional: Perform hyperparameter tuning
        print("\nOptional: Hyperparameter Tuning")
        print("Note: This may take some time to run...")
        
        # Simplified parameter grid for demonstration
        param_grid = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5]
        }
        
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print("\nBest Parameters:", grid_search.best_params_)
        print("Best F1 Score:", grid_search.best_score_)
        
        # Use the best model for final evaluation
        best_model = grid_search.best_estimator_
        y_pred_best = best_model.predict(X_test)
        
        print("\nFinal Model Evaluation (with tuned hyperparameters):")
        print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_best))
        
        # Feature importance for the tuned model
        if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
            feature_names = (
                numeric_cols + 
                list(best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols))
            )
            
            # Get feature importances
            importances = best_model.named_steps['classifier'].feature_importances_
            
            # Sort feature importances in descending order
            indices = np.argsort(importances)[::-1]
            
            # Rearrange feature names so they match the sorted feature importances
            names = [feature_names[i] for i in indices]
            
            # Create plot
            plt.figure(figsize=(12, 8))
            plt.title("Feature Importance (Tuned Model)")
            plt.bar(range(len(indices)), importances[indices], align='center')
            plt.xticks(range(len(indices)), names, rotation=90)
            plt.tight_layout()
            plt.savefig('feature_importance_tuned.png')
            
            print("\nTop 10 most important features (tuned model):")
            for i in range(min(10, len(names))):
                print(f"{names[i]}: {importances[indices[i]]:.4f}")
else:
    print("\nSkipping model training as target variable was not identified.")

print("\nAnalysis Complete!")
