import os
import numpy as np
import pandas as pd
import pickle
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from utils import ensure_dir, seed_everything

def train_and_evaluate(features_path, labels_path, output_dir):
    seed_everything()
    ensure_dir(output_dir)
    
    # Load data
    X = np.load(features_path)
    df_labels = pd.read_csv(labels_path)
    y = df_labels['page_id'].values
    
    # Split data: 70% train, 15% validation, 15% test
    # Note: In a real scenario, we'd ensure strips from the same page are split by x-position
    # Here we'll do a simple split but the metadata has x_start for better splitting if needed
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    
    classifiers = {
        'SVM': LinearSVC(random_state=42, max_iter=10000),
        'LogReg': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    results = []
    
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        clf.fit(X_train, y_train)
        
        # Save model
        with open(os.path.join(output_dir, f'{name}_model.pkl'), 'wb') as f:
            pickle.dump(clf, f)
            
        # Basic evaluation
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")
        
    # Save split data for evaluation script
    np.savez(os.path.join(output_dir, 'split_data.npz'), 
             X_train=X_train, X_test=X_test, X_val=X_val,
             y_train=y_train, y_test=y_test, y_val=y_val)

if __name__ == "__main__":
    train_and_evaluate(
        "wasil_project/outputs/features/features_pca.npy",
        "wasil_project/outputs/features/labels.csv",
        "wasil_project/outputs/models"
    )
