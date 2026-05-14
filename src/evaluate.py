import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score)
from sklearn.preprocessing import LabelBinarizer
from utils import ensure_dir

def top_k_accuracy(y_true, y_probs, k=3, classes=None):
    if y_probs is None:
        return 0
    top_k_indices = np.argsort(y_probs, axis=1)[:, -k:]
    correct = 0
    for i in range(len(y_true)):
        predicted_classes = classes[top_k_indices[i]]
        if y_true[i] in predicted_classes:
            correct += 1
    return correct / len(y_true)

def page_diff_accuracy(y_true_ids, y_pred_ids, threshold=1):
    # Extract numeric page numbers from 'page_001' strings
    y_true_num = np.array([int(pid.split('_')[1]) for pid in y_true_ids])
    y_pred_num = np.array([int(pid.split('_')[1]) for pid in y_pred_ids])
    
    diffs = np.abs(y_true_num - y_pred_num)
    return np.mean(diffs <= threshold)

def evaluate_all(models_dir, split_data_path, output_dir):
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, 'plots'))
    ensure_dir(os.path.join(output_dir, 'confusion_matrices'))
    
    data = np.load(split_data_path, allow_pickle=True)
    X_test = data['X_test']
    y_test = data['y_test']
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.pkl')]
    
    summary_results = []
    
    # For ROC-AUC
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test_bin = lb.transform(y_test)
    
    for model_file in model_files:
        name = model_file.replace('_model.pkl', '')
        with open(os.path.join(models_dir, model_file), 'rb') as f:
            clf = pickle.load(f)
            
        y_pred = clf.predict(X_test)
        
        # Get probabilities for top-k and ROC-AUC
        if hasattr(clf, "predict_proba"):
            y_probs = clf.predict_proba(X_test)
        elif hasattr(clf, "decision_function"):
            # For LinearSVC, convert decision function to "probabilities" using softmax
            dfunc = clf.decision_function(X_test)
            y_probs = np.exp(dfunc) / np.sum(np.exp(dfunc), axis=1, keepdims=True)
        else:
            y_probs = None

        # Metrics
        acc1 = accuracy_score(y_test, y_pred)
        acc3 = top_k_accuracy(y_test, y_probs, k=3, classes=clf.classes_)
        acc5 = top_k_accuracy(y_test, y_probs, k=5, classes=clf.classes_)
        f1 = f1_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        
        acc_pm1 = page_diff_accuracy(y_test, y_pred, threshold=1)
        acc_pm5 = page_diff_accuracy(y_test, y_pred, threshold=5)
        
        try:
            roc_auc = roc_auc_score(y_test_bin, y_probs, multi_class='ovr', average='macro')
        except:
            roc_auc = 0
            
        summary_results.append({
            'Classifier': name,
            'Top-1 Acc': acc1,
            'Top-3 Acc': acc3,
            'Top-5 Acc': acc5,
            'F1': f1,
            'Recall': recall,
            'Precision': precision,
            '±1 Acc': acc_pm1,
            '±5 Acc': acc_pm5,
            'ROC-AUC': roc_auc
        })
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Page')
        plt.xlabel('Predicted Page')
        plt.savefig(os.path.join(output_dir, 'confusion_matrices', f'cm_{name}.png'))
        plt.close()

    df_summary = pd.DataFrame(summary_results)
    df_summary.to_csv(os.path.join(output_dir, 'results.csv'), index=False)
    
    # Print table
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    evaluate_all(
        "wasil_project/outputs/models",
        "wasil_project/outputs/models/split_data.npz",
        "wasil_project/outputs"
    )
