import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_comparison_plots(results_csv, output_dir):
    if not os.path.exists(results_csv):
        print(f"Results file not found: {results_csv}")
        return
        
    df = pd.read_csv(results_csv)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style
    sns.set_theme(style="whitegrid")
    
    # 1. Accuracy Comparison
    plt.figure(figsize=(10, 6))
    metrics = ['Top-1 Acc', 'Top-3 Acc', 'Top-5 Acc', '±1 Acc']
    df_melted = df.melt(id_vars='Classifier', value_vars=metrics, var_name='Metric', value_name='Score')
    sns.barplot(data=df_melted, x='Metric', y='Score', hue='Classifier')
    plt.title('Accuracy Comparison across Classifiers')
    plt.ylim(0, 1.1)
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    plt.close()
    
    # 2. F1, Precision, Recall
    plt.figure(figsize=(10, 6))
    metrics = ['F1', 'Precision', 'Recall']
    df_melted = df.melt(id_vars='Classifier', value_vars=metrics, var_name='Metric', value_name='Score')
    sns.barplot(data=df_melted, x='Metric', y='Score', hue='Classifier')
    plt.title('Classification Metrics Comparison')
    plt.ylim(0, 1.1)
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
    plt.close()

if __name__ == "__main__":
    generate_comparison_plots(
        "wasil_project/outputs/results.csv",
        "wasil_project/outputs/plots"
    )
