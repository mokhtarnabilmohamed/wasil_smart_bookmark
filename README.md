# Wasil Project: Smart Bookmark Concept

## Project Idea

Wasil is a smart bookmark concept designed to identify the **book name and page number** from a narrow vertical strip captured from a book page. This project simulates a CIS-like scanner digitally, using full-page images as source material to generate narrow vertical strips as model input. The system is designed for scalability to accommodate additional books in the future, with the initial implementation focusing on a single book: **Mushaf Al-Madinah**.

## Project Objective

The primary objective of the Wasil project is to develop a computer vision system capable of accurately predicting the page number of a book from a 3mm vertical strip. This involves:

1.  **Dataset Selection and Preparation**: Utilizing a controlled dataset of Mushaf Al-Madinah page images and generating a strip dataset from these images.
2.  **Data Preprocessing**: Implementing image preprocessing techniques for normalization and optional augmentation.
3.  **Feature Extraction**: Employing a pre-trained Convolutional Neural Network (CNN), specifically ResNet50, for extracting robust features from the image strips, followed by PCA for dimensionality reduction and StandardScaler.
4.  **Classification**: Training and evaluating at least two traditional machine learning classifiers (Linear SVM, Logistic Regression, and optionally Random Forest) to predict the page number.
5.  **Evaluation**: Assessing model performance using a comprehensive set of metrics including Top-K accuracy, F1-score, Recall, Precision, ROC-AUC, and accuracy within a certain page difference.
6.  **Analysis**: Providing a conclusive analysis of the system's viability as a smart bookmark and its potential for future scalability.

## Dataset Preparation

The project uses a controlled dataset based on Mushaf Al-Madinah page images. Since automatic download was not implemented, the system expects full-page images to be placed in the `dataset/pages/` directory, named sequentially (e.g., `page_001.png`, `page_002.png`, up to `page_604.png`).

For this demonstration, 10 synthetic full-page images were generated to simulate the input. These synthetic pages contain varying patterns and text to make them distinguishable for the model.

## 3 mm Strip Generation Method

Vertical strips are generated from the full-page images to simulate the output of a 3mm bookmark scanner. The strip generation process involves:

*   **Strip Height**: Full page height.
*   **Strip Width**: 3 mm, converted to pixels based on the actual image width using the formula `strip_width_px = int((3 / 140) * image_width_px)`. For the synthetic pages, this resulted in a pixel width proportional to the 3mm physical width.
*   **Cropping**: Multiple vertical strips are generated for each page using random x-position cropping across the page width. This ensures a diverse set of strips for training and testing.
*   **Labeling**: Each generated strip is labeled with its book name (e.g., `mushaf`) and page number (e.g., `page_014`).
*   **Metadata**: A `metadata.csv` file is created, storing details such as `filename`, `book_id`, `page_id`, `strip_width_mm`, `x_start`, and `x_end` for each strip.

### Experimental Design for Data Splitting

To ensure robust evaluation and prevent data leakage, the generated 3mm strips are split into training, validation, and testing sets with the following proportions:

*   **Training**: 70%
*   **Validation**: 15%
*   **Test**: 15%

Crucially, test strips are ensured to come from x-positions different from the training strips. This addresses the main research question: 
Can the model recognize the correct book and page from a new 3 mm strip position that it did not see during training?

## Preprocessing

For each generated strip, the following preprocessing steps are applied:

1.  **Color Conversion**: Convert to RGB if the image is grayscale.
2.  **Resizing/Padding**: Resize or pad the image to `224x224` pixels, which is the standard input size for many pre-trained CNNs.
3.  **Normalization**: Normalize pixel values using ImageNet mean and standard deviation (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
4.  **Optional Augmentations**: To improve model robustness and generalization, optional augmentations are applied:
    *   Small rotation between ±2 and ±5 degrees.
    *   Brightness/contrast jitter.
    *   Slight blur.
    *   Light noise.

## CNN Feature Extraction

Feature extraction is performed using a pre-trained Convolutional Neural Network (CNN). The chosen model is **ResNet50**, pre-trained on the ImageNet dataset. The final classification head of ResNet50 is removed, and the Global Average Pooling embedding is used as the feature vector. This results in a 2048-dimensional feature vector for each strip.

To further process these features, the following steps are applied:

1.  **PCA**: Principal Component Analysis (PCA) is used to reduce the dimensionality from 2048 features to 256 features. This helps in reducing computational complexity and mitigating the curse of dimensionality.
2.  **StandardScaler**: The features are then scaled using `StandardScaler` to ensure that each feature contributes equally to the classification task.

## Classifiers Used

At least two traditional machine learning classifiers are trained on the extracted and processed features. For this project, the following classifiers were implemented:

1.  **LinearSVC (Support Vector Machine)**: A linear classifier that aims to find the hyperplane that best separates the classes.
2.  **Logistic Regression**: A linear model used for binary and multi-class classification.
3.  **Random Forest**: An ensemble learning method that constructs a multitude of decision trees during training and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

The task for these classifiers is to predict the `page_id` given a 3mm strip image. While the current implementation focuses on a single book (Mushaf Al-Madinah), the architecture is designed to be extensible for future multi-book classification.

## Evaluation Metrics

For each classifier, a comprehensive set of evaluation metrics is computed to assess performance:

*   **Top-1 Accuracy**: The proportion of correctly predicted pages.
*   **Top-3 Accuracy**: The proportion of cases where the correct page is among the top 3 predicted pages.
*   **Top-5 Accuracy**: The proportion of cases where the correct page is among the top 5 predicted pages.
*   **Macro Precision**: The average precision for each class, weighted by the number of true instances for each class.
*   **Macro Recall / Sensitivity**: The average recall for each class, weighted by the number of true instances for each class.
*   **Macro F1-score**: The harmonic mean of precision and recall, averaged across classes.
*   **Confusion Matrix**: A table that visualizes the performance of an algorithm, typically used in supervised learning.
*   **Macro ROC-AUC**: The Area Under the Receiver Operating Characteristic Curve, averaged across classes using a One-vs-Rest strategy.
*   **Within ±1 page accuracy**: The proportion of predictions that are within ±1 page of the true page number.
*   **Within ±5 pages accuracy**: The proportion of predictions that are within ±5 pages of the true page number.

## Final Results

The evaluation results for the classifiers are summarized in the table below:

| Classifier   | Top-1 Acc | Top-3 Acc | Top-5 Acc | F1       | Recall   | Precision | ±1 Acc   | ±5 Acc   | ROC-AUC  |
| :----------- | :-------- | :-------- | :-------- | :------- | :------- | :-------- | :------- | :------- | :------- |
| SVM          | 0.033     | 0.167     | 0.300     | 0.010    | 0.033    | 0.006     | 0.267    | 0.700    | 0.380    |
| LogReg       | 0.067     | 0.100     | 0.300     | 0.043    | 0.067    | 0.039     | 0.367    | 0.900    | 0.428    |
| RandomForest | 0.400     | 0.700     | 0.800     | 0.391    | 0.400    | 0.400     | 0.667    | 0.900    | 0.819    |

*Note: The precision values for SVM and Logistic Regression are very low due to the small synthetic dataset and the nature of multi-class classification with limited samples per class, leading to undefined precision for some classes.* 

Confusion matrices for each classifier are generated and saved in `outputs/confusion_matrices/`.

## Conclusion

Based on the experimental results with the synthetic dataset:

1.  **Is a 3 mm strip enough to identify the correct page?**
    The results suggest that a 3mm strip can provide some signal for page identification, especially with the Random Forest classifier achieving a Top-1 accuracy of 40% and a Top-5 accuracy of 80%. The ±5 Acc of 90% for both Logistic Regression and Random Forest indicates that the model can often narrow down the page to a small range around the correct one. However, for precise identification (Top-1 accuracy), 3mm might be challenging with complex real-world book pages without more advanced techniques or multiple readings.

2.  **Which classifier performed best?**
    The **Random Forest** classifier significantly outperformed both Linear SVM and Logistic Regression across all metrics, achieving the highest Top-1, Top-3, Top-5, F1, Recall, Precision, and ROC-AUC scores. This indicates its superior ability to handle the extracted features and the complexity of the classification task.

3.  **Is this project realistic as a smart bookmark?**
    With the current performance on synthetic data, the project shows promise. A Top-1 accuracy of 40% is not sufficient for a reliable smart bookmark, but a Top-5 accuracy of 80% and ±5 Acc of 90% suggests that it could provide a list of highly probable pages. For a practical smart bookmark, higher precision and Top-1 accuracy would be desirable.

4.  **Does it need multiple readings or a wider strip in future work?**
    Given the current results, it is highly probable that **multiple readings** (e.g., scanning several 3mm strips from different parts of the page) or a **wider strip** would significantly improve accuracy. Multiple readings could be combined using a voting mechanism or by concatenating features. A wider strip would provide more visual information, making the classification task easier.

5.  **Are the results good enough for a Computer Vision project?**
    For a proof-of-concept or initial research project, these results are a good starting point, especially considering the limited and synthetic nature of the dataset. However, for a production-ready application, the accuracy would need to be substantially higher.

6.  **What is the best success rate achieved?**
    The best Top-1 accuracy achieved was **40%** by the Random Forest classifier. The best Top-5 accuracy was **80%** (Random Forest), and the best ±5 pages accuracy was **90%** (Logistic Regression and Random Forest).

7.  **Can the architecture scale later to multiple books?**
    Yes, the architecture is designed to scale to multiple books. The `book_id` is already part of the strip labeling and metadata. For multi-book classification, the output layer of the classifiers would need to predict both `book_id` and `page_id` (potentially as a hierarchical classification or a combined label), and the dataset would need to include strips from various books. The feature extraction pipeline (ResNet50 + PCA) is generic enough to handle features from different books.

## Future Work

*   Acquire and use a real dataset of Mushaf Al-Madinah pages for more realistic evaluation.
*   Experiment with different strip widths and explore the impact of multiple strip readings per page.
*   Investigate more advanced deep learning architectures for classification, potentially fine-tuning the ResNet50 or using other state-of-the-art models.
*   Implement a hierarchical classification approach for multi-book and multi-page prediction.
*   Explore techniques for handling variations in lighting, paper quality, and scanning artifacts.

## Project Structure

```
wasil_project/
│
├── dataset/
│   ├── pages/              # Full page images (e.g., page_001.png)
│   └── strips_3mm/         # Generated 3mm strip images
│   └── metadata.csv        # Metadata for generated strips
│
├── notebooks/
│   └── wasil_experiment.ipynb # Jupyter notebook for experimentation (future work)
│
├── src/
│   ├── generate_strips.py  # Script to generate vertical strips from full pages
│   ├── preprocess.py       # Image preprocessing utilities
│   ├── extract_features.py # Script for CNN feature extraction and PCA
│   ├── train_classifiers.py# Script to train traditional ML classifiers
│   ├── evaluate.py         # Script to evaluate classifier performance
│   └── utils.py            # Utility functions (seeding, device setup, etc.)
│   ├── create_demo_data.py # Script to create synthetic demo pages
│   └── plot_results.py     # Script to generate comparison plots
│
├── outputs/
│   ├── features/           # Extracted features (features_pca.npy, labels.csv, scaler.pkl, pca.pkl)
│   ├── models/             # Trained classifier models (e.g., SVM_model.pkl)
│   ├── results.csv         # Summary of evaluation results
│   ├── confusion_matrices/ # Confusion matrix plots
│   └── plots/              # Comparison plots
│
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```
