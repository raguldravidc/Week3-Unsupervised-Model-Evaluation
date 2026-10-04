# Week 3: Unsupervised Learning & Model Evaluation

## Project Title
**Customer Segmentation and Machine Learning Model Evaluation**

## Objective
This project demonstrates:
- K-Means clustering
- Hierarchical clustering
- Elbow method
- PCA dimensionality reduction
- Confusion matrix
- Accuracy, precision, recall and F1-score
- ROC-AUC and ROC curve
- 5-fold cross-validation
- Hyperparameter tuning with GridSearchCV

## Datasets
1. `customer_segmentation.csv` – 300 synthetic customer records used for clustering.
2. `classification_dataset.csv` – 500 synthetic binary-classification records used for model evaluation.

The datasets are generated reproducibly with `random_state=42`.

## Technologies
Python, Pandas, NumPy, Matplotlib, Scikit-learn, SciPy.

## How to Run
```bash
pip install -r requirements.txt
python week3_project.py
```

The script creates the analysis graphs inside the `images/` folder and saves CSV result files.

## Clustering Results
K-Means was tested for k=2 to k=6. The project uses **k=3** for the final segmentation.
- K-Means silhouette score: **0.229**
- Hierarchical clustering silhouette score: **0.192**

PCA was used to reduce the standardized four-feature customer data to two dimensions for visualization.

## Model Evaluation
A Logistic Regression classifier was evaluated using a stratified 80/20 train-test split.

- Accuracy: **0.850**
- Precision: **0.857**
- Recall: **0.800**
- F1-score: **0.828**
- ROC-AUC: **0.936**
- 5-fold CV mean accuracy: **0.886**
- Best tuned C: **1**
- Tuned CV accuracy: **0.890**

## Files
- `week3_project.py` – complete Python implementation
- `customer_segmentation.csv` – clustering dataset
- `classification_dataset.csv` – classification dataset
- `model_evaluation_metrics.csv` – evaluation metrics
- `cluster_summary.csv` – cluster-wise averages
- `images/elbow_curve.png`
- `images/clusters_pca.png`
- `images/dendrogram.png`
- `images/confusion_matrix.png`
- `images/roc_curve.png`

## Conclusion
The project shows how unsupervised learning can group similar observations without target labels and how classification models can be evaluated using multiple performance measures. Cross-validation and hyperparameter tuning were also applied to improve the reliability of the evaluation.
