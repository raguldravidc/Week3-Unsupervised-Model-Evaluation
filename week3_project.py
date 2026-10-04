import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(BASE, "images")
os.makedirs(IMG, exist_ok=True)

# ---------------- Load datasets ----------------
customers = pd.read_csv(os.path.join(BASE, "customer_segmentation.csv"))
clf = pd.read_csv(os.path.join(BASE, "classification_dataset.csv"))

# ---------------- Unsupervised learning ----------------
features = ["Age", "Annual_Income", "Spending_Score", "Annual_Visits"]
X = customers[features]
X_scaled = StandardScaler().fit_transform(X)

# Elbow method
inertias = []
for k in range(2, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(range(2, 7), inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Method")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG, "elbow_curve.png"), dpi=160)
plt.close()

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
customers["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)
kmeans_silhouette = silhouette_score(X_scaled, customers["KMeans_Cluster"])

# PCA visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(7, 5))
for c in sorted(customers["KMeans_Cluster"].unique()):
    idx = customers["KMeans_Cluster"] == c
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], s=28, label=f"Cluster {c}")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clusters using PCA")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG, "clusters_pca.png"), dpi=160)
plt.close()

# Hierarchical clustering + dendrogram
Z = linkage(X_scaled, method="ward")
plt.figure(figsize=(9, 5))
dendrogram(Z, truncate_mode="lastp", p=20, leaf_rotation=90, leaf_font_size=8)
plt.xlabel("Samples / merged clusters")
plt.ylabel("Distance")
plt.title("Hierarchical Clustering Dendrogram")
plt.tight_layout()
plt.savefig(os.path.join(IMG, "dendrogram.png"), dpi=160)
plt.close()

hier = AgglomerativeClustering(n_clusters=3, linkage="ward")
customers["Hierarchical_Cluster"] = hier.fit_predict(X_scaled)
hier_silhouette = silhouette_score(X_scaled, customers["Hierarchical_Cluster"])

# ---------------- Model evaluation ----------------
X = clf.drop(columns="Target")
y = clf["Target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])
pipe.fit(X_train, y_train)

pred = pipe.predict(X_test)
prob = pipe.predict_proba(X_test)[:, 1]

print("\nMODEL EVALUATION")
print("Accuracy :", round(accuracy_score(y_test, pred), 3))
print("Precision:", round(precision_score(y_test, pred), 3))
print("Recall   :", round(recall_score(y_test, pred), 3))
print("F1-score :", round(f1_score(y_test, pred), 3))
print("ROC-AUC  :", round(roc_auc_score(y_test, prob), 3))

# Confusion matrix
cm = confusion_matrix(y_test, pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot(colorbar=False)
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig(os.path.join(IMG, "confusion_matrix.png"), dpi=160)
plt.close()

# ROC curve
RocCurveDisplay.from_predictions(y_test, prob)
plt.title("ROC Curve - Logistic Regression")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG, "roc_curve.png"), dpi=160)
plt.close()

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
print("5-Fold CV Mean:", round(cv_scores.mean(), 3))

# Hyperparameter tuning
grid = GridSearchCV(
    pipe,
    {"model__C": [0.01, 0.1, 1, 10, 100]},
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)
grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)
print("Tuned CV Accuracy:", round(grid.best_score_, 3))

print("\nK-Means Silhouette:", round(kmeans_silhouette, 3))
print("Hierarchical Silhouette:", round(hier_silhouette, 3))
print("\nGraphs saved in:", IMG)
