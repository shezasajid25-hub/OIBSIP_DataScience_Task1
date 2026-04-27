# ============================================================
#  OIBSIP Task 1 — Iris Flower Classification
#  Domain : Data Science / Machine Learning
#  Author : <YOUR NAME>
# ============================================================

# ---------- 1. Install / Import Libraries ----------
# Run this in your terminal first (once):
#   pip install pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("   OIBSIP — TASK 1 : IRIS FLOWER CLASSIFICATION")
print("=" * 60)


# ---------- 2. Load Dataset ----------
# Option A – using sklearn (no download needed, easiest)
from sklearn.datasets import load_iris

iris_raw = load_iris(as_frame=True)
df = iris_raw.frame
df.rename(columns={"target": "species"}, inplace=True)
# Map numeric labels back to species names
df["species_name"] = df["species"].map(
    {0: "setosa", 1: "versicolor", 2: "virginica"}
)

print("\n[INFO] Dataset loaded successfully.")
print(f"Shape : {df.shape}")
print("\nFirst 5 rows:")
print(df.head())


# ---------- 3. Exploratory Data Analysis (EDA) ----------
print("\n--- Basic Statistics ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Class Distribution ---")
print(df["species_name"].value_counts())

# --- Pairplot ---
sns.pairplot(df, hue="species_name", palette="Set2",
             vars=["sepal length (cm)", "sepal width (cm)",
                   "petal length (cm)", "petal width (cm)"])
plt.suptitle("Iris — Pairplot by Species", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig("pairplot.png", dpi=150)
plt.show()
print("[INFO] Pairplot saved as pairplot.png")

# --- Correlation Heatmap ---
plt.figure(figsize=(7, 5))
corr = df.drop(columns=["species", "species_name"]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("heatmap.png", dpi=150)
plt.show()
print("[INFO] Heatmap saved as heatmap.png")

# --- Boxplot ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
features = ["sepal length (cm)", "sepal width (cm)",
            "petal length (cm)", "petal width (cm)"]
for ax, feat in zip(axes.flatten(), features):
    sns.boxplot(x="species_name", y=feat, data=df,
                palette="Set2", ax=ax)
    ax.set_title(feat)
plt.suptitle("Feature Distributions by Species", fontsize=14)
plt.tight_layout()
plt.savefig("boxplot.png", dpi=150)
plt.show()
print("[INFO] Boxplot saved as boxplot.png")


# ---------- 4. Preprocessing ----------
X = df[features]
y = df["species"]          # numeric labels (0, 1, 2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[INFO] Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")


# ---------- 5. Train & Evaluate Multiple Models ----------
models = {
    "Logistic Regression"  : LogisticRegression(max_iter=200),
    "Decision Tree"        : DecisionTreeClassifier(random_state=42),
    "Random Forest"        : RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM"                  : SVC(kernel="rbf", random_state=42),
    "K-Nearest Neighbors"  : KNeighborsClassifier(n_neighbors=5),
}

results = {}
print("\n--- Model Comparison ---")
print(f"{'Model':<25} {'Accuracy':>10}")
print("-" * 37)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name:<25} {acc*100:>9.2f}%")


# ---------- 6. Best Model — Detailed Report ----------
best_name = max(results, key=results.get)
best_model = models[best_name]
y_pred_best = best_model.predict(X_test)

print(f"\n[INFO] Best Model : {best_name}  ({results[best_name]*100:.2f}%)")
print("\n--- Classification Report ---")
print(classification_report(
    y_test, y_pred_best,
    target_names=["setosa", "versicolor", "virginica"]
))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["setosa", "versicolor", "virginica"],
            yticklabels=["setosa", "versicolor", "virginica"])
plt.title(f"Confusion Matrix — {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("[INFO] Confusion matrix saved as confusion_matrix.png")


# ---------- 7. Model Accuracy Bar Chart ----------
plt.figure(figsize=(9, 5))
bars = plt.barh(list(results.keys()),
                [v * 100 for v in results.values()],
                color=sns.color_palette("Set2", len(results)))
plt.xlabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")
plt.xlim(80, 102)
for bar, val in zip(bars, results.values()):
    plt.text(val * 100 + 0.2, bar.get_y() + bar.get_height() / 2,
             f"{val*100:.2f}%", va="center", fontsize=10)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()
print("[INFO] Model comparison chart saved as model_comparison.png")


# ---------- 8. Predict on New Sample ----------
print("\n--- Predict a New Flower ---")
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=features)
species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
prediction = best_model.predict(sample)[0]
print(f"Input  : {sample.values.tolist()[0]}")
print(f"Predicted Species : {species_map[prediction]}")

print("\n[DONE] Task 1 complete! All plots saved in the current folder.")