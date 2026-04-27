# 🌸 OIBSIP_DataScience_Task1 — Iris Flower Classification

## 📌 Objective
Train a Machine Learning model to classify Iris flowers into three species —
**Setosa**, **Versicolor**, and **Virginica** — based on their sepal and petal measurements.

---

## 📂 Repository Structure
```
OIBSIP_DataScience_Task1/
│
├── iris_classification.py     # Main Python script
├── pairplot.png               # EDA: pairplot by species
├── heatmap.png                # Feature correlation heatmap
├── boxplot.png                # Feature distributions
├── confusion_matrix.png       # Best model confusion matrix
├── model_comparison.png       # Accuracy comparison across models
└── README.md                  # This file
```

---

## 🛠️ Tools & Libraries Used
| Tool | Purpose |
|------|---------|
| Python 3.x | Programming language |
| Pandas | Data loading & manipulation |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | ML models & evaluation |

---

## 🔢 Steps Performed
1. **Data Loading** — Used `sklearn.datasets.load_iris` (built-in, no download needed)
2. **Exploratory Data Analysis** — Pairplot, heatmap, boxplots
3. **Preprocessing** — Train/test split (80/20, stratified)
4. **Model Training** — Compared 5 classifiers:
   - Logistic Regression
   - Decision Tree
   - Random Forest ⭐ (best)
   - SVM
   - K-Nearest Neighbors
5. **Evaluation** — Accuracy, Classification Report, Confusion Matrix
6. **Prediction** — Tested on a new sample input

---

## 📊 Results
| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~96.67% |
| Decision Tree | ~96.67% |
| **Random Forest** | **~96.67%** |
| SVM | ~96.67% |
| K-Nearest Neighbors | ~96.67% |

> All models perform excellently on this classic dataset.

---

## ▶️ How to Run
```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/OIBSIP_DataScience_Task1.git
cd OIBSIP_DataScience_Task1

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# 3. Run the script
python iris_classification.py
```

---

## 🎯 Outcome
Successfully built and evaluated multiple ML classification models on the Iris dataset.
The best model achieved **~96–100% accuracy**, correctly classifying all three species.

---

## 🏢 Internship
**Organization:** Oasis Infobyte  
**Domain:** Data Science  
**Task:** Task 1 — Iris Flower Classification
