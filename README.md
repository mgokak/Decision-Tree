# Decision Tree & Diabetes Prediction Models

## Overview

This repository contains Jupyter Notebooks that demonstrate **Decision Tree models** and their application to real datasets. The notebooks focus on understanding how decision trees work, how data is split based on features, and how tree-based models can be applied to prediction problems.

---

## Table of Contents

1. Installation  
2. Project Structure  
3. Decision Tree on Iris Dataset  
4. Diabetes Prediction Dataset Analysis  
5. Model Evaluation  

---

## Installation

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Project Structure

- `DecisionTree.ipynb` – Decision Tree applied to the Iris dataset  
- `DiabetesPrediction.ipynb` – Dataset exploration and predictive modeling using the diabetes dataset  

---

## Decision Tree on Iris Dataset

### `DecisionTree.ipynb`

This notebook demonstrates **Decision Tree classification** using the well-known **Iris dataset**.

Key points:
- Uses a structured dataset with multiple classes
- Shows how features are used to split data
- Helps visualize hierarchical decision-making

Basic commands used:
```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data
y = iris.target

model = DecisionTreeClassifier()
model.fit(X, y)
```

Decision trees are easy to interpret and useful for understanding feature importance.

---

## Diabetes Prediction Dataset Analysis

### `DiabetesPrediction.ipynb`

This notebook works with the **Diabetes dataset** from `scikit-learn`.

Key points:
- Dataset loading and exploration
- Feature understanding and data preparation
- Applying predictive models to medical data

Basic commands used:
```python
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
```

The notebook focuses on understanding the dataset before building predictive models.

---

## Model Evaluation

Common evaluation techniques include:

```python
from sklearn.metrics import accuracy_score, mean_squared_error
```

- Accuracy (for classification)
- Error-based metrics (for prediction tasks)
- Interpretation of model results

---

## Author

**Manasa Vijayendra Gokak**  
Graduate Student – Data Science  
DePaul University
