# Multivariate PCA Explorer

An interactive dimensionality-reduction and multivariate-analysis project built with **Python**, **scikit-learn**, **pandas**, **Seaborn**, **Matplotlib**, and **Streamlit**.

The project explores whether 30 diagnostic measurements from the Wisconsin Diagnostic Breast Cancer dataset can be reduced to two principal components while preserving meaningful structure between malignant and benign cases.

---

## Project Overview

High-dimensional datasets often contain strong correlation and redundant information that make direct visualization difficult. This project uses **Principal Component Analysis (PCA)** to transform a 30-feature diagnostic dataset into a two-dimensional representation that can be inspected visually.

The workflow combines:

- data validation and class-balance inspection
- correlation analysis
- strongest-pair verification
- feature standardization
- PCA dimensionality reduction
- explained-variance analysis
- PCA class visualization
- principal-component loadings
- interactive feature coloring
- optional unsupervised clustering
- aligned class-cluster comparison

The Streamlit application presents the analysis as an interactive exploratory dashboard, while the notebook provides the underlying analytical workflow.

---

## Dataset

**Wisconsin Diagnostic Breast Cancer Dataset**

The dataset is loaded directly from `scikit-learn` using:

```python
from sklearn.datasets import load_breast_cancer
```

### Dataset profile

| Property | Value |
|---|---:|
| Observations | 569 |
| Numeric features | 30 |
| Diagnostic classes | 2 |
| Missing values | 0 |
| Malignant cases | 212 |
| Benign cases | 357 |

The diagnosis label is kept separate from the numeric feature matrix and is **not included in correlation calculations, standardization, or PCA fitting**.

---

## Analytical Workflow

```text
30 Numeric Features
        |
        v
Data Validation
        |
        v
Correlation Matrix
        |
        v
Strongest Pair Verification
        |
        v
StandardScaler
        |
        v
PCA
        |
        +----> PC1
        |
        +----> PC2
        |
        v
2D PCA Projection
        |
        +----> Diagnostic Class View
        |
        +----> Interactive Feature Coloring
        |
        v
Feature Loadings
        |
        v
Interpretation
```

An optional extension applies **K-Means clustering** to the standardized feature space and compares the resulting clusters with the known diagnostic classes.

---

## Key Results

### Correlation structure

The strongest off-diagonal correlation is between:

- **mean radius**
- **mean perimeter**

with:

```text
r = 0.998
```

The verification scatterplot confirms an almost perfectly linear positive relationship.

### PCA explained variance

| Component | Explained Variance |
|---|---:|
| PC1 | 44.3% |
| PC2 | 19.0% |
| PC1 + PC2 | 63.2% |

The first two principal components preserve approximately **63.2% of the standardized variance** in two dimensions.

### PCA interpretation

**PC1** is driven strongly by features related to:

- concavity
- concave points
- compactness
- perimeter
- radius
- area

It can be interpreted as a broad **tumor size and morphological-irregularity dimension**.

**PC2** is influenced positively by fractal-dimension and complexity-related features, while size-related variables such as radius, area, and perimeter contribute in the opposite direction.

It can be interpreted as a **complexity-versus-size dimension**.

---

## Optional Clustering Extension

The project also includes an unsupervised K-Means comparison using two clusters.

### Clustering results

| Metric | Result |
|---|---:|
| Adjusted Rand Index | 0.671 |
| Silhouette Score | 0.345 |
| Aligned class-cluster agreement | 91.0% |

Aligned confusion matrix:

```text
                     Cluster -> Malignant   Cluster -> Benign
Actual Malignant              175                  37
Actual Benign                  14                 343
```

The aligned agreement is approximately:

- **82.5%** for malignant cases
- **96.1%** for benign cases

This is reported as **class-cluster agreement**, not supervised classification accuracy, because K-Means does not use diagnosis labels during clustering.

---

## Interactive Streamlit Application

The Streamlit dashboard includes:

- dataset overview metrics
- class-balance summary
- correlation heatmap
- strongest-correlation verification scatterplot
- PCA explained-variance metrics
- interactive PCA projection
- color-by selector for diagnosis or any numeric feature
- PC1 and PC2 loading visualizations
- component interpretation
- optional K-Means clustering
- aligned class-cluster confusion matrix

---

## Project Structure

```text
multivariate-pca-explorer/
|
|-- app.py
|-- pca_analysis.ipynb
|-- requirements.txt
|-- README.md
|-- .gitignore
```

### File descriptions

**`app.py`**  
Interactive Streamlit application containing the final exploratory visualization workflow.

**`pca_analysis.ipynb`**  
Notebook containing the analytical workflow, correlation analysis, PCA, loadings, interpretation, and clustering extension.

**`requirements.txt`**  
Python package dependencies required to reproduce the project.

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd multivariate-pca-explorer
```

### 2. Create a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Streamlit App

```bash
streamlit run app.py
```

Then open the local address shown by Streamlit, typically:

```text
http://localhost:8501
```

---

## Run the Notebook

Start Jupyter:

```bash
jupyter notebook
```

or open:

```text
pca_analysis.ipynb
```

directly in VS Code with the Jupyter extension.

---

## Technology Stack

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- SciPy
- Jupyter

---

## Methods Used

### Correlation Analysis

A Pearson correlation matrix is calculated across all 30 numeric diagnostic features. The strongest off-diagonal feature pair is identified programmatically and verified with a scatterplot.

### Standardization

All numeric predictors are standardized with:

```python
StandardScaler()
```

This prevents features with larger numeric scales from dominating the PCA solution.

### Principal Component Analysis

PCA is fitted on the standardized feature matrix:

```python
PCA(n_components=2)
```

The first two components are then used to visualize the observations in a two-dimensional space.

### PCA Loadings

Feature loadings are extracted from:

```python
pca.components_
```

The largest absolute loadings are used to interpret what PC1 and PC2 represent.

### K-Means Extension

K-Means clustering is applied to the standardized feature space:

```python
KMeans(
    n_clusters=2,
    random_state=42,
    n_init=20
)
```

Because cluster IDs are arbitrary, the Hungarian assignment algorithm is used to align cluster IDs with the known diagnostic classes before producing the comparison matrix.

---

## Interpretation Notes

PCA is an **unsupervised dimensionality-reduction method**. The diagnostic labels are used only for visualization and evaluation; they are not used when fitting the principal components.

Likewise, K-Means is an **unsupervised clustering algorithm**. The diagnostic labels are not used to construct the clusters.

The two-dimensional PCA projection preserves substantial structure, but it does not retain all information from the original 30-dimensional feature space.

---

## Future Improvements

Potential extensions include:

- cumulative explained-variance and scree plots
- additional principal components
- interactive Plotly or Altair PCA charts
- PCA biplots
- comparison with UMAP or t-SNE
- hierarchical clustering
- alternative clustering algorithms
- model-based classification as a separate supervised-learning extension

---

## Author

**Nazish Atta**

Data Science | Machine Learning | Data Visualization | Analytics Engineering
