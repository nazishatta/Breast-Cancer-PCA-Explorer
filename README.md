# Breast Cancer PCA Explorer

An interactive dimensionality-reduction and multivariate-analysis project using **PCA**, **correlation analysis**, and **clustering** on the Wisconsin Diagnostic Breast Cancer dataset.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://breast-cancer-pca-explorer.streamlit.app/)

**Live App:** [breast-cancer-pca-explorer.streamlit.app](https://breast-cancer-pca-explorer.streamlit.app/)

---

## Project Overview

High-dimensional datasets often contain strong correlation and redundant information that make direct visualization difficult. This project uses **Principal Component Analysis (PCA)** to transform a 30-feature diagnostic dataset into a lower-dimensional representation that can be inspected visually and evaluated through explained variance.

The workflow combines:

- data validation and class-balance inspection
- correlation analysis
- strongest-pair verification
- feature standardization
- PCA dimensionality reduction
- explained-variance analysis
- scree and cumulative explained-variance visualization
- PCA class visualization
- principal-component loadings
- interactive feature coloring
- optional unsupervised clustering
- aligned class-cluster comparison

The Streamlit application presents the analysis as an interactive exploratory dashboard, while the notebook provides the underlying analytical workflow. A rendered HTML version of the notebook is also included for browser-based viewing.

---

## Dataset

**Wisconsin Diagnostic Breast Cancer Dataset**

The dataset is loaded directly from `scikit-learn` using:

```python
from sklearn.datasets import load_breast_cancer
```

### Dataset Profile

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
        +----> PC1 + PC2 Projection
        |
        +----> Scree Plot
        |
        +----> Cumulative Explained Variance
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

### Correlation Structure

The strongest off-diagonal correlation is between:

- **mean radius**
- **mean perimeter**

with:

```text
r = 0.998
```

The verification scatterplot confirms an almost perfectly linear positive relationship.

### PCA Explained Variance

| Component Set | Explained Variance |
|---|---:|
| PC1 | 44.3% |
| PC2 | 19.0% |
| PC1 + PC2 | 63.2% |
| First 5 PCs | 84.7% |
| First 7 PCs | 91.0% |
| First 10 PCs | 95.2% |

The first two principal components preserve approximately **63.2% of the standardized variance** in two dimensions.

The scree and cumulative explained-variance analysis shows that variance is concentrated strongly in the first few components. The first **7 principal components retain approximately 91.0%** of the standardized variance, indicating substantial redundancy in the original 30-feature space.

### PCA Interpretation

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

## Scree Plot and Cumulative Explained Variance

A full PCA model is fitted to the standardized feature matrix to examine how much variance is retained as additional principal components are included.

The scree plot displays:

- **individual explained variance** for each principal component
- **cumulative explained variance** as components are added

The contribution of individual components decreases rapidly after the first few components, showing diminishing gains from adding additional dimensions.

Key cumulative results are:

```text
First 5 components:   84.7%
First 7 components:   91.0%
First 10 components:  95.2%
```

This confirms that the original 30 diagnostic measurements contain substantial redundancy. Although two components provide a useful visual representation, additional components are required to preserve most of the original standardized variance.

---

## Optional Clustering Extension

The project also includes an unsupervised K-Means comparison using two clusters.

### Clustering Results

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
- scree plot
- cumulative explained-variance curve
- interactive PCA projection
- color-by selector for diagnosis or any numeric feature
- PC1 and PC2 loading visualizations
- component interpretation
- optional K-Means clustering
- aligned class-cluster confusion matrix

**Launch the live application:**  
[https://breast-cancer-pca-explorer.streamlit.app/](https://breast-cancer-pca-explorer.streamlit.app/)

---

## Project Structure

```text
Breast-Cancer-PCA-Explorer/
|
|-- app.py
|-- pca_analysis.ipynb
|-- pca_analysis.html
|-- requirements.txt
|-- README.md
```

### File Descriptions

**`app.py`**  
Interactive Streamlit application containing the final exploratory visualization workflow.

**`pca_analysis.ipynb`**  
Jupyter notebook containing data validation, correlation analysis, PCA, scree analysis, PCA loadings, interpretation, and the clustering extension.

**`pca_analysis.html`**  
Rendered HTML export of the notebook for browser-based viewing without requiring Jupyter.

**`requirements.txt`**  
Python dependencies required to reproduce and run the project.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/nazishatta/Breast-Cancer-PCA-Explorer.git
cd Breast-Cancer-PCA-Explorer
```

### 2. Create a Virtual Environment

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

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

## Run the Streamlit App

```bash
python -m streamlit run app.py
```

Streamlit will display a local development address, typically:

```text
http://localhost:8501
```

The deployed application is available at:

```text
https://breast-cancer-pca-explorer.streamlit.app/
```

---

## Run the Notebook

Open:

```text
pca_analysis.ipynb
```

directly in VS Code with the Jupyter extension, or start Jupyter with:

```bash
jupyter notebook
```

A rendered browser-viewable version is also available as:

```text
pca_analysis.html
```

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

All numeric predictors are standardized using:

```python
StandardScaler()
```

Standardization prevents variables with larger numeric scales from dominating the PCA solution.

### Principal Component Analysis

A two-component PCA model is fitted on the standardized feature matrix for visualization:

```python
PCA(n_components=2)
```

The first two components are used to represent the 30-dimensional observations in a two-dimensional PCA space.

### Scree and Cumulative Explained Variance

A separate full PCA model is fitted using:

```python
PCA()
```

The explained-variance ratio of every principal component is calculated, along with cumulative explained variance.

The analysis shows:

```text
PC1 + PC2:           63.2%
First 5 components: 84.7%
First 7 components: 91.0%
First 10 components: 95.2%
```

The scree curve demonstrates diminishing variance contributions from later components, while the cumulative curve shows how rapidly the original feature space can be compressed.

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

PCA is an **unsupervised dimensionality-reduction method**. The diagnostic labels are used only for visualization and interpretation; they are not used when fitting the principal components.

Likewise, K-Means is an **unsupervised clustering algorithm**. The diagnostic labels are not used to construct the clusters.

The two-dimensional PCA projection preserves substantial diagnostic structure but retains approximately **63.2%** of the total standardized variance rather than all information in the original 30-dimensional feature space.

The scree analysis further shows that approximately **7 components are required to retain 91.0%** of the standardized variance.

---

## Future Improvements

Potential extensions include:

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
