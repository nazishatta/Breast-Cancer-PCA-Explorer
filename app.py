"""
Multivariate PCA Explorer
Wisconsin Diagnostic Breast Cancer Dataset

Interactive analysis of correlation structure, principal components,
feature loadings, diagnostic classes, and an optional class-cluster comparison.

Run locally with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------
# Matplotlib dark chart theme
# ---------------------------------------------------------------------

plt.rcParams.update(
    {
        "figure.facecolor": "#1A1D24",
        "axes.facecolor": "#1A1D24",
        "axes.edgecolor": "#3B4250",
        "axes.labelcolor": "#E5E7EB",
        "axes.titlecolor": "#F8FAFC",
        "text.color": "#F3F4F6",
        "xtick.color": "#D1D5DB",
        "ytick.color": "#D1D5DB",
        "grid.color": "#303642",
        "legend.facecolor": "#20242D",
        "legend.edgecolor": "#3B4250",
        "legend.labelcolor": "#F3F4F6",
        "savefig.facecolor": "#1A1D24",
    }
)



# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Multivariate PCA Explorer",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>

    /* =========================================================
       GLOBAL THEME
       ========================================================= */

    :root {
        --bg: #111318;
        --surface: #1A1D24;
        --surface-soft: #20242D;
        --border: #303642;

        --text: #F3F4F6;
        --text-muted: #A7B0BE;

        --heading: #F8FAFC;
        --primary: #5B8DEF;
        --primary-hover: #78A4F5;

        --success: #4FB3A5;
        --shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
    }


    /* =========================================================
       MAIN PAGE
       ========================================================= */

    .stApp {
        background-color: var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }


    /* =========================================================
       TYPOGRAPHY
       ========================================================= */

    h1 {
        color: var(--heading) !important;
        font-weight: 750 !important;
        letter-spacing: -0.03em;
    }

    h2 {
        color: var(--heading) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        margin-top: 1.8rem !important;
    }

    h3 {
        color: #DCE6F2 !important;
        font-weight: 650 !important;
    }

    p,
    li,
    label {
        color: var(--text);
    }

    .stCaption,
    [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
    }


    /* =========================================================
       METRIC CARDS
       ========================================================= */

    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: var(--shadow);
        min-height: 118px;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text-muted);
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: var(--heading);
        font-weight: 750;
    }


    /* =========================================================
       DATAFRAMES / TABLES
       ========================================================= */

    div[data-testid="stDataFrame"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--shadow);
    }


    /* =========================================================
       EXPANDERS
       ========================================================= */

    details {
        background: var(--surface);
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        box-shadow: var(--shadow);
    }

    details summary {
        color: var(--heading) !important;
        font-weight: 600 !important;
    }


    /* =========================================================
       SELECTBOX / INPUT CONTROLS
       ========================================================= */

    div[data-baseweb="select"] > div {
        background-color: var(--surface) !important;
        border-color: #3B4250 !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="select"] > div:hover {
        border-color: var(--primary) !important;
    }

    div[data-baseweb="input"] > div {
        background-color: var(--surface) !important;
        border-radius: 10px !important;
    }


    /* =========================================================
       BUTTONS
       ========================================================= */

    .stButton > button,
    .stDownloadButton > button {
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: 9px;
        padding: 0.55rem 1rem;
        font-weight: 600;
        transition: 0.2s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background-color: var(--primary-hover);
        color: white;
        border: none;
    }


    /* =========================================================
       ALERTS / INFO BOXES
       ========================================================= */

    div[data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid var(--border);
    }


    /* =========================================================
       DIVIDERS
       ========================================================= */

    hr {
        border: none;
        border-top: 1px solid var(--border);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }


    /* =========================================================
       LINKS
       ========================================================= */

    a {
        color: var(--primary) !important;
        text-decoration: none;
    }

    a:hover {
        color: var(--primary-hover) !important;
    }


    /* =========================================================
       PLOT / ELEMENT SPACING
       ========================================================= */

    div[data-testid="stImage"] {
        background: var(--surface);
        border-radius: 12px;
    }

    div[data-testid="stVerticalBlock"] > div {
        gap: 0.8rem;
    }


    /* =========================================================
       TOP STREAMLIT HEADER
       ========================================================= */

    header[data-testid="stHeader"] {
        background: rgba(17, 19, 24, 0.94);
        backdrop-filter: blur(8px);
    }


    /* =========================================================
       OPTIONAL SIDEBAR
       ========================================================= */

    section[data-testid="stSidebar"] {
        background-color: #16191F;
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--heading) !important;
    }


    /* DARK THEME FINISHING */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: var(--text) !important;
    }

    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }

    div[data-testid="stMetricLabel"] p {
        color: var(--text-muted) !important;
    }

    div[data-baseweb="select"] span {
        color: var(--text) !important;
    }

    div[data-baseweb="popover"] {
        background-color: var(--surface-soft) !important;
    }

    div[role="listbox"] {
        background-color: var(--surface-soft) !important;
        color: var(--text) !important;
    }

    div[role="option"] {
        color: var(--text) !important;
    }

    div[role="option"]:hover {
        background-color: #2A303B !important;
    }

    div[data-testid="stExpander"] {
        background: var(--surface);
        border-radius: 12px;
    }

    code {
        color: #B9D1FF !important;
        background-color: #20242D !important;
        border-radius: 4px;
        padding: 0.1rem 0.25rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.title("Multivariate PCA Explorer")
st.caption("Wisconsin Diagnostic Breast Cancer Dataset")

st.markdown(
    """
This interactive analysis explores whether **30 diagnostic measurements**
can be summarized in two principal components while preserving meaningful
structure between **malignant** and **benign** cases.
"""
)




# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

@st.cache_data
def load_data():
    cancer = load_breast_cancer(as_frame=True)

    features = cancer.data.copy()

    diagnosis = cancer.target.map(
        {
            0: "Malignant",
            1: "Benign",
        }
    )

    df = features.copy()
    df["diagnosis"] = diagnosis

    return cancer, features, diagnosis, df


cancer, features, diagnosis, df = load_data()
feature_names = list(features.columns)


# ---------------------------------------------------------------------
# Dataset overview
# ---------------------------------------------------------------------

st.header("1. Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Observations", f"{len(df):,}")
c2.metric("Numeric Features", features.shape[1])
c3.metric("Classes", diagnosis.nunique())
c4.metric("Missing Values", int(features.isnull().sum().sum()))

st.markdown(
    """
The feature matrix contains only the **30 numeric predictors**.
The diagnosis label is kept separate so it is never included in the
correlation matrix, standardization step, or PCA fit.
"""
)

left, right = st.columns([2, 1])

with left:
    with st.expander("Preview data"):
        st.dataframe(df.head(10), width="stretch")

with right:
    class_balance = (
        df["diagnosis"]
        .value_counts()
        .rename_axis("diagnosis")
        .reset_index(name="count")
    )
    class_balance["share"] = (
        class_balance["count"] / class_balance["count"].sum()
    ).map("{:.1%}".format)

    st.subheader("Class Balance")
    st.dataframe(class_balance, hide_index=True, width="stretch")


# ---------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------

st.header("2. Correlation Structure")

corr = features.corr()

fig_corr, ax_corr = plt.subplots(figsize=(14, 11))

fig_corr.patch.set_facecolor("#1A1D24")

ax_corr.set_facecolor("#1A1D24")
sns.heatmap(
    corr,
    cmap="vlag",
    center=0,
    square=True,
    cbar_kws={"shrink": 0.65},
    ax=ax_corr,
)
ax_corr.set_title("Correlation Among Diagnostic Features")
fig_corr.tight_layout()
st.pyplot(fig_corr)
plt.close(fig_corr)

corr_abs = corr.abs()

upper_triangle = corr_abs.where(
    np.triu(np.ones(corr_abs.shape), k=1).astype(bool)
)

strongest_pair = upper_triangle.stack().idxmax()
strongest_r = corr.loc[strongest_pair[0], strongest_pair[1]]

st.markdown(
    f"""
**Strongest off-diagonal correlation:**  
`{strongest_pair[0]}` and `{strongest_pair[1]}` with **r = {strongest_r:.3f}**.
"""
)

x_var, y_var = strongest_pair

fig_pair, ax_pair = plt.subplots(figsize=(8, 5.5))

fig_pair.patch.set_facecolor("#1A1D24")

ax_pair.set_facecolor("#1A1D24")

for label, group in df.groupby("diagnosis"):
    ax_pair.scatter(
        group[x_var],
        group[y_var],
        alpha=0.7,
        label=label,
    )

ax_pair.set_xlabel(x_var)
ax_pair.set_ylabel(y_var)
ax_pair.set_title(
    f"{x_var} vs {y_var} (r = {strongest_r:.2f})"
)
ax_pair.legend(title="Diagnosis", frameon=True)
fig_pair.tight_layout()
st.pyplot(fig_pair)
plt.close(fig_pair)

st.markdown(
    f"""
The heatmap reveals substantial redundancy among several measurements,
especially features related to tumor size and geometry. The strongest
relationship is between **{x_var}** and **{y_var}** with **r = {strongest_r:.3f}**.
The verification scatterplot shows an almost perfectly linear positive
relationship rather than a correlation created by a small number of outliers.
Malignant cases also extend toward larger values of these measurements.
"""
)


# ---------------------------------------------------------------------
# Standardization and PCA
# ---------------------------------------------------------------------

st.header("3. Principal Component Analysis")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)

pc1_var = pca.explained_variance_ratio_[0]
pc2_var = pca.explained_variance_ratio_[1]
total_var = pc1_var + pc2_var

m1, m2, m3 = st.columns(3)
m1.metric("PC1 Explained Variance", f"{pc1_var:.1%}")
m2.metric("PC2 Explained Variance", f"{pc2_var:.1%}")
m3.metric("PC1 + PC2", f"{total_var:.1%}")

st.markdown(
    """
`StandardScaler` is applied before PCA because the original measurements
have very different numeric ranges. PCA is then fitted on the standardized
30-feature matrix.
"""
)

pca_df = pd.DataFrame(
    pcs,
    columns=["PC1", "PC2"],
    index=features.index,
)
pca_df["diagnosis"] = diagnosis.values

for col in feature_names:
    pca_df[col] = features[col].values


# ---------------------------------------------------------------------
# Interactive PCA projection
# ---------------------------------------------------------------------

st.header("4. Interactive PCA Projection")

color_col = st.selectbox(
    "Color points by",
    ["diagnosis"] + feature_names,
)

fig_pca, ax_pca = plt.subplots(figsize=(9, 6.5))

fig_pca.patch.set_facecolor("#1A1D24")

ax_pca.set_facecolor("#1A1D24")

if color_col == "diagnosis":
    for label, group in pca_df.groupby("diagnosis"):
        ax_pca.scatter(
            group["PC1"],
            group["PC2"],
            alpha=0.72,
            label=label,
        )

    ax_pca.legend(title="Diagnosis", frameon=True)

else:
    scatter = ax_pca.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        c=pca_df[color_col],
        alpha=0.72,
    )

    colorbar = fig_pca.colorbar(scatter, ax=ax_pca)
    colorbar.set_label(color_col)

ax_pca.set_xlabel(f"PC1 ({pc1_var:.1%} explained variance)")
ax_pca.set_ylabel(f"PC2 ({pc2_var:.1%} explained variance)")
ax_pca.set_title("PCA Projection of Diagnostic Measurements")

fig_pca.tight_layout()
st.pyplot(fig_pca)
plt.close(fig_pca)

st.markdown(
    f"""
The first two principal components retain approximately **{total_var:.1%}**
of the standardized variance. Benign observations are concentrated mainly
on the negative side of PC1, while malignant observations extend more
strongly toward positive PC1 values. The overlap near the center shows that
the two-dimensional representation preserves substantial diagnostic
structure, but the classes are not perfectly separated.
"""
)


# ---------------------------------------------------------------------
# Loadings
# ---------------------------------------------------------------------

st.header("5. What PC1 and PC2 Represent")

loadings = pd.DataFrame(
    pca.components_.T,
    index=feature_names,
    columns=["PC1", "PC2"],
)

top_pc1 = (
    loadings["PC1"]
    .abs()
    .sort_values(ascending=False)
    .head(10)
    .index
)

top_pc2 = (
    loadings["PC2"]
    .abs()
    .sort_values(ascending=False)
    .head(10)
    .index
)

col_pc1, col_pc2 = st.columns(2)

with col_pc1:
    pc1_sorted = loadings["PC1"].sort_values()

    fig_l1, ax_l1 = plt.subplots(figsize=(7.5, 8))

    fig_l1.patch.set_facecolor("#1A1D24")

    ax_l1.set_facecolor("#1A1D24")
    pc1_sorted.plot.barh(ax=ax_l1)
    ax_l1.axvline(0, linewidth=0.8)
    ax_l1.set_xlabel("Loading")
    ax_l1.set_title("PC1 Feature Loadings")
    fig_l1.tight_layout()

    st.pyplot(fig_l1)
    plt.close(fig_l1)

with col_pc2:
    pc2_sorted = loadings["PC2"].sort_values()

    fig_l2, ax_l2 = plt.subplots(figsize=(7.5, 8))

    fig_l2.patch.set_facecolor("#1A1D24")

    ax_l2.set_facecolor("#1A1D24")
    pc2_sorted.plot.barh(ax=ax_l2)
    ax_l2.axvline(0, linewidth=0.8)
    ax_l2.set_xlabel("Loading")
    ax_l2.set_title("PC2 Feature Loadings")
    fig_l2.tight_layout()

    st.pyplot(fig_l2)
    plt.close(fig_l2)

st.subheader("Top Contributors")

t1, t2 = st.columns(2)

with t1:
    st.markdown("**PC1**")
    st.dataframe(
        loadings.loc[top_pc1, ["PC1"]]
        .assign(abs_loading=lambda x: x["PC1"].abs())
        .sort_values("abs_loading", ascending=False),
        width="stretch",
    )

with t2:
    st.markdown("**PC2**")
    st.dataframe(
        loadings.loc[top_pc2, ["PC2"]]
        .assign(abs_loading=lambda x: x["PC2"].abs())
        .sort_values("abs_loading", ascending=False),
        width="stretch",
    )

st.markdown(
    """
**PC1 interpretation.**  
PC1 is driven strongly by concavity, concave points, compactness, perimeter,
radius, and area measurements. It therefore represents a broad
**tumor size and morphological-irregularity dimension**.

**PC2 interpretation.**  
PC2 places strong positive weight on fractal-dimension and several
complexity-related measurements, while radius, area, and perimeter variables
contribute in the opposite direction. It therefore captures a contrast between
**boundary complexity / fractal characteristics and overall tumor size**.

The sign of a PCA component is arbitrary; interpretation depends on the
relative loading pattern rather than whether a loading happens to be positive
or negative in one run.
"""
)


# ---------------------------------------------------------------------
# Optional clustering extension
# ---------------------------------------------------------------------

st.header("6. Class–Cluster Comparison")
st.caption("Optional extension: clustering is separate from PCA.")

with st.expander("Show K-Means clustering and aligned confusion matrix"):
    kmeans = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=20,
    )

    cluster_labels = kmeans.fit_predict(X_scaled)

    ari = adjusted_rand_score(cancer.target, cluster_labels)
    silhouette = silhouette_score(X_scaled, cluster_labels)

    k1, k2 = st.columns(2)
    k1.metric("Adjusted Rand Index", f"{ari:.3f}")
    k2.metric("Silhouette Score", f"{silhouette:.3f}")

    cluster_plot_df = pca_df[["PC1", "PC2"]].copy()
    cluster_plot_df["cluster"] = cluster_labels.astype(str)

    fig_cluster, ax_cluster = plt.subplots(figsize=(9, 6.5))

    fig_cluster.patch.set_facecolor("#1A1D24")

    ax_cluster.set_facecolor("#1A1D24")

    for label, group in cluster_plot_df.groupby("cluster"):
        ax_cluster.scatter(
            group["PC1"],
            group["PC2"],
            alpha=0.72,
            label=f"Cluster {label}",
        )

    ax_cluster.set_xlabel(f"PC1 ({pc1_var:.1%} explained variance)")
    ax_cluster.set_ylabel(f"PC2 ({pc2_var:.1%} explained variance)")
    ax_cluster.set_title("K-Means Clusters in PCA Space")
    ax_cluster.legend(frameon=True)

    fig_cluster.tight_layout()
    st.pyplot(fig_cluster)
    plt.close(fig_cluster)

    # Align arbitrary cluster IDs to the known classes.
    raw_cm = confusion_matrix(cancer.target, cluster_labels)
    row_ind, col_ind = linear_sum_assignment(-raw_cm)

    cluster_to_class = {
        cluster_id: class_id
        for class_id, cluster_id in zip(row_ind, col_ind)
    }

    aligned_clusters = np.array(
        [cluster_to_class[c] for c in cluster_labels]
    )

    aligned_cm = confusion_matrix(
        cancer.target,
        aligned_clusters,
    )

    class_names = list(cancer.target_names)

    cm_df = pd.DataFrame(
        aligned_cm,
        index=[f"Actual {name}" for name in class_names],
        columns=[f"Cluster mapped to {name}" for name in class_names],
    )

    fig_cm, ax_cm = plt.subplots(figsize=(6.5, 5))

    fig_cm.patch.set_facecolor("#1A1D24")

    ax_cm.set_facecolor("#1A1D24")
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cbar=False,
        cmap="mako",
        linewidths=1,
        linecolor="#303642",
        ax=ax_cm,
    )
    ax_cm.set_title("Aligned Class–Cluster Confusion Matrix")
    ax_cm.set_xlabel("Aligned Cluster")
    ax_cm.set_ylabel("True Diagnosis")

    fig_cm.tight_layout()
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    agreement = np.trace(aligned_cm) / aligned_cm.sum()
    malignant_agreement = aligned_cm[0, 0] / aligned_cm[0].sum()
    benign_agreement = aligned_cm[1, 1] / aligned_cm[1].sum()

    st.markdown(
        f"""
The discovered clusters show meaningful agreement with the known diagnostic
classes. The **ARI is {ari:.3f}**, while the **silhouette score is
{silhouette:.3f}**, indicating moderate internal cluster separation.

After aligning the arbitrary K-Means cluster IDs to the known classes,
**{np.trace(aligned_cm)} of {aligned_cm.sum()} observations
({agreement:.1%})** fall into the corresponding class-aligned cluster.
The aligned agreement is **{malignant_agreement:.1%} for malignant cases**
and **{benign_agreement:.1%} for benign cases**.

This should be interpreted as **class–cluster agreement**, not supervised
classification accuracy, because K-Means does not use the diagnosis labels
when creating the clusters.
"""
    )


# ---------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------

st.header("7. Summary")

st.markdown(
    f"""
The 30 diagnostic measurements contain substantial correlation and redundancy.
The strongest observed relationship is between **{x_var}** and **{y_var}**
(**r = {strongest_r:.3f}**).

After standardization, PC1 explains **{pc1_var:.1%}** of the variance and
PC2 explains **{pc2_var:.1%}**, retaining **{total_var:.1%}** in two
dimensions. The PCA projection reveals substantial but incomplete diagnostic
separation. PC1 primarily reflects **tumor size and morphological
irregularity**, while PC2 captures a contrasting **complexity-versus-size**
dimension.
"""
)
