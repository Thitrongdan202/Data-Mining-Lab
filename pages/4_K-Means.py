import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.title("K-Means Clustering")

# --- Dữ liệu ---
st.header("1) Dữ liệu")
use_sample = st.radio("Chọn dữ liệu", ["Dataset mẫu (Iris)", "Upload CSV"])
if use_sample.startswith("Dataset mẫu"):
    iris = load_iris(as_frame=True)
    df = iris.data
else:
    file = st.file_uploader("Upload CSV", type="csv")
    df = pd.read_csv(file) if file else pd.DataFrame()

if df.empty:
    st.warning("Dữ liệu trống.")
    st.stop()

num_cols = df.select_dtypes(include="number").columns.tolist()
st.dataframe(df, use_container_width=True)

# --- Tham số ---
st.header("2) Tham số")
cols = st.multiselect("Chọn cột số dùng để phân cụm", num_cols, default=num_cols)
k = st.slider("Số cụm k", 2, 10, 3)
rs = st.number_input("Random state", 0, 9999, 0)
run_elbow = st.checkbox("Chạy Elbow (k=2..10)")

if not cols:
    st.warning("Không có cột số được chọn.")
    st.stop()

X = df[cols].dropna()
if X.empty:
    st.warning("Dữ liệu chứa NA hoặc trống.")
    st.stop()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Elbow Method ---
if run_elbow:
    inertias = []
    ks = range(2, 11)
    for i in ks:
        km = KMeans(n_clusters=i, random_state=rs, n_init="auto").fit(X_scaled)
        inertias.append(km.inertia_)
    fig, ax = plt.subplots()
    ax.plot(list(ks), inertias, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    st.pyplot(fig)

# --- Huấn luyện KMeans ---
km = KMeans(n_clusters=k, random_state=rs, n_init="auto")
labels = km.fit_predict(X_scaled)
score = silhouette_score(X_scaled, labels)
st.write(f"Silhouette score: {score:.3f}")

# PCA 2D để vẽ
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
fig, ax = plt.subplots()
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
st.pyplot(fig)

cluster_counts = pd.Series(labels).value_counts().sort_index()
st.dataframe(cluster_counts.rename("count"), use_container_width=True)