import pandas as pd
import streamlit as st
from utils.rs_reduct import greedy_reduct, positive_region, discernibility_matrix

st.title("Thuật toán Reduct (Tập thô)")

# --- Dữ liệu ---
st.header("1) Dữ liệu")
use_sample = st.radio("Chọn dữ liệu", ["Dataset mẫu", "Upload CSV"])
if use_sample == "Dataset mẫu":
    df = pd.read_csv("data/employees_roughset.csv")
else:
    file = st.file_uploader("Upload CSV", type="csv")
    df = pd.read_csv(file) if file else pd.DataFrame()

if df.empty:
    st.warning("Dữ liệu trống.")
    st.stop()

st.dataframe(df, use_container_width=True)

# --- Tham số ---
st.header("2) Tham số")
decision = st.selectbox("Decision attribute", df.columns, index=len(df.columns)-1)
cond_attrs = [c for c in df.columns if c != decision]

# --- Tính reduct ---
try:
    reduct = greedy_reduct(df, cond_attrs, decision)
    disc = discernibility_matrix(df, cond_attrs, decision)
    pos = positive_region(df, reduct, decision) if reduct else 0.0
except Exception as e:
    st.warning(f"Lỗi khi tính reduct: {e}")
    st.stop()

st.header("3) Kết quả")
st.write(f"Reduct: {reduct}")
st.write(f"Số cặp cần phân biệt: {len(disc)}")
st.write(f"Positive Region: {pos:.3f}")

# Sinh luật đơn giản
if reduct:
    rules = []
    grouped = df.groupby(reduct)
    for key, sub in grouped:
        if sub[decision].nunique() == 1:
            vals = key if isinstance(key, tuple) else (key,)
            conds = [f"{a}={v}" for a, v in zip(reduct, vals)]
            rules.append(f"IF {' AND '.join(conds)} THEN {decision}={sub[decision].iloc[0]}")
    if rules:
        st.subheader("Luật")
        for r in rules:
            st.write(r)
else:
    st.warning("Không tìm được reduct. Dữ liệu phải rời rạc.")