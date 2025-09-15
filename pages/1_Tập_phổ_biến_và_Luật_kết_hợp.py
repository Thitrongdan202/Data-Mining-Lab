import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.title("Tập phổ biến & Luật kết hợp")

# --- Sidebar tham số ---
st.sidebar.header("Tham số")
min_sup = st.sidebar.slider("Min support", 0.01, 1.0, 0.2, 0.01)
min_conf = st.sidebar.slider("Min confidence", 0.01, 1.0, 0.6, 0.01)
num_rules = st.sidebar.number_input("Số luật hiển thị", 1, 100, 10)

st.header("1) Dữ liệu")
use_sample = st.radio("Chọn dữ liệu", ["Dataset mẫu", "Upload CSV"])
if use_sample == "Dataset mẫu":
    try:
        df_raw = pd.read_csv("data/market_basket.csv")
    except Exception as e:
        st.warning(f"Không đọc được file mẫu: {e}")
        df_raw = pd.DataFrame()
else:
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        df_raw = pd.read_csv(file)
    else:
        df_raw = pd.DataFrame()

if df_raw.empty:
    st.warning("Dữ liệu trống. Kiểm tra lại file CSV.")
    st.stop()

st.dataframe(df_raw, use_container_width=True)

# --- Tiền xử lý ---
if "items" in df_raw.columns:
    # tách chuỗi 'a,b,c' thành list
    transactions = (
        df_raw["items"].dropna().apply(lambda x: [i.strip() for i in str(x).split(",")])
    )
    te = TransactionEncoder()
    oht = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(oht, columns=te.columns_)
else:
    df = df_raw.astype(bool)

st.header("2) Tham số")
st.write(f"Min support: {min_sup}, Min confidence: {min_conf}")

# --- Apriori ---
try:
    freq = apriori(df, min_support=min_sup, use_colnames=True)
except Exception as e:
    st.warning(f"Lỗi khi chạy Apriori: {e}")
    st.stop()

if freq.empty:
    st.warning("Không có itemset thoả mãn. Hãy giảm min_support.")
    st.stop()

freq["length"] = freq["itemsets"].apply(len)
freq = freq.sort_values("support", ascending=False)

st.header("3) Kết quả")
st.subheader("Itemsets")
st.dataframe(freq, use_container_width=True)

rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
if rules.empty:
    st.warning("Không có luật thoả mãn. Giảm ngưỡng confidence.")
else:
    rules = rules.sort_values(["confidence", "lift"], ascending=False)
    st.subheader("Rules")
    st.dataframe(rules.head(num_rules), use_container_width=True)
    csv = rules.to_csv(index=False).encode("utf-8")
    st.download_button("Download rules.csv", data=csv, file_name="rules.csv")