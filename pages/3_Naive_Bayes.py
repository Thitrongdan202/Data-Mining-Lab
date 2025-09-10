import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.title("Naïve Bayes")

# --- Dữ liệu ---
st.header("1) Dữ liệu")
use_sample = st.radio("Chọn dữ liệu", ["Dataset mẫu (Play Tennis)", "Upload CSV"])
if use_sample.startswith("Dataset mẫu"):
    df = pd.read_csv("data/play_tennis.csv")
else:
    file = st.file_uploader("Upload CSV", type="csv")
    df = pd.read_csv(file) if file else pd.DataFrame()

if df.empty:
    st.warning("Dữ liệu trống.")
    st.stop()

st.dataframe(df, use_container_width=True)

# --- Tham số ---
st.header("2) Tham số")
target = st.selectbox("Chọn cột mục tiêu", df.columns, index=len(df.columns)-1)
model_type = st.selectbox("Thuật toán", ["Auto", "GaussianNB", "MultinomialNB", "BernoulliNB"])
test_size = st.slider("Test size", 0.1, 0.9, 0.3, 0.05)
rs = st.number_input("Random state", 0, 9999, 0)

X = df.drop(columns=[target])
y = df[target]
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols),
])

# Chọn mô hình NB
def choose_model():
    if model_type == "Auto":
        return GaussianNB() if num_cols else BernoulliNB()
    if model_type == "GaussianNB":
        return GaussianNB()
    if model_type == "MultinomialNB":
        return MultinomialNB()
    return BernoulliNB()

model = choose_model()
pipe = Pipeline([("preprocess", preprocess), ("clf", model)])

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=rs
    )
    pipe.fit(X_train, y_train)
except Exception as e:
    st.warning(f"Lỗi huấn luyện: {e}")
    st.stop()

# Kiểm tra giá trị âm cho MultinomialNB
if isinstance(model, MultinomialNB):
    transformed = pipe.named_steps["preprocess"].transform(X)
    if (transformed < 0).any():
        st.warning("Dữ liệu có giá trị âm, không phù hợp MultinomialNB.")

# --- Kết quả ---
st.header("3) Kết quả")
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {acc:.3f}")

report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).T, use_container_width=True)

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
ax.matshow(cm)
for (i, j), val in pd.DataFrame(cm).stack().items():
    ax.text(j, i, val, ha="center", va="center")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
st.pyplot(fig)