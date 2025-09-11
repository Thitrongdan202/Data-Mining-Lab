import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

st.title("Cây quyết định (ID3)")

# --- Dữ liệu ---
st.header("1) Dữ liệu")
use_sample = st.radio("Chọn dữ liệu", ["Dataset mẫu", "Upload CSV"])
if use_sample == "Dataset mẫu":
    try:
        df = pd.read_csv("data/play_tennis.csv")
    except Exception as e:
        st.warning(f"Không đọc được file mẫu: {e}")
        df = pd.DataFrame()
else:
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        df = pd.read_csv(file)
    else:
        df = pd.DataFrame()

if df.empty:
    st.warning("Dữ liệu trống.")
    st.stop()

st.dataframe(df, use_container_width=True)

# --- Tham số ---
st.header("2) Tham số")
target = st.selectbox("Chọn cột mục tiêu", df.columns, index=len(df.columns)-1)
test_size = st.slider("Test size", 0.1, 0.9, 0.3, 0.05)
max_depth = st.number_input("Max depth (0 = None)", 0, 50, 0)
rs = st.number_input("Random state", 0, 9999, 0)

X = df.drop(columns=[target])
y = df[target]
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols),
])

clf = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth or None, random_state=rs)
pipe = Pipeline([("preprocess", preprocess), ("clf", clf)])

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=rs
    )
    pipe.fit(X_train, y_train)
except Exception as e:
    st.warning(f"Lỗi huấn luyện: {e}")
    st.stop()

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

# Vẽ cây quyết định
ohe = pipe.named_steps["preprocess"].named_transformers_.get("cat")
cat_names = ohe.get_feature_names_out(cat_cols) if cat_cols else []
feature_names = list(cat_names) + num_cols
fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(
    pipe.named_steps["clf"],
    feature_names=feature_names,
    class_names=pipe.named_steps["clf"].classes_,
    filled=False,
    ax=ax,
)
st.pyplot(fig)

st.subheader("Luật dạng văn bản")
text_rules = export_text(pipe.named_steps["clf"], feature_names=list(feature_names))
st.text(text_rules)