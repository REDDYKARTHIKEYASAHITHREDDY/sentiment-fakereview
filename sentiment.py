import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from lime.lime_text import LimeTextExplainer


# ================= PAGE CONFIG =================
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")


# ================= ORIGINAL UI (UNCHANGED) =================
st.markdown("""
<style>
body, .stApp {
    background: radial-gradient(circle at 20% 20%, #0a0f2d 0%, #020409 100%) !important;
    color: #e0e6ff !important;
    font-family: 'Segoe UI', sans-serif;
}

.block-container {
    background: rgba(20, 25, 50, 0.55);
    border: 1px solid rgba(90, 120, 255, 0.25);
    border-radius: 12px;
    padding: 18px;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 18px rgba(64, 105, 255, 0.18);
}

.stSidebar {
    background: rgba(15,15,40,0.6) !important;
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(80,110,255,0.2);
    box-shadow: 4px 0 12px rgba(50,70,255,0.15);
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Sentiment Analyzer")


# ================= MODEL =================
@st.cache_resource
def load_model():
    texts = [
        "I love this amazing product! It's fantastic and wonderful!",
        "This is absolutely terrible and the worst experience ever.",
        "It's okay, nothing too special or exciting.",
        "The service was decent but could definitely be much better.",
        "I'm extremely disappointed and frustrated right now.",
        "Today was such a great and beautiful day!",
        "Bad quality, poor service, very unhappy with purchase.",
        "Excellent quality, outstanding service, highly recommend this!",
        "Average product, met expectations but nothing more.",
        "Hate this completely, total waste of money and time."
    ]

    labels = [1, 0, 2, 2, 0, 1, 0, 1, 2, 0]

    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
        ('clf', LogisticRegression(random_state=42))
    ])

    model.fit(texts, labels)
    return model


model = load_model()


# ================= SIDEBAR =================
examples = [
    "I love this product!",
    "This is the worst experience ever.",
    "It's okay, nothing special.",
    "The service was fine but could be better.",
    "I'm so disappointed right now.",
    "Today was a good day."
]

choice = st.sidebar.radio("Try an example:", options=examples)

if "text_input" not in st.session_state:
    st.session_state.text_input = choice

if st.sidebar.button("Use this"):
    st.session_state.text_input = choice


# ================= HELPERS =================
def get_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return "Neutral", 0.0, "neutral"

    proba = model.predict_proba([text])[0]
    label = model.predict([text])[0]

    names = ["Negative", "Positive", "Neutral"]
    tone = ["negative", "positive", "neutral"][label]

    return names[label], max(proba), tone


def feature_importance(text):
    vec = model.named_steps['tfidf']
    clf = model.named_steps['clf']

    text_vec = vec.transform([text])
    names = vec.get_feature_names_out()

    pred = clf.predict(text_vec)[0]
    coefs = clf.coef_[pred]

    idxs = text_vec.indices

    words, weights = [], []

    for i in idxs:
        weight = coefs[i] * text_vec.data[np.where(text_vec.indices == i)[0][0]]
        if abs(weight) > 0.01:
            words.append(names[i])
            weights.append(weight)

    if weights:
        return sorted(zip(words, weights), key=lambda x: abs(x[1]), reverse=True)[:10]

    return []


def lime_explain(text):
    explainer = LimeTextExplainer(class_names=['Negative', 'Positive', 'Neutral'])
    return explainer.explain_instance(text, model.predict_proba, num_features=10)


# ================= SHAP =================
@st.cache_resource
def load_shap_explainer():
    masker = shap.maskers.Text()

    def predict_fn(texts):
        return model.predict_proba(texts)

    return shap.Explainer(predict_fn, masker)


shap_explainer = load_shap_explainer()


# ================= MAIN INPUT =================
st.subheader("Try It Out With Manual Text")
text_input = st.text_area("Write something here to analyze:",
                          value=st.session_state.get("text_input", ""))


# ================= RESULTS =================
if text_input:
    mood, conf, tone = get_sentiment(text_input)

    st.markdown("### Sentiment Result")
    st.write(f"**Mood:** {mood}")
    st.write(f"**Confidence:** {conf:.3f}")
    st.write(f"**Type:** {tone.capitalize()}")

    c1, c2, c3 = st.columns(3)

    # ----- Feature Importance -----
    with c1:
        st.markdown("#### Feature Importance")
        features = feature_importance(text_input)

        if features:
            words, weights = zip(*features)

            fig, ax = plt.subplots()
            colors = ['red' if w < 0 else 'green' for w in weights]
            ax.barh(range(len(words)), weights, color=colors)
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words)
            ax.set_xlabel("Weight")
            ax.set_title("Top Words")

            st.pyplot(fig, clear_figure=True)
        else:
            st.info("No strong features found.")

    # ----- LIME WHITE BOARD -----
    with c2:
        st.markdown("#### LIME Explanation")
        try:
            lime_exp = lime_explain(text_input)
            lime_html = lime_exp.as_html()

            white_box = f"""
            <div style="background:white;padding:15px;border-radius:10px;">
                {lime_html}
            </div>
            """

            st.components.v1.html(white_box, height=420, scrolling=True)
        except Exception as e:
            st.error(f"LIME Error: {e}")

    # ----- SHAP WHITE BOARD -----
    with c3:
        st.markdown("#### SHAP Explanation")
        try:
            shap_values = shap_explainer([text_input])
            shap_html = shap.plots.text(shap_values[0], display=False)

            white_box = f"""
            <div style="background:white;padding:15px;border-radius:10px;">
                {shap_html}
            </div>
            """

            st.components.v1.html(white_box, height=420, scrolling=True)
        except Exception as e:
            st.error(f"SHAP Error: {e}")


# ================= BULK ANALYSIS =================
st.subheader("📊 Bulk Sentiment Analysis")

file = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])


def read_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_json(file)


if file:
    df = read_file(file)

    if not df.empty:
        st.write(df.head())

        text_col = st.selectbox("Select text column", df.columns)

        if st.button("Run Analysis"):
            results = [get_sentiment(str(t)) for t in df[text_col]]

            df["Sentiment"], df["Confidence"], df["Class"] = zip(*results)

            st.success("Analysis complete!")

            summary = df["Class"].value_counts().reindex(
                ["negative", "neutral", "positive"]
            ).fillna(0)

            st.bar_chart(summary)
            st.dataframe(df.head(10))

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results", csv, "sentiment_results.csv", "text/csv")