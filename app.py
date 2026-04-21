import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from lime.lime_text import LimeTextExplainer
import shap

st.set_page_config(
    page_title="Review Intelligence Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0b0d17;
    color: #d8deff;
}

.block-container {
    padding: 1.5rem 2rem;
    max-width: 1300px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0e1020 !important;
    border-right: 1px solid #1e2240;
}

/* Headers */
h1 { font-family: 'Space Mono', monospace !important; color: #a5b4fc !important; letter-spacing: -1px; }
h2, h3 { font-family: 'Space Mono', monospace !important; color: #c7d2fe !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #12152a;
    border: 1px solid #2a2e55;
    border-radius: 10px;
    padding: 12px 18px;
}
[data-testid="stMetricValue"] { color: #a5b4fc !important; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"] { color: #7c85c7 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    padding: 0.5rem 1.2rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Tabs */
[data-testid="stTab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #7c85c7;
}

/* Text area & inputs */
textarea, .stTextInput input, .stSelectbox > div {
    background: #12152a !important;
    color: #d8deff !important;
    border: 1px solid #2a2e55 !important;
    border-radius: 8px !important;
}

/* Progress bar */
.stProgress > div > div { background: linear-gradient(90deg, #4f46e5, #7c3aed); }

/* Info / success / error boxes */
.stAlert { border-radius: 8px; }

/* Divider */
hr { border-color: #1e2240; }

.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.badge-pos { background: #14532d; color: #86efac; }
.badge-neg { background: #450a0a; color: #fca5a5; }
.badge-neu { background: #1c1f40; color: #a5b4fc; }
.badge-low  { background: #14532d; color: #86efac; }
.badge-med  { background: #451a03; color: #fdba74; }
.badge-high { background: #450a0a; color: #fca5a5; }

.card {
    background: #12152a;
    border: 1px solid #1e2240;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## Sentiment and Fake Review Analyser")

@st.cache_resource(show_spinner="Loading sentiment model…")
def load_sentiment_model():
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
        "Hate this completely, total waste of money and time.",
        "Not bad, works as expected.",
        "Pretty good overall, minor issues.",
    ]
    labels = [1, 0, 2, 2, 0, 1, 0, 1, 2, 0, 2, 1]

    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
        ('clf', LogisticRegression(random_state=42, max_iter=500))
    ])
    model.fit(texts, labels)
    return model


@st.cache_resource(show_spinner="Loading SHAP explainer…")
def load_shap_explainer(_model):
    masker = shap.maskers.Text()
    return shap.Explainer(lambda texts: _model.predict_proba(texts), masker)


# Load models at startup
sentiment_model = load_sentiment_model()
shap_explainer = load_shap_explainer(sentiment_model)

LABEL_NAMES = {0: "Negative", 1: "Positive", 2: "Neutral"}
TONE_NAMES  = {0: "negative", 1: "positive", 2: "neutral"}
GENERIC_PHRASES = [
    'best product ever', 'waste of money', 'highly recommend',
    "don't buy", 'perfect', 'terrible', 'amazing product',
    'worst ever', 'must buy', 'love it',
]


def get_sentiment(text: str):
    """Returns (label_name, confidence, tone_str, proba_array)."""
    if not isinstance(text, str) or not text.strip():
        return "Neutral", 0.0, "neutral", np.array([0.0, 0.0, 1.0])
    proba = sentiment_model.predict_proba([text])[0]
    label = int(sentiment_model.predict([text])[0])
    return LABEL_NAMES[label], float(max(proba)), TONE_NAMES[label], proba


def fake_score(text: str) -> tuple[int, list[str]]:
    """Heuristic fake-review score (0-100) with reasons."""
    score = 0
    reasons = []
    words = text.split()
    wc = len(words)
    exc = text.count('!')
    uc_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

    if wc < 10:
        score += 30; reasons.append("Very short review (< 10 words)")
    if exc > 3:
        score += 20; reasons.append(f"Excessive exclamation marks ({exc})")
    if uc_ratio > 0.30:
        score += 15; reasons.append(f"High uppercase ratio ({uc_ratio:.0%})")
    if any(p in text.lower() for p in GENERIC_PHRASES):
        score += 15; reasons.append("Contains generic/spammy phrases")

    # Extreme sentiment adds risk
    proba = sentiment_model.predict_proba([text])[0]
    if max(proba) > 0.97:
        score += 20; reasons.append("Extreme model confidence (potential polarised spam)")

    return min(score, 100), reasons


def risk_label(score: int) -> str:
    if score < 30: return "Low"
    if score < 60: return "Medium"
    return "High"


def feature_importance(text: str):
    vec = sentiment_model.named_steps['tfidf']
    clf = sentiment_model.named_steps['clf']
    tv = vec.transform([text])
    names = vec.get_feature_names_out()
    pred = int(clf.predict(tv)[0])
    coefs = clf.coef_[pred]
    idxs = tv.indices
    pairs = []
    for i in idxs:
        w = coefs[i] * tv.data[np.where(tv.indices == i)[0][0]]
        if abs(w) > 0.01:
            pairs.append((names[i], float(w)))
    return sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:10]


def lime_explain(text: str):
    explainer = LimeTextExplainer(class_names=['Negative', 'Positive', 'Neutral'])
    return explainer.explain_instance(
        text, sentiment_model.predict_proba, num_features=10, num_samples=300
    )


def badge_sentiment(tone: str) -> str:
    cls = {"positive": "badge-pos", "negative": "badge-neg", "neutral": "badge-neu"}.get(tone, "badge-neu")
    return f'<span class="badge {cls}">{tone.upper()}</span>'


def badge_risk(r: str) -> str:
    cls = {"Low": "badge-low", "Medium": "badge-med", "High": "badge-high"}.get(r, "badge-neu")
    return f'<span class="badge {cls}">{r.upper()} RISK</span>'


EXAMPLES = {
    "😍 Glowing (Suspicious)": "AMAZING PRODUCT!!! BEST EVER!!! BUY NOW!!! 10/10 PERFECT!!!",
    "💬 Authentic positive":   "Been using this for 3 months. Build quality is solid and battery lasts all day. The app could use polish but it works.",
    "😠 Angry (Suspicious)":   "Terrible waste of money don't buy",
    "💬 Authentic negative":   "Sound is decent for the price but the ear cushions deteriorated after 4 months. Customer service took two weeks to respond.",
    "😐 Neutral":              "It's okay, nothing too special or exciting. Meets basic expectations.",
    "⚠️ Borderline":           "Good product highly recommend to everyone! Amazing quality great value!",
}

with st.sidebar:
    st.markdown("### 📋 Example Reviews")
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    for label, review in EXAMPLES.items():
        if st.button(label, use_container_width=True):
            st.session_state.input_text = review

    st.markdown("---")
    st.markdown("### ℹ️ Risk Indicators")
    st.markdown("""
<div class="card" style="font-size:0.82rem;color:#9ba3d4;">
<b style="color:#c7d2fe;">High risk signals:</b><br>
• Very short (< 10 words)<br>
• Excessive ! marks<br>
• High uppercase ratio<br>
• Generic spam phrases<br>
• Extreme model confidence<br><br>
<b style="color:#c7d2fe;">Authentic signals:</b><br>
• Specific product details<br>
• Mixed positive/negative<br>
• Reasonable length<br>
• Natural language tone
</div>
""", unsafe_allow_html=True)
    st.markdown("---")
    show_explainability = st.checkbox("Show Explainability (LIME + SHAP)", value=True)
    st.caption("Uncheck to speed up single analysis")

tab1, tab2 = st.tabs(["🔍 Single Review", "📊 Bulk Analysis"])

with tab1:
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown("#### Enter Review")
        text_input = st.text_area(
            "Review text:",
            value=st.session_state.get("input_text", ""),
            height=160,
            placeholder="Paste or type a review here…",
            label_visibility="collapsed",
        )
        b1, b2 = st.columns(2)
        run_btn   = b1.button("▶ Analyse", use_container_width=True, type="primary")
        clear_btn = b2.button("✕ Clear",   use_container_width=True)

        if clear_btn:
            st.session_state.input_text = ""
            st.rerun()

    with col_out:
        if run_btn and text_input.strip():
            with st.spinner("Running analysis…"):
                mood, conf, tone, proba = get_sentiment(text_input)
                fscore, reasons = fake_score(text_input)
                risk = risk_label(fscore)
                auth = 100 - fscore

            st.markdown("#### Results")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Sentiment",     mood)
            m2.metric("Confidence",    f"{conf:.1%}")
            m3.metric("Authenticity",  f"{auth}%")
            m4.metric("Risk Score",    f"{fscore}/100")

            st.markdown(
                badge_sentiment(tone) + " &nbsp; " + badge_risk(risk),
                unsafe_allow_html=True
            )
            st.markdown("")
            st.markdown("**Class probabilities**")
            prob_df = pd.DataFrame({
                "Class": ["Negative", "Positive", "Neutral"],
                "Probability": proba,
            })
            st.bar_chart(prob_df.set_index("Class"), height=160)

            # Risk reasons
            if reasons:
                st.markdown("**Risk factors detected:**")
                for r in reasons:
                    st.markdown(f"- ⚠️ {r}")
            else:
                st.success("✅ No significant risk factors detected.")

        elif run_btn:
            st.warning("Please enter some text first.")
    if run_btn and text_input.strip() and show_explainability:
        st.markdown("---")
        st.markdown("#### Explainability")
        ex1, ex2, ex3 = st.columns(3)

        with ex1:
            st.markdown("**Feature Importance**")
            features = feature_importance(text_input)
            if features:
                words, weights = zip(*features)
                fig, ax = plt.subplots(figsize=(5, 3))
                fig.patch.set_facecolor('#12152a')
                ax.set_facecolor('#12152a')
                colors = ['#f87171' if w < 0 else '#4ade80' for w in weights]
                ax.barh(range(len(words)), weights, color=colors)
                ax.set_yticks(range(len(words)))
                ax.set_yticklabels(words, color='#d8deff', fontsize=9)
                ax.set_xlabel("Weight", color='#7c85c7')
                ax.tick_params(colors='#7c85c7')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#2a2e55')
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("No strong features found.")

        with ex2:
            st.markdown("**LIME Explanation**")
            try:
                lime_exp = lime_explain(text_input)
                lime_html = lime_exp.as_html()
                st.components.v1.html(
                    f'<div style="background:white;padding:10px;border-radius:8px;">{lime_html}</div>',
                    height=380, scrolling=True
                )
            except Exception as e:
                st.error(f"LIME error: {e}")

        with ex3:
            st.markdown("**SHAP Explanation**")
            try:
                shap_vals = shap_explainer([text_input])
                shap_html = shap.plots.text(shap_vals[0], display=False)
                st.components.v1.html(
                    f'<div style="background:white;padding:10px;border-radius:8px;">{shap_html}</div>',
                    height=380, scrolling=True
                )
            except Exception as e:
                st.error(f"SHAP error: {e}")

with tab2:
    st.markdown("#### Upload CSV or JSON for Bulk Analysis")

    uploaded = st.file_uploader("Upload file", type=["csv", "json"], label_visibility="collapsed")

    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_json(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            df_raw = pd.DataFrame()

        if not df_raw.empty:
            st.markdown(f"**Preview** — {len(df_raw)} rows × {len(df_raw.columns)} cols")
            st.dataframe(df_raw.head(5), use_container_width=True)

            text_col = st.selectbox("Select text column", df_raw.columns)

            if st.button("▶ Run Bulk Analysis", type="primary"):
                rows = df_raw[text_col].fillna("").astype(str).tolist()
                results = []

                progress = st.progress(0, text="Analysing…")
                total = len(rows)

                for idx, txt in enumerate(rows):
                    mood, conf, tone, proba = get_sentiment(txt)
                    fs, _ = fake_score(txt)
                    results.append({
                        "Review":          txt[:120] + ("…" if len(txt) > 120 else ""),
                        "Sentiment":       mood,
                        "Confidence":      round(conf, 3),
                        "Authenticity %":  100 - fs,
                        "Risk Level":      risk_label(fs),
                        "Risk Score":      fs,
                    })
                    progress.progress((idx + 1) / total, text=f"Processed {idx+1}/{total}")

                progress.empty()
                res_df = pd.DataFrame(results)

                # Summary metrics
                st.markdown("---")
                st.markdown("#### Summary")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total",        len(res_df))
                s2.metric("High Risk",    int((res_df["Risk Level"] == "High").sum()))
                s3.metric("Avg Auth %",   f"{res_df['Authenticity %'].mean():.1f}")
                s4.metric("Positives",    int((res_df["Sentiment"] == "Positive").sum()))

                # Charts
                ch1, ch2 = st.columns(2)
                with ch1:
                    st.markdown("**Sentiment distribution**")
                    st.bar_chart(res_df["Sentiment"].value_counts(), height=200)
                with ch2:
                    st.markdown("**Risk level distribution**")
                    st.bar_chart(
                        res_df["Risk Level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0),
                        height=200
                    )

                st.markdown("#### Full Results")
                st.dataframe(res_df, use_container_width=True)

                csv_bytes = res_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download CSV", csv_bytes,
                    "sentiment_fake_results.csv", "text/csv"
                )

