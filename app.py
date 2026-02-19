import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from lime.lime_text import LimeTextExplainer
from transformers import pipeline as hf_pipeline


# ══════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="Sentiment & Fake Review Analyzer",
    page_icon="🧠",
    layout="wide"
)

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
    padding: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 18px rgba(64, 105, 255, 0.18);
}
.stSidebar {
    background: rgba(15,15,40,0.6) !important;
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(80,110,255,0.2);
    box-shadow: 4px 0 12px rgba(50,70,255,0.15);
}
.verdict-card {
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align:center;color:#7eb8ff;margin-bottom:0.3rem;'>🧠 Sentiment & Fake Review Analyzer</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;color:#8899bb;margin-top:0;'>Powered by TF-IDF · Logistic Regression · DistilBERT · LIME · SHAP</p>",
    unsafe_allow_html=True
)
st.markdown("---")


# ══════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════
@st.cache_resource
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
        "Hate this completely, total waste of money and time."
    ]
    labels = [1, 0, 2, 2, 0, 1, 0, 1, 2, 0]
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
        ('clf',   LogisticRegression(random_state=42))
    ])
    model.fit(texts, labels)
    return model


@st.cache_resource
def load_bert_classifier():
    device = 0 if torch.cuda.is_available() else -1
    return hf_pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        batch_size=16 if torch.cuda.is_available() else 1,
    )


@st.cache_resource
def load_shap_explainer(_model):
    masker = shap.maskers.Text()
    return shap.Explainer(lambda texts: _model.predict_proba(texts), masker)


sent_model     = load_sentiment_model()
shap_explainer = load_shap_explainer(sent_model)


# ══════════════════════════════════════════════
#  SHARED HELPERS
# ══════════════════════════════════════════════
LABEL_NAMES = ["Negative", "Positive", "Neutral"]
LABEL_TONES = ["negative", "positive", "neutral"]
TONE_ICONS  = {"positive": "😊", "negative": "😠", "neutral": "😐"}
TONE_BG     = {"positive": "#1a3a26", "negative": "#3a1a1a", "neutral": "#2a2a1a"}

GENERIC_PHRASES = [
    'best product ever', 'waste of money', 'highly recommend',
    "don't buy", 'perfect', 'terrible', 'amazing product'
]


def get_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return "Neutral", 0.0, "neutral", [0.0, 0.0, 1.0]
    proba = sent_model.predict_proba([text])[0]
    label = sent_model.predict([text])[0]
    return LABEL_NAMES[label], float(max(proba)), LABEL_TONES[label], proba.tolist()


def compute_fake_score(text, bert_result):
    score, reasons = 0, []
    wc      = len(text.split())
    excl    = text.count('!')
    upper_r = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if wc < 10:
        score += 25; reasons.append("⚠️ Very short review (< 10 words)")
    if excl > 3:
        score += 20; reasons.append("⚠️ Excessive exclamation marks")
    if upper_r > 0.3:
        score += 15; reasons.append("⚠️ Excessive capitalization")
    if bert_result['score'] > 0.98:
        score += 20; reasons.append("⚠️ Extreme sentiment confidence (BERT)")
    if any(p in text.lower() for p in GENERIC_PHRASES):
        score += 10; reasons.append("⚠️ Generic / stock phrases detected")
    return min(score, 100), reasons


def risk_label(score):
    if score < 30:  return "LOW",    "✅", "#1a3a26"
    if score < 60:  return "MEDIUM", "⚠️", "#3a2f10"
    return           "HIGH",          "🚨", "#3a1a1a"


def feature_importance_chart(text):
    vec   = sent_model.named_steps['tfidf']
    clf   = sent_model.named_steps['clf']
    tvec  = vec.transform([text])
    names = vec.get_feature_names_out()
    pred  = clf.predict(tvec)[0]
    coefs = clf.coef_[pred]
    idxs  = tvec.indices
    words, weights = [], []
    for i in idxs:
        w = coefs[i] * tvec.data[np.where(tvec.indices == i)[0][0]]
        if abs(w) > 0.01:
            words.append(names[i])
            weights.append(w)
    if not weights:
        return None
    pairs = sorted(zip(words, weights), key=lambda x: abs(x[1]), reverse=True)[:10]
    words, weights = zip(*pairs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(range(len(words)), weights,
            color=['#e05c5c' if w < 0 else '#5ce075' for w in weights])
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.set_xlabel("Weight")
    ax.set_title("Top Words")
    fig.patch.set_facecolor('#0d1127')
    ax.set_facecolor('#0d1127')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')
    return fig


def lime_html_block(text):
    exp  = LimeTextExplainer(class_names=['Negative', 'Positive', 'Neutral'])
    expl = exp.explain_instance(text, sent_model.predict_proba, num_features=10)
    return expl.as_html()


def shap_html_block(text):
    sv = shap_explainer([text])
    return shap.plots.text(sv[0], display=False)


def white_iframe(html, height=420):
    st.components.v1.html(
        f'<div style="background:white;padding:15px;border-radius:10px;">{html}</div>',
        height=height, scrolling=True
    )


def read_uploaded(f):
    return pd.read_csv(f) if f.name.endswith(".csv") else pd.read_json(f)


# ══════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.header("🧭 Module")
    module = st.radio("", [
        "🔍 Fake Review Analyzer",
        "🎭 Sentiment Analyzer",
        "🔀 Combined Analyzer",
    ], index=0, label_visibility="collapsed")

    st.markdown("---")
    st.header("💡 Try Examples")
    examples = {
        "😊 Positive":    "I love this amazing product! It's fantastic and wonderful!",
        "😠 Negative":    "This is absolutely terrible and the worst experience ever.",
        "😐 Neutral":     "It's okay, nothing too special or exciting.",
        "🚩 Fake (high)": "AMAZING PRODUCT!!! BEST EVER!!! BUY NOW!!!",
        "✅ Authentic":   "I've been using this coffee maker for 3 months. Consistent brew, convenient timer. Carafe keeps coffee hot ~2 hrs. Water reservoir could be slightly larger.",
        "⚠️ Borderline":  "Good product, highly recommend to everyone! Amazing quality!",
    }
    if 'shared_text' not in st.session_state:
        st.session_state.shared_text = ""
    for label, ex in examples.items():
        if st.button(label, use_container_width=True):
            st.session_state.shared_text = ex

    st.markdown("---")
    st.header("🖥️ System")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        st.success(f"✅ {gpu}")
        st.caption(f"VRAM: {mem:.1f} GB  |  CUDA {torch.version.cuda}")
    else:
        st.warning("⚠️ CPU only (BERT will be slow)")


# ══════════════════════════════════════════════════════════════════
#  MODULE 1 — FAKE REVIEW ANALYZER  (single + bulk)
# ══════════════════════════════════════════════════════════════════
if module == "🔍 Fake Review Analyzer":
    st.subheader("🔍 Fake Review Analyzer")
    st.caption("DistilBERT confidence + 5-dimension heuristic scoring. Detects short length, punctuation abuse, capitalization, sentiment extremity & generic phrases.")

    single_tab, bulk_tab = st.tabs(["📝 Single Review", "📦 Bulk Analysis"])

    # ── SINGLE ──────────────────────────────────────────
    with single_tab:
        depth = st.selectbox("Analysis depth", ["Quick", "Detailed"], index=1, key="fake_depth")
        text  = st.text_area(
            "Paste a review:",
            value=st.session_state.shared_text,
            height=140,
            placeholder="Enter any review to check for authenticity...",
            key="fake_single_input"
        )
        c1, c2, _ = st.columns([1, 1, 4])
        run   = c1.button("🔍 Analyze", type="primary", use_container_width=True, key="fake_run")
        clear = c2.button("🗑️ Clear",                    use_container_width=True, key="fake_clr")
        if clear:
            st.session_state.shared_text = ""
            st.rerun()

        if run and text.strip():
            with st.spinner("Running BERT + heuristics..."):
                bert_clf   = load_bert_classifier()
                bert_res   = bert_clf(text[:512])[0]
                fake_score, reasons = compute_fake_score(text, bert_res)
                risk, risk_icon, risk_bg = risk_label(fake_score)
                auth_score  = 100 - fake_score
                wc          = len(text.split())
                upper_r     = sum(1 for c in text if c.isupper()) / max(len(text), 1)

            st.markdown("---")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Authenticity",    f"{auth_score}%")
            m2.metric("Fake Risk Score", f"{fake_score}/100")
            m3.metric("Risk Level",      risk)
            m4.metric("BERT Label",      bert_res['label'])
            m5.metric("BERT Confidence", f"{bert_res['score']:.2%}")

            st.markdown(f"""
            <div class="verdict-card" style="background:{risk_bg};">
                <div style="font-size:2.5rem;">{risk_icon}</div>
                <div style="font-size:1.4rem;font-weight:bold;color:#e0e6ff;">{risk} RISK</div>
                <div style="color:#aab4cc;margin-top:4px;">Authenticity Score: {auth_score}%</div>
            </div>""", unsafe_allow_html=True)

            st.write("**Fake Review Risk Meter:**")
            st.progress(fake_score / 100)
            st.caption(f"Score: {fake_score}/100")

            if reasons:
                st.markdown("**🚩 Risk Flags:**")
                cols = st.columns(min(len(reasons), 3))
                for i, r in enumerate(reasons):
                    cols[i % 3].warning(r)
            else:
                st.success("✅ No risk flags triggered.")

            if depth == "Detailed":
                st.markdown("---")
                st.markdown("### 🔍 Detailed Breakdown")
                d1, d2 = st.columns(2)
                with d1:
                    st.markdown("**📝 Text Characteristics**")
                    st.write(f"- Words: **{wc}**")
                    st.write(f"- Characters: **{len(text)}**")
                    st.write(f"- Exclamation marks: **{text.count('!')}**")
                    st.write(f"- Question marks: **{text.count('?')}**")
                    st.write(f"- Uppercase ratio: **{upper_r:.1%}**")
                with d2:
                    st.markdown("**🚩 Risk Factor Detail**")
                    if reasons:
                        for r in reasons: st.write(f"- {r}")
                    else:
                        st.write("- ✅ No risk factors detected")

                st.markdown("---")
                st.markdown("#### 🟡 LIME Word Influence")
                try:
                    white_iframe(lime_html_block(text), height=380)
                except Exception as e:
                    st.error(f"LIME Error: {e}")

        elif run:
            st.warning("Please enter a review.")

    # ── BULK ────────────────────────────────────────────
    with bulk_tab:
        st.markdown("Upload a **CSV or JSON** file — one review per row.")
        f = st.file_uploader("Upload file", type=["csv", "json"], key="fake_bulk_upload")
        if f:
            df = read_uploaded(f)
            st.dataframe(df.head(), use_container_width=True)
            col = st.selectbox("Text column:", df.columns, key="fake_bulk_col")

            if st.button("▶️ Run Fake Detection", type="primary", key="fake_bulk_run"):
                bert_clf = load_bert_classifier()
                results, progress = [], st.progress(0)
                total = len(df)

                for idx, row in df.iterrows():
                    raw = str(row[col])
                    try:
                        br        = bert_clf(raw[:512])[0]
                        fs, _     = compute_fake_score(raw, br)
                        rl, _, _  = risk_label(fs)
                        bl, bc    = br['label'], f"{br['score']:.2%}"
                    except Exception:
                        fs, rl, bl, bc = 0, "Error", "N/A", "N/A"

                    results.append({
                        "Snippet":          raw[:80] + ("..." if len(raw) > 80 else ""),
                        "Authenticity (%)": 100 - fs,
                        "Fake Risk Score":  fs,
                        "Risk Level":       rl,
                        "BERT Label":       bl,
                        "BERT Confidence":  bc,
                        "Word Count":       len(raw.split()),
                    })
                    progress.progress((idx + 1) / total)

                rdf = pd.DataFrame(results)
                st.success(f"✅ Analyzed {len(rdf)} reviews!")

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total",          len(rdf))
                s2.metric("🚨 High Risk",   int((rdf["Risk Level"] == "HIGH").sum()))
                s3.metric("⚠️ Medium Risk", int((rdf["Risk Level"] == "MEDIUM").sum()))
                s4.metric("✅ Low Risk",    int((rdf["Risk Level"] == "LOW").sum()))

                st.markdown("#### 📊 Risk Distribution")
                st.bar_chart(rdf["Risk Level"].value_counts())
                st.markdown("#### 📋 Full Results")
                st.dataframe(rdf, use_container_width=True)
                st.download_button("📥 Download CSV", rdf.to_csv(index=False).encode(),
                                   "fake_review_results.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════
#  MODULE 2 — SENTIMENT ANALYZER  (single + bulk)
# ══════════════════════════════════════════════════════════════════
elif module == "🎭 Sentiment Analyzer":
    st.subheader("🎭 Sentiment Analyzer")
    st.caption("TF-IDF + Logistic Regression · 3-class (Positive / Negative / Neutral) · Feature Importance · LIME · SHAP")

    single_tab, bulk_tab = st.tabs(["📝 Single Text", "📦 Bulk Analysis"])

    # ── SINGLE ──────────────────────────────────────────
    with single_tab:
        text = st.text_area(
            "Enter text:",
            value=st.session_state.shared_text,
            height=140,
            placeholder="Type or paste any text to analyze its sentiment...",
            key="sent_single_input"
        )
        c1, c2, _ = st.columns([1, 1, 4])
        run   = c1.button("🎭 Analyze", type="primary", use_container_width=True, key="sent_run")
        clear = c2.button("🗑️ Clear",                    use_container_width=True, key="sent_clr")
        if clear:
            st.session_state.shared_text = ""
            st.rerun()

        if run and text.strip():
            with st.spinner("Analyzing sentiment..."):
                mood, conf, tone, proba = get_sentiment(text)

            st.markdown("---")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Sentiment",  mood)
            m2.metric("Confidence", f"{conf:.2%}")
            m3.metric("Tone",       tone.capitalize())
            m4.metric("Neg %",      f"{proba[0]:.2%}")
            m5.metric("Pos %",      f"{proba[1]:.2%}")

            icon = TONE_ICONS.get(tone, "❓")
            bg   = TONE_BG.get(tone, "#1a1a2e")
            st.markdown(f"""
            <div class="verdict-card" style="background:{bg};">
                <div style="font-size:2.5rem;">{icon}</div>
                <div style="font-size:1.4rem;font-weight:bold;color:#e0e6ff;">{mood}</div>
                <div style="color:#aab4cc;">Confidence: {conf:.2%}</div>
                <div style="color:#667788;font-size:0.85rem;margin-top:6px;">
                    Neg {proba[0]:.2%} &nbsp;|&nbsp; Pos {proba[1]:.2%} &nbsp;|&nbsp; Neu {proba[2]:.2%}
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 🧠 Explainability")
            x1, x2, x3 = st.columns(3)

            with x1:
                st.markdown("#### 📌 Feature Importance")
                fig = feature_importance_chart(text)
                if fig:
                    st.pyplot(fig, clear_figure=True)
                else:
                    st.info("No strong features found.")

            with x2:
                st.markdown("#### 🟡 LIME Explanation")
                try:
                    white_iframe(lime_html_block(text))
                except Exception as e:
                    st.error(f"LIME Error: {e}")

            with x3:
                st.markdown("#### 🔵 SHAP Explanation")
                try:
                    white_iframe(shap_html_block(text))
                except Exception as e:
                    st.error(f"SHAP Error: {e}")

        elif run:
            st.warning("Please enter some text.")

    # ── BULK ────────────────────────────────────────────
    with bulk_tab:
        st.markdown("Upload a **CSV or JSON** file — one text per row.")
        f = st.file_uploader("Upload file", type=["csv", "json"], key="sent_bulk_upload")
        if f:
            df = read_uploaded(f)
            st.dataframe(df.head(), use_container_width=True)
            col = st.selectbox("Text column:", df.columns, key="sent_bulk_col")

            if st.button("▶️ Run Sentiment Analysis", type="primary", key="sent_bulk_run"):
                results, progress = [], st.progress(0)
                total = len(df)

                for idx, row in df.iterrows():
                    raw = str(row[col])
                    mood, conf, tone, proba = get_sentiment(raw)
                    results.append({
                        "Snippet":       raw[:80] + ("..." if len(raw) > 80 else ""),
                        "Sentiment":     mood,
                        "Confidence":    round(conf, 3),
                        "Tone":          tone,
                        "Negative (%)":  f"{proba[0]:.2%}",
                        "Positive (%)":  f"{proba[1]:.2%}",
                        "Neutral (%)":   f"{proba[2]:.2%}",
                        "Word Count":    len(raw.split()),
                    })
                    progress.progress((idx + 1) / total)

                rdf = pd.DataFrame(results)
                st.success(f"✅ Analyzed {len(rdf)} texts!")

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total",       len(rdf))
                s2.metric("😊 Positive", int((rdf["Tone"] == "positive").sum()))
                s3.metric("😠 Negative", int((rdf["Tone"] == "negative").sum()))
                s4.metric("😐 Neutral",  int((rdf["Tone"] == "neutral").sum()))

                st.markdown("#### 📊 Sentiment Distribution")
                st.bar_chart(rdf["Tone"].value_counts())
                st.markdown("#### 📋 Full Results")
                st.dataframe(rdf, use_container_width=True)
                st.download_button("📥 Download CSV", rdf.to_csv(index=False).encode(),
                                   "sentiment_results.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════
#  MODULE 3 — COMBINED ANALYZER  (single + bulk)
# ══════════════════════════════════════════════════════════════════
elif module == "🔀 Combined Analyzer":
    st.subheader("🔀 Combined Sentiment & Fake Review Analyzer")
    st.caption("Both models run simultaneously — full sentiment verdict + authenticity score + explainability in one view.")

    single_tab, bulk_tab = st.tabs(["📝 Single Text", "📦 Bulk Analysis"])

    # ── SINGLE ──────────────────────────────────────────
    with single_tab:
        depth = st.selectbox("Analysis depth", ["Quick", "Detailed"], index=1, key="comb_depth")
        text  = st.text_area(
            "Enter text:",
            value=st.session_state.shared_text,
            height=140,
            placeholder="Paste any review or text to get both sentiment and authenticity analysis...",
            key="comb_single_input"
        )
        c1, c2, _ = st.columns([1, 1, 4])
        run   = c1.button("⚡ Analyze Both", type="primary", use_container_width=True, key="comb_run")
        clear = c2.button("🗑️ Clear",                         use_container_width=True, key="comb_clr")
        if clear:
            st.session_state.shared_text = ""
            st.rerun()

        if run and text.strip():
            with st.spinner("Running Sentiment + Fake Review models..."):
                mood, conf, tone, proba      = get_sentiment(text)
                bert_clf                     = load_bert_classifier()
                bert_res                     = bert_clf(text[:512])[0]
                fake_score, reasons          = compute_fake_score(text, bert_res)
                risk, risk_icon, risk_bg     = risk_label(fake_score)
                auth_score                   = 100 - fake_score
                wc                           = len(text.split())
                upper_r                      = sum(1 for c in text if c.isupper()) / max(len(text), 1)

            st.markdown("---")

            # 6 top metrics
            st.markdown("### 📊 At a Glance")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Sentiment",    mood)
            m2.metric("Sent. Conf.",  f"{conf:.2%}")
            m3.metric("Tone",         tone.capitalize())
            m4.metric("Authenticity", f"{auth_score}%")
            m5.metric("Risk Score",   f"{fake_score}/100")
            m6.metric("Risk Level",   risk)

            st.markdown("---")

            # Two verdict cards side by side
            vc1, vc2 = st.columns(2)

            with vc1:
                st.markdown("#### 🎭 Sentiment Verdict")
                icon = TONE_ICONS.get(tone, "❓")
                bg   = TONE_BG.get(tone, "#1a1a2e")
                st.markdown(f"""
                <div class="verdict-card" style="background:{bg};">
                    <div style="font-size:2.5rem;">{icon}</div>
                    <div style="font-size:1.4rem;font-weight:bold;color:#e0e6ff;">{mood}</div>
                    <div style="color:#aab4cc;">Confidence: {conf:.2%}</div>
                    <div style="color:#667788;font-size:0.85rem;margin-top:6px;">
                        Neg {proba[0]:.2%} &nbsp;|&nbsp; Pos {proba[1]:.2%} &nbsp;|&nbsp; Neu {proba[2]:.2%}
                    </div>
                </div>""", unsafe_allow_html=True)

            with vc2:
                st.markdown("#### 🔍 Authenticity Verdict")
                st.markdown(f"""
                <div class="verdict-card" style="background:{risk_bg};">
                    <div style="font-size:2.5rem;">{risk_icon}</div>
                    <div style="font-size:1.4rem;font-weight:bold;color:#e0e6ff;">{risk} RISK</div>
                    <div style="color:#aab4cc;">Authenticity: {auth_score}%</div>
                    <div style="color:#667788;font-size:0.85rem;margin-top:6px;">
                        BERT: {bert_res['label']} ({bert_res['score']:.2%})
                    </div>
                </div>""", unsafe_allow_html=True)
                st.progress(fake_score / 100)
                st.caption(f"Risk Score: {fake_score}/100")

            # Risk flags
            if reasons:
                st.markdown("**🚩 Risk Flags:**")
                fl = st.columns(min(len(reasons), 3))
                for i, r in enumerate(reasons):
                    fl[i % 3].warning(r)
            else:
                st.success("✅ No fake review risk flags triggered.")

            # Detailed explainability
            if depth == "Detailed":
                st.markdown("---")
                st.markdown("### 🧠 Explainability (Sentiment Model)")
                x1, x2, x3 = st.columns(3)

                with x1:
                    st.markdown("#### 📌 Feature Importance")
                    fig = feature_importance_chart(text)
                    if fig:
                        st.pyplot(fig, clear_figure=True)
                    else:
                        st.info("No strong features found.")

                with x2:
                    st.markdown("#### 🟡 LIME")
                    try:
                        white_iframe(lime_html_block(text))
                    except Exception as e:
                        st.error(f"LIME: {e}")

                with x3:
                    st.markdown("#### 🔵 SHAP")
                    try:
                        white_iframe(shap_html_block(text))
                    except Exception as e:
                        st.error(f"SHAP: {e}")

                st.markdown("---")
                st.markdown("### 📝 Text Characteristics")
                t1, t2, t3, t4, t5 = st.columns(5)
                t1.metric("Words",           wc)
                t2.metric("Characters",      len(text))
                t3.metric("Exclamations",    text.count('!'))
                t4.metric("Questions",       text.count('?'))
                t5.metric("Uppercase Ratio", f"{upper_r:.1%}")

        elif run:
            st.warning("Please enter some text.")

    # ── BULK ────────────────────────────────────────────
    with bulk_tab:
        st.markdown("Upload a **CSV or JSON** file. Runs **both** sentiment and fake review detection on every row.")
        f = st.file_uploader("Upload file", type=["csv", "json"], key="comb_bulk_upload")
        if f:
            df = read_uploaded(f)
            st.dataframe(df.head(), use_container_width=True)
            col = st.selectbox("Text column:", df.columns, key="comb_bulk_col")

            if st.button("▶️ Run Combined Analysis", type="primary", key="comb_bulk_run"):
                bert_clf = load_bert_classifier()
                results, progress = [], st.progress(0)
                total = len(df)

                for idx, row in df.iterrows():
                    raw = str(row[col])
                    mood, conf, tone, proba = get_sentiment(raw)
                    try:
                        br            = bert_clf(raw[:512])[0]
                        fs, _         = compute_fake_score(raw, br)
                        rl, _, _      = risk_label(fs)
                        bert_lbl      = br['label']
                        bert_conf_str = f"{br['score']:.2%}"
                    except Exception:
                        fs, rl, bert_lbl, bert_conf_str = 0, "Error", "N/A", "N/A"

                    results.append({
                        "Snippet":          raw[:80] + ("..." if len(raw) > 80 else ""),
                        "Sentiment":        mood,
                        "Sent. Confidence": round(conf, 3),
                        "Tone":             tone,
                        "Authenticity (%)": 100 - fs,
                        "Fake Risk Score":  fs,
                        "Risk Level":       rl,
                        "BERT Label":       bert_lbl,
                        "BERT Confidence":  bert_conf_str,
                        "Word Count":       len(raw.split()),
                    })
                    progress.progress((idx + 1) / total)

                rdf = pd.DataFrame(results)
                st.success(f"✅ Analyzed {len(rdf)} rows!")

                s1, s2, s3, s4, s5, s6 = st.columns(6)
                s1.metric("Total",          len(rdf))
                s2.metric("😊 Positive",    int((rdf["Tone"] == "positive").sum()))
                s3.metric("😠 Negative",    int((rdf["Tone"] == "negative").sum()))
                s4.metric("😐 Neutral",     int((rdf["Tone"] == "neutral").sum()))
                s5.metric("🚨 High Risk",   int((rdf["Risk Level"] == "HIGH").sum()))
                s6.metric("✅ Low Risk",    int((rdf["Risk Level"] == "LOW").sum()))

                ch1, ch2 = st.columns(2)
                with ch1:
                    st.markdown("#### 📊 Sentiment Distribution")
                    st.bar_chart(rdf["Tone"].value_counts())
                with ch2:
                    st.markdown("#### 📊 Risk Distribution")
                    st.bar_chart(rdf["Risk Level"].value_counts())

                st.markdown("#### 📋 Full Results")
                st.dataframe(rdf, use_container_width=True)
                st.download_button("📥 Download CSV", rdf.to_csv(index=False).encode(),
                                   "combined_results.csv", "text/csv")


# ══════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#445;font-size:0.85rem;'>"
    "🧠 Sentiment & Fake Review Analyzer &nbsp;·&nbsp; "
    "Streamlit · Scikit-learn · DistilBERT · LIME · SHAP"
    "</div>",
    unsafe_allow_html=True
)