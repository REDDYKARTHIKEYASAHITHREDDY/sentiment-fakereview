import streamlit as st
import pandas as pd
from transformers import pipeline
import torch

# Set page configuration
st.set_page_config(
    page_title="Fake Review Analyzer",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🔍 Fake Review Analyzer</p>', unsafe_allow_html=True)
st.markdown("---")


# Initialize session state for caching the model
@st.cache_resource
def load_model():
    """Load the pre-trained sentiment analysis model"""
    try:
        # Check for GPU availability
        device = 0 if torch.cuda.is_available() else -1
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

        st.info(f"🖥️ Running on: **{device_name}**")

        # Using a robust sentiment analysis model that can help detect fake reviews
        # You can replace this with a custom fine-tuned model for fake review detection
        classifier = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            batch_size=16 if torch.cuda.is_available() else 1  # Larger batch size for GPU
        )
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This app analyzes reviews to detect potential fake or suspicious content using:
    - **Sentiment Analysis**: Examines emotional tone
    - **Text Pattern Analysis**: Identifies common fake review patterns
    - **Length & Structure**: Checks review characteristics
    """)

    # Display GPU/CPU info
    st.header("🖥️ System Info")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        st.success(f"✅ GPU Detected: {gpu_name}")
        st.write(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        st.write(f"🚀 CUDA Version: {torch.version.cuda}")
    else:
        st.warning("⚠️ Running on CPU (slower)")
        st.write("For faster processing, use a GPU-enabled environment")

    st.header("⚙️ Settings")
    analysis_depth = st.selectbox(
        "Analysis Depth",
        ["Quick", "Detailed"],
        index=0
    )

    st.header("📝 Example Reviews")
    st.write("Click to try these examples:")

    example_reviews = {
        "🚩 Suspicious #1": "AMAZING PRODUCT!!! BEST EVER!!! BUY NOW!!!",
        "🚩 Suspicious #2": "Terrible waste of money don't buy",
        "✅ Authentic #1": "I've been using this coffee maker for about 3 months now. The brew quality is consistent and the programmable timer is really convenient for busy mornings. The carafe keeps coffee hot for about 2 hours. Only minor complaint is the water reservoir could be a bit larger.",
        "✅ Authentic #2": "Decent headphones for the price. Sound quality is good for casual listening, though audiophiles might want something better. Comfortable for 1-2 hour sessions but my ears get tired after that. Battery lasts about 8 hours as advertised.",
        "🚩 Suspicious #3": "Perfect",
        "⚠️ Borderline": "Good product highly recommend to everyone! Amazing quality!",
    }

    # Store selected example in session state
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = ""

    for label, review in example_reviews.items():
        if st.button(label, use_container_width=True):
            st.session_state.selected_example = review

    st.header("📊 Indicators")
    st.write("""
    **High Risk Signs:**
    - Extreme sentiments
    - Very short reviews
    - Excessive punctuation
    - Generic language
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["Single Review", "Batch Analysis", "Statistics"])

with tab1:
    st.subheader("Analyze a Single Review")

    # Use selected example if available
    default_text = st.session_state.selected_example if st.session_state.selected_example else ""

    review_text = st.text_area(
        "Enter the review text:",
        value=default_text,
        placeholder="Paste the review you want to analyze here...",
        height=150
    )

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        analyze_button = st.button("🔍 Analyze Review", type="primary", use_container_width=True)

    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)

    if clear_button:
        st.session_state.selected_example = ""
        st.rerun()

    if analyze_button and review_text:
        with st.spinner("Analyzing review..."):
            # Load model
            classifier = load_model()

            if classifier:
                # Basic text analysis
                word_count = len(review_text.split())
                char_count = len(review_text)
                exclamation_count = review_text.count('!')
                question_count = review_text.count('?')
                uppercase_ratio = sum(1 for c in review_text if c.isupper()) / max(len(review_text), 1)

                # Sentiment analysis
                sentiment_result = classifier(review_text[:512])[0]  # Truncate to model max length

                # Calculate fake review score (simple heuristic)
                fake_score = 0
                reasons = []

                # Check various indicators
                if word_count < 10:
                    fake_score += 25
                    reasons.append("⚠️ Review is very short")

                if exclamation_count > 3:
                    fake_score += 20
                    reasons.append("⚠️ Excessive exclamation marks")

                if uppercase_ratio > 0.3:
                    fake_score += 15
                    reasons.append("⚠️ Excessive capitalization")

                if sentiment_result['label'] in ['POSITIVE', 'NEGATIVE'] and sentiment_result['score'] > 0.98:
                    fake_score += 20
                    reasons.append("⚠️ Extreme sentiment detected")

                # Generic phrases that might indicate fake reviews
                generic_phrases = ['best product ever', 'waste of money', 'highly recommend',
                                   'don\'t buy', 'perfect', 'terrible', 'amazing product']
                if any(phrase in review_text.lower() for phrase in generic_phrases):
                    fake_score += 10
                    reasons.append("⚠️ Contains generic phrases")

                fake_score = min(fake_score, 100)

                # Display results
                st.markdown("---")
                st.subheader("📊 Analysis Results")

                # Metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Authenticity Score", f"{100 - fake_score}%")

                with col2:
                    st.metric("Word Count", word_count)

                with col3:
                    st.metric("Sentiment", sentiment_result['label'])

                with col4:
                    st.metric("Confidence", f"{sentiment_result['score']:.2%}")

                # Risk assessment
                st.markdown("---")
                if fake_score < 30:
                    st.success("✅ **LOW RISK**: This review appears to be authentic")
                elif fake_score < 60:
                    st.warning("⚠️ **MEDIUM RISK**: This review has some suspicious characteristics")
                else:
                    st.error("🚨 **HIGH RISK**: This review shows multiple signs of being fake")

                # Detailed indicators
                if analysis_depth == "Detailed":
                    st.markdown("---")
                    st.subheader("🔍 Detailed Analysis")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Text Characteristics:**")
                        st.write(f"- Characters: {char_count}")
                        st.write(f"- Words: {word_count}")
                        st.write(f"- Exclamation marks: {exclamation_count}")
                        st.write(f"- Question marks: {question_count}")
                        st.write(f"- Uppercase ratio: {uppercase_ratio:.2%}")

                    with col2:
                        st.write("**Risk Factors:**")
                        if reasons:
                            for reason in reasons:
                                st.write(f"- {reason}")
                        else:
                            st.write("- ✅ No major risk factors detected")

                # Progress bar for fake score
                st.markdown("---")
                st.write("**Fake Review Risk Level:**")
                st.progress(fake_score / 100)
                st.caption(f"Risk Score: {fake_score}/100")

with tab2:
    st.subheader("Batch Analysis")
    st.write("Upload a CSV file with reviews to analyze multiple reviews at once.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.write("**Preview of uploaded data:**")
            st.dataframe(df.head())

            # Let user select the column with reviews
            review_column = st.selectbox("Select the column containing reviews:", df.columns)

            if st.button("🔍 Analyze All Reviews", type="primary"):
                with st.spinner("Analyzing all reviews..."):
                    classifier = load_model()

                    if classifier:
                        results = []

                        # Filter valid reviews
                        valid_reviews = [str(r)[:512] for r in df[review_column]
                                         if pd.notna(r) and isinstance(r, str)]

                        # Batch processing for GPU efficiency
                        batch_size = 16 if torch.cuda.is_available() else 1
                        progress_bar = st.progress(0)

                        # Process in batches
                        for i in range(0, len(valid_reviews), batch_size):
                            batch = valid_reviews[i:i + batch_size]

                            try:
                                # Batch sentiment analysis (GPU optimized)
                                sentiments = classifier(batch)

                                for j, (review, sentiment) in enumerate(zip(batch, sentiments)):
                                    original_review = df[review_column].iloc[i + j]
                                    word_count = len(str(original_review).split())

                                    # Simple fake score calculation
                                    fake_score = 0
                                    if word_count < 10:
                                        fake_score += 25
                                    if str(original_review).count('!') > 3:
                                        fake_score += 20
                                    if sentiment['score'] > 0.98:
                                        fake_score += 20

                                    fake_score = min(fake_score, 100)

                                    results.append({
                                        'Review': str(original_review)[:100] + '...' if len(
                                            str(original_review)) > 100 else str(original_review),
                                        'Authenticity Score': 100 - fake_score,
                                        'Risk Level': 'Low' if fake_score < 30 else (
                                            'Medium' if fake_score < 60 else 'High'),
                                        'Sentiment': sentiment['label'],
                                        'Word Count': word_count
                                    })

                            except Exception as e:
                                st.warning(f"Error processing batch: {e}")

                            progress_bar.progress(min((i + batch_size) / len(valid_reviews), 1.0))

                        results_df = pd.DataFrame(results)

                        st.success(f"✅ Analyzed {len(results)} reviews!")

                        # Summary statistics
                        st.markdown("---")
                        st.subheader("📊 Summary Statistics")

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Total Reviews", len(results_df))

                        with col2:
                            high_risk = len(results_df[results_df['Risk Level'] == 'High'])
                            st.metric("High Risk Reviews", high_risk)

                        with col3:
                            avg_score = results_df['Authenticity Score'].mean()
                            st.metric("Avg Authenticity", f"{avg_score:.1f}%")

                        with col4:
                            positive_reviews = len(results_df[results_df['Sentiment'] == 'POSITIVE'])
                            st.metric("Positive Reviews", positive_reviews)

                        # Display results table
                        st.markdown("---")
                        st.subheader("📋 Detailed Results")
                        st.dataframe(results_df, use_container_width=True)

                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="review_analysis_results.csv",
                            mime="text/csv"
                        )

        except Exception as e:
            st.error(f"Error processing file: {e}")

with tab3:
    st.subheader("📈 Understanding the Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**What Makes a Review Suspicious?**")
        st.write("""
        - **Very short length** (< 10 words)
        - **Extreme sentiments** (overly positive or negative)
        - **Excessive punctuation** (!!!, ???)
        - **Generic phrases** commonly found in fake reviews
        - **Unusual capitalization patterns**
        - **Lack of specific details** about the product
        """)

    with col2:
        st.write("**How to Use This Tool**")
        st.write("""
        1. **Single Review**: Paste one review for quick analysis
        2. **Batch Analysis**: Upload a CSV with multiple reviews
        3. **Review Results**: Check authenticity score and risk level
        4. **Take Action**: Investigate high-risk reviews further
        """)

    st.markdown("---")

    # Performance tips
    st.subheader("⚡ Performance Tips")

    if torch.cuda.is_available():
        st.success("""
        **GPU Acceleration Active! 🚀**
        - Batch processing is optimized for your GPU
        - Large datasets will process much faster
        - Recommended batch size: 16-32 reviews
        """)
    else:
        st.info("""
        **Running on CPU**

        To enable GPU acceleration:
        1. Use a GPU-enabled environment (Google Colab, AWS, etc.)
        2. Install CUDA toolkit and PyTorch with CUDA support
        3. Run: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
        """)

    st.markdown("---")
    st.info("""
    **Note**: This tool uses machine learning models and heuristics to identify potentially fake reviews. 
    It should be used as a screening tool, and suspicious reviews should be manually verified. 
    No automated system is 100% accurate.
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Built with Streamlit and Transformers | Fake Review Analyzer v1.0 | GPU Optimized 🚀
    </div>
    """,
    unsafe_allow_html=True
)