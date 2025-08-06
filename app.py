import streamlit as st
import requests
import pdfplumber
import docx
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import logging

nltk.download("punkt")

API_URL = "https://teqzjokho5.execute-api.us-east-2.amazonaws.com/Prod/predict"
CONFIDENCE_THRESHOLD = 50.0

logger = logging.getLogger("sentiment-analyzer")
logger.setLevel(logging.INFO)

st.set_page_config(page_title="Sentiment Analyzer", layout="wide")
st.title("💬 Sentiment Analyzer")
st.write("Analyze text or documents using AWS SageMaker with visual insights.")

# --- Utility Functions ---

def extract_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() or '' for page in pdf.pages)
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return ""

def classify_sentences(text):
    sentences = sent_tokenize(text)
    results = []
    for i, sent in enumerate(sentences):
        try:
            res = requests.post(API_URL, json={"text": sent})
            pred = res.json()["result"][0]
            results.append({
                "Index": i + 1,
                "Sentence": sent,
                "Label": pred["label"],
                "Confidence": round(pred["score"] * 100, 2)
            })
        except Exception as e:
            logger.error(f"Error on sentence {i+1}: {str(e)}")
            results.append({
                "Index": i + 1,
                "Sentence": sent,
                "Label": "ERROR",
                "Confidence": 0
            })
    return pd.DataFrame(results)

def generate_wordcloud(sentences):
    words = []
    for sent in sentences:
        tokens = word_tokenize(sent.lower())
        words.extend([w for w in tokens if w.isalpha()])
    freq = Counter(words)
    wc = WordCloud(width=600, height=400, background_color="white").generate_from_frequencies(freq)
    return wc

def render_analysis(df, source_label):
    st.subheader(f"📌 Summary ({source_label})")
    total = len(df)
    avg_conf = round(df["Confidence"].mean(), 2)
    counts = df["Label"].value_counts().to_dict()
    st.markdown(f"""
    - **Total Sentences:** {total}
    - **Average Confidence:** {avg_conf}%
    - **Label Counts:** {counts}
    """)

    # Show results table
    st.dataframe(df)

    # Sentiment Distribution
    st.subheader("📊 Sentiment Distribution")
    st.bar_chart(df["Label"].value_counts())

    # Word clouds
    st.subheader("☁️ Word Clouds")
    col1, col2 = st.columns(2)
    pos_sent = df[df["Label"] == "POSITIVE"]["Sentence"].tolist()
    neg_sent = df[df["Label"] == "NEGATIVE"]["Sentence"].tolist()
    if pos_sent:
        with col1:
            st.caption("Positive")
            st.image(generate_wordcloud(pos_sent).to_array(), use_container_width=True)
    if neg_sent:
        with col2:
            st.caption("Negative")
            st.image(generate_wordcloud(neg_sent).to_array(), use_container_width=True)

    # Heatmap
    if len(df) >= 2:
        st.subheader("🔥 Sentiment Timeline (Heatmap)")
        heatmap_df = df.pivot_table(index="Label", columns="Index", values="Confidence", aggfunc="mean", fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 2))
        sns.heatmap(heatmap_df, cmap="coolwarm", cbar=True, ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough data to generate heatmap.")

    # Export
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results as CSV", data=csv, file_name="sentiment_results.csv", mime="text/csv")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🔍 Single Sentence", "📝 Paragraph", "📄 Document"])

# --- Tab 1: Single Sentence ---
with tab1:
    input_text = st.text_area("Enter a short sentence", placeholder="Type something here...")
    if st.button("Analyze Sentence"):
        if not input_text.strip():
            st.warning("Please enter a sentence.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    response = requests.post(API_URL, json={"text": input_text})
                    result = response.json()["result"][0]
                    st.success(f"Prediction: **{result['label']}**")
                    st.info(f"Confidence: {round(result['score'] * 100, 2)}%")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# --- Tab 2: Paragraph with Analysis Persistence ---
with tab2:
    para_text = st.text_area("Paste your paragraph here", key="paragraph_input")
    if "para_df" not in st.session_state:
        st.session_state.para_df = None
        st.session_state.para_text = ""

    analyze_paragraph = st.button("Analyze Paragraph")
    if analyze_paragraph or (st.session_state.para_text != para_text and para_text.strip()):
        if not para_text.strip():
            st.warning("Please paste a paragraph.")
        else:
            with st.spinner("Analyzing paragraph..."):
                df_para = classify_sentences(para_text)
                df_para = df_para[df_para["Confidence"] >= CONFIDENCE_THRESHOLD]
                st.session_state.para_df = df_para
                st.session_state.para_text = para_text

    if st.session_state.para_df is not None:
        render_analysis(st.session_state.para_df, "Paragraph")

# --- Tab 3: Document Upload with Persistence ---
with tab3:
    uploaded_file = st.file_uploader("Upload a .txt, .pdf, or .docx file", type=["txt", "pdf", "docx"])

    if "doc_df" not in st.session_state:
        st.session_state.doc_df = None
        st.session_state.doc_text = ""
        st.session_state.doc_name = ""

    if uploaded_file:
        doc_text = extract_text(uploaded_file)
        st.text_area("Extracted Text Preview", value=doc_text[:2000] + "..." if len(doc_text) > 2000 else doc_text, height=200)

        analyze = st.button("Analyze Document")
        new_file_uploaded = uploaded_file.name != st.session_state.doc_name

        if analyze or new_file_uploaded:
            with st.spinner("Analyzing document..."):
                df_doc = classify_sentences(doc_text)
                df_doc = df_doc[df_doc["Confidence"] >= CONFIDENCE_THRESHOLD]
                st.session_state.doc_df = df_doc
                st.session_state.doc_text = doc_text
                st.session_state.doc_name = uploaded_file.name

    if st.session_state.doc_df is not None:
        render_analysis(st.session_state.doc_df, "Document")

