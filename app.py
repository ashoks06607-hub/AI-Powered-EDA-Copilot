import streamlit as st
import pandas as pd

from modules.eda import (
    get_overview,
    get_column_summary,
    plot_numeric_distributions,
    plot_categorical_distributions,
    plot_bivariate,
    plot_correlation_heatmap
)

from modules.data_quality import generate_quality_report
from modules.feature_engineering import suggest_features

st.set_page_config(page_title="AI EDA Copilot", layout="wide")

# =========================
# TITLE
# =========================
st.title("📊 AI EDA Copilot")
st.markdown(
    "Analyze your dataset with **EDA, data quality checks, feature engineering suggestions, and AI insights**."
)

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully")

    # =========================
    # OVERVIEW
    # =========================
    st.header("📊 Dataset Overview")

    overview = get_overview(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", overview["rows"])
    col2.metric("Columns", overview["columns"])
    col3.metric("Missing", overview["missing"])
    col4.metric("Duplicates", overview["duplicates"])

    # =========================
    # COLUMN SUMMARY
    # =========================
    st.header("📋 Column Summary")
    st.dataframe(get_column_summary(df), use_container_width=True)

    # =========================
    # DATA QUALITY
    # =========================
    st.header("⚠️ Data Quality Report")
    report = generate_quality_report(df)

    if report.empty:
        st.success("No major data quality issues found")
    else:
        st.dataframe(report, use_container_width=True)

    # =========================
    # UNIVARIATE
    # =========================
    st.header("📈 Univariate Analysis")

    for fig in plot_numeric_distributions(df):
        st.plotly_chart(fig, use_container_width=True)

    for fig in plot_categorical_distributions(df):
        st.plotly_chart(fig, use_container_width=True)

    # =========================
    # BIVARIATE
    # =========================
    st.header("🔗 Bivariate Analysis")

    for fig in plot_bivariate(df):
        st.plotly_chart(fig, use_container_width=True)

    # =========================
    # CORRELATION
    # =========================
    st.header("🔥 Correlation Heatmap")
    st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)

    # =========================
    # FEATURE ENGINEERING
    # =========================
    st.header("🧠 Feature Engineering Suggestions")

    suggestions = suggest_features(df)

    if len(suggestions) == 0:
        st.info("No suggestions found")
    else:
        for s in suggestions:
            with st.expander(f"{s['feature']} ({s['type']})"):
                st.write("**Reason:**", s["reason"])
                st.code(s["code"], language="python")

    # =========================
    # 🤖 AI CHAT
    # =========================
    st.header("🤖 Ask AI About Your Data")

    question = st.text_input("Ask a question about your dataset")

    if question:
      with st.spinner("Thinking..."):
         answer = ask_ai(df, question)

      st.success("AI Answer:")
      st.write(answer)
                
    
else:
    st.info("👆 Upload a CSV file to begin")