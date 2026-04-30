import google.generativeai as genai
import streamlit as st

# Configure Gemini
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

model = genai.GenerativeModel("gemini-2.5-flash")


def ask_ai(df, question):
    """
    Takes dataframe + question → returns AI answer
    """

    context = f"""
    Dataset Summary:
    Rows: {df.shape[0]}
    Columns: {df.shape[1]}

    Columns List:
    {list(df.columns)}

    Sample Data:
    {df.head(5).to_string()}
    """

    prompt = f"""
    You are a data analyst.

    Use the dataset below to answer the question.

    DATA:
    {context}

    QUESTION:
    {question}

    Give clear and simple answer.
    """

    response = model.generate_content(prompt)
    return response.text