import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env file
load_dotenv()

# Get API key from .env
api_key = os.getenv("GOOGLE_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")


def ask_ai(df, question):
    context = f"""
    Dataset Summary:
    Rows: {df.shape[0]}
    Columns: {df.shape[1]}

    Column Types:
    Numerical: {df.select_dtypes(include='number').columns.tolist()}
    Categorical: {df.select_dtypes(include='object').columns.tolist()}

    Missing Values:
    {df.isnull().sum().to_dict()}

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