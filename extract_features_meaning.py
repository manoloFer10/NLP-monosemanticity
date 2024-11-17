import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser


parser = StrOutputParser()

GROQ_API_KEY = ... #rellenar con la key o el camino hacia la key

reviewer = ChatGroq(
            model="llama-3.1-70b-versatile",
            groq_api_key= GROQ_API_KEY,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

def extract_text_from_feature(feature_csv: pd.DataFrame):
    feature_text=""

def extract_meaning_from_feature(feature_csv: pd.DataFrame, reviewer = reviewer):
    feature_text= extract_text_from_feature(feature_csv)

    prompted_q = [
            (
                "system",
                "You are a helpful assistant that answers plant biology questions. Answer concisely in one paragraph.",
            ),
            ("human", pregunta),
        ]




