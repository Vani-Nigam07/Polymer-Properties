from transformers import pipeline
from sentence_transformers import SentenceTransformer

def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def load_zero_shot_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
