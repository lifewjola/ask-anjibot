import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from components.groq_response import get_groq_response

dataset_path = "Datasets/anjibot_data.json"

def load_qa_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

def embed_questions(questions):
    # Fit and transform the questions to TF-IDF vectors
    return vectorizer.fit_transform(questions)

def embed_sentence(sentence):
    # Transform the user's question to a TF-IDF vector
    return vectorizer.transform([sentence])

def answer_general_query(user_question):
    qa_data = load_qa_file(dataset_path)
    
    # Embed the questions from the dataset
    questions = [item['question'] for item in qa_data]
    question_embeddings = embed_questions(questions)
    
    # Embed the user's question
    user_question_embedding = embed_sentence(user_question)
    
    # Calculate cosine similarity between user's question and all dataset questions
    similarities = cosine_similarity(user_question_embedding, question_embeddings)
    most_similar_index = np.argmax(similarities)
    max_similarity = similarities[0][most_similar_index]
    
    if max_similarity > 0.6:
        return qa_data[most_similar_index]['answer']
    else:
        return get_groq_response(user_question)