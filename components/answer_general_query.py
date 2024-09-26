# Required imports
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from components.groq_response import get_groq_response


model = SentenceTransformer('all-MiniLM-L6-v2') 
dataset_path = "Datasets/anjibot_data.json"

def load_qa_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def answer_general_query(user_question):
    qa_data = load_qa_file(dataset_path)
    questions = [item['question'] for item in qa_data]
    question_embeddings = model.encode(questions)
    user_question_embedding = model.encode([user_question])
    similarities = cosine_similarity(user_question_embedding, question_embeddings)
    most_similar_index = np.argmax(similarities)
    max_similarity = similarities[0][most_similar_index]
    if max_similarity > 0.6:
        return qa_data[most_similar_index]['answer']
    else:
        return get_groq_response(user_question)
    
answer_general_query("hello")