# Required imports
import json
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from components.groq_response import get_groq_response

nlp = spacy.load("en_core_web_md")  
dataset_path = "Datasets/anjibot_data.json"

def load_qa_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def get_embedding(text):
    return nlp(text).vector 

def answer_general_query(user_question):
    qa_data = load_qa_file(dataset_path)
    
    questions = [item['question'] for item in qa_data]
    question_embeddings = np.array([get_embedding(question) for question in questions])
    user_question_embedding = get_embedding(user_question).reshape(1, -1)
    
    similarities = cosine_similarity(user_question_embedding, question_embeddings)
    most_similar_index = np.argmax(similarities)
    max_similarity = similarities[0][most_similar_index]
    
    if max_similarity > 0.6:
        return qa_data[most_similar_index]['answer']
    else:
        return get_groq_response(user_question)
