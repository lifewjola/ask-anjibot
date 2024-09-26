# Required imports
import json
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
from components.groq_response import get_groq_response

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

dataset_path = "Datasets/anjibot_data.json"

def load_qa_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def embed_sentence(sentence):
    # Tokenize the input sentence and create input tensors
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the embeddings from the last hidden state
    return outputs.last_hidden_state.mean(dim=1).numpy()

def answer_general_query(user_question):
    qa_data = load_qa_file(dataset_path)
    
    # Embed the questions from the dataset
    questions = [item['question'] for item in qa_data]
    question_embeddings = np.array([embed_sentence(question) for question in questions])
    
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

# Test the function
answer_general_query("hello")
