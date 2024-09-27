import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from components.groq_response import get_groq_response
from components.anjibot_logging import append_to_sheet
import streamlit as st

dataset_path = "Datasets/anjibot_data.json"
model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)


def load_qa_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def embed_questions(questions):
    return model.encode(questions)

def embed_query(query):
    return model.encode([query])

def answer_general_query(user_question):
    qa_data = load_qa_file(dataset_path)
    
    questions = [item['question'] for item in qa_data]
    question_embeddings = embed_questions(questions)
    
    user_question_embedding = embed_query(user_question)
    
    similarities = cosine_similarity(user_question_embedding, question_embeddings)
    most_similar_index = np.argmax(similarities)
    max_similarity = similarities[0][most_similar_index]
    
    if max_similarity > 0.6:
        return qa_data[most_similar_index]['answer']
    else:
        return get_groq_response(user_question)
    
def main():
    st.title("Ask Anjibot 2.0")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = st.write_stream(answer_general_query(prompt))
        st.session_state.messages.append({"role": "assistant", "content": response})

        append_to_sheet(prompt, response)

if __name__ == "__main__":
    main()
    