'''
TO-DO
- Update datasets to represent current state of things and wider range of questions
- Add more intents variation to docs link and lecturers logic functions
'''

# Required imports
import streamlit as st
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from google.oauth2 import service_account
from googleapiclient.discovery import build
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize Groq model
model = ChatGroq(model="llama3-8b-8192")

# Define prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''You're the AI course representative of Computer Science Department 400 lvl Group A. 
            You're always helpful and you answer your classmates questions only based on the provided information. 
            If you don't know the answer - just reply with an excuse that you don't know. 
            Keep your answers brief and to the point. 
            Be kind, jovial, funny, and respectful.
            If it's a general info in natural language that isn't class related that you know the answer to, you can attempt answering.
            But anything school related or class related that you don't know the answer to, direct user to 'Anji' who's the human course rep of the department.''',
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Load additional datasets
lecturer_data = pd.read_csv("Datasets/lecturers.csv").astype(str)
doc_link_data = pd.read_csv("Datasets/docs_link.csv").astype(str)

# Function to load Q&A data from a JSON file
filepath = 'Datasets/anjibot_data.json'

def load_qa_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

qa_data = load_qa_file(filepath)

# Initialize conversation history
messages = [
    SystemMessage(content="You're the AI course representative of Computer Science Department 400 lvl Group A. You're always helpful..."),
]

def get_groq_response(query):
    # Add the new user query to the message history
    messages.append(HumanMessage(content=query))

    # Create a prompt with the message history
    chain = (
        prompt
        | model
    )

    # Invoke the model with the complete message history
    response = chain.invoke(
        {
            "messages": messages,
        }
    )
    
    # Add AI response to the message history
    if response.content:
        messages.append(AIMessage(content=response.content))
    
    # Return the response or a fallback message
    return response.content if response.content else "Sorry, I don't know the answer to that."


def answer_general_query(user_question):
    vectorizer = TfidfVectorizer()
    questions = [item['question'] for item in qa_data]
    vectorized_data = vectorizer.fit_transform(questions)

    user_question_vectorized = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_question_vectorized, vectorized_data)
    most_similar_index = np.argmax(similarities)
    max_similarity = similarities[0][most_similar_index]

    # Set a threshold for similarity
    if max_similarity > 0.6:
        return qa_data[most_similar_index]['answer']
    else:
        return get_groq_response(user_question)

def normalize_text(text):
    # Convert text to lowercase and remove non-alphanumeric characters
    clean_text = ''.join(char.lower() for char in text if char.isalnum() or char.isspace())
    # Split text into words and remove possessive forms
    words = clean_text.split()
    normalized_words = []
    for word in words:
        # Remove possessive apostrophe if present
        word = word.rstrip("'s")
        normalized_words.append(word)
    return set(normalized_words)

# custom similarity matching function
def custom_similarity(text, query, exceptions=["Dr. ", "the", "I", "in", "to", "close", "i", "o" "O", "dr", "Dr", "dr."]):
    # Normalize text and query
    text_words = normalize_text(text)
    query_words = normalize_text(query)

    # Find matching sequences excluding exceptions
    matching_sequences = set()
    for word in text_words:
        if word in query_words and word not in exceptions:
            matching_sequences.add(word)

    # Return the count of matching sequences
    return len(matching_sequences)

# Function to find lecturer details using custom matching
def answer_lecturer_query(query):

    query = query.lower()
    max_score = 0
    best_match = None

    for index, row in lecturer_data.iterrows():
        text = f"{row['course']} {row['course_code']} {row['name']}".lower()
        score = custom_similarity(query, text)

        # Find the highest score
        if score > max_score:
            max_score = score
            best_match = row

    # Check if the query contains only one word
    if len(query.split()) == 1:
        return "I'm sorry, I need more information to assist you."

    elif max_score >= 1:
        # Process specific requests for phone number or office
        if "phone number" in query or "number" in query:
            if best_match['phone_number']:
                return f"Sure! {best_match['name']} the {best_match['course']} ({best_match['course_code']}) lecturer's phone number is {best_match['phone_number']}."
            else:
                return f"Sorry, I don't recall the phone number for that lecturer."
        elif "office" in query:
            if best_match['office']:
                return f"Sure thing! {best_match['name']} the {best_match['course']} ({best_match['course_code']}) lecturer's office is at {best_match['office']}."
            else:
                return f"Sorry, I seem to have forgotten the office of that lecturer."
        else:
            return f"{best_match['name']} is the {best_match['course']} ({best_match['course_code']}) lecturer."
    else:
        return answer_general_query(query)

# custom similarity matching function
def word_lookup(text, query, exceptions=["Study", "study"]):
    # Normalize text and query
    text_words = normalize_text(text)
    query_words = normalize_text(query)

    # Find matching sequences excluding exceptions
    matching_sequences = set()
    for word in text_words:
        if word in query_words and word not in exceptions:
            matching_sequences.add(word)

    # Return the count of matching sequences
    return len(matching_sequences)

def answer_doc_link_query(query):
    query = query.lower()
    max_score = 0
    best_match = None

    for index, row in doc_link_data.iterrows():
        text = f"{row['course']} {row['course_code']}".lower()
        score = word_lookup(query, text)

        # Find the highest score
        if score > max_score:
            max_score = score
            best_match = row

    # Check if the query contains only one word
    if len(query.split()) == 1:
        return "I'm sorry, I need more information to assist you."

    elif max_score >= 1:
        if "slide" in query or "past questions" in query or "slides" in query:
            if best_match['School files Link'] != "Unavailable":
                return f"Looking for slides and/or past questions for {best_match['course']})({best_match['course_code']})? This link should help you:  {best_match['School files Link']}"
            else:
                return f"Oops! Sorry, I can't find slides or past questions for that course."
        elif "study smarter" in query or "studysmarter" in query or "flash cards" in query or "today" in query:
            if best_match['Study Smarter Link'] != "Unavailable":
                return f"The Study Smarter study set for {best_match['course']})({best_match['course_code']}) contains the recent slides sent by the lecturer (and possibly flashcards, notes, and more learning resources). The link to the study set:  {best_match['Study Smarter Link']}"
            else:
                return f"Sorry, I can't find any study smarter study set for that course."
        else:
            return f"I didn't understand your question, try putting it differently?"
    else:
        return answer_general_query(query)


# Define function to determine intent
def get_intent(query):
    # Define keywords or phrases associated with each intent
    lecturer_keywords = ["lecturer", "lecturer's" "phone number", "number", "office"]
    doc_link_keywords = ["past questions", "pstq", "study materials", "flashcards", "studysmarter",
                         "study smarter", "slides", "slide", "pdf"]

    # Check for keywords in the query
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in lecturer_keywords):
        return "lecturer"
    elif any(keyword in query_lower for keyword in doc_link_keywords):
        return "doc_link"
    else:
        return "general"

# Define Gradio interface inputs and outputs
def handle_query(query):
    intent = get_intent(query)

    if intent == "lecturer":
        response = answer_lecturer_query(query)
    elif intent == 'doc_link':
        response =  answer_doc_link_query(query)
    else:
        response = answer_general_query(query)
    return response

# Path to your service account key file
SERVICE_ACCOUNT_FILE = 'anjibot-435113-7c0712b3fd28.json'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# Create credentials and build the service
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('sheets', 'v4', credentials=creds)

# The ID and range of the spreadsheet.
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")
RANGE_NAME = 'Sheet1!A1'

def append_to_sheet(user_query, bot_response):
    values = [
        [user_query, bot_response]
    ]
    body = {
        'values': values
    }
    result = service.spreadsheets().values().append(
        spreadsheetId=SPREADSHEET_ID,
        range=RANGE_NAME,
        valueInputOption='RAW',
        body=body
    ).execute()


# Streamlit app
def main():
    append_to_sheet("Hi", "Hello")
    # st.title("Ask Anjibot")
    
    # # User input field
    # user_query = st.text_input("Hello! I'm Anjibot, CS Group A AI Course Rep at your service ><. How can I help you?")
    
    # if user_query:
    #     response = handle_query(user_query)
    #     append_to_sheet(user_query, response)
    #     st.write(response)

if __name__ == "__main__":
    main()
