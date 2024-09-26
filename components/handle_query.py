# Required imports
from components.answer_docs_link_query import answer_doc_link_query
from components.answer_general_query import answer_general_query
from components.answer_lecturer_query import answer_lecturer_query
import time

def get_intent(query):
    lecturer_keywords = ["who", "name", "lecturer", "lecturer's" "phone number", "no", "phone", "number", "office"]
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

# handle queries
def handle_query(query):
    intent = get_intent(query)

    if intent == "lecturer":
        response = answer_lecturer_query(query)
    elif intent == 'doc_link':
        response =  answer_doc_link_query(query)
    else:
        response = answer_general_query(query)
    for word in response.split():
        yield word + " "
        time.sleep(0.05)