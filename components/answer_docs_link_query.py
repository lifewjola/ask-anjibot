# Required imports
import pandas as pd
from components.custom_matching import custom_similarity
from components.answer_general_query import answer_general_query

doc_link_data = pd.read_csv("Datasets/docs_link.csv").astype(str)

def answer_doc_link_query(query):
    query = query.lower()
    max_score = 0
    best_match = None

    for index, row in doc_link_data.iterrows():
        text = f"{row['course']} {row['course_code']}".lower()
        score = custom_similarity(query, text)

        # Find the highest score
        if score > max_score:
            max_score = score
            best_match = row

    course_code_prefix = ["cosc", "geds", "ged", "math", "stat", "stats", "seng", "itgy"]
    if any(word in query for word in course_code_prefix):
        if max_score < 2:
            return "I'm sorry, I couldn't find any information on the course you're asking about."

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