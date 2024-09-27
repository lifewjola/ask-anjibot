# Required imports
import pandas as pd
import numpy as np
from components.custom_matching import custom_similarity 
from components.answer_general_query import answer_general_query

file_path = "Datasets/lecturers.csv"
lecturer_data = pd.read_csv(file_path).astype(str)

def answer_lecturer_query(query):

    query = query.lower()
    max_score = 0
    best_match = None

    for index, row in lecturer_data.iterrows():
        text = f"{row['course']} {row['course_code']} {row['name']}".lower()
        score = custom_similarity(query, text)

        if score > max_score:
            max_score = score
            best_match = row

    course_code_prefix = ["cosc", "geds", "ged", "math", "stat", "stats", "seng", "itgy"]
    if any(word in query for word in course_code_prefix):
        if max_score < 2:
            return "I'm sorry, I couldn't find any information on the lecturer you're asking about."

        elif max_score >= 1:
            if "number" in query and "office" in query:
                if best_match['phone_number']:
                    return f"Sure! {best_match['name']} the {best_match['course']} ({best_match['course_code']}) lecturer's phone number is {best_match['phone_number']} and the office is at {best_match['office']}."
            elif "no" in query or "number" in query:
                if best_match['phone_number']:
                    return f"Sure! {best_match['name']} the {best_match['course']} ({best_match['course_code']}) lecturer's phone number is {best_match['phone_number']}."
            elif "office" in query:
                if best_match['office'] == "No longer in Babcock":
                    return f"Sure thing! {best_match['name']} the {best_match['course']} ({best_match['course_code']}) lecturer's is {best_match['office']}."
                elif best_match['office']:
                    return f"Sure thing! {best_match['name']} the {best_match['course']} ({best_match['course_code']}) lecturer's office is at {best_match['office']}."         
            else:
                return f"{best_match['name']} is the {best_match['course']} ({best_match['course_code']}) lecturer."
        else:
            return answer_general_query(query)
        