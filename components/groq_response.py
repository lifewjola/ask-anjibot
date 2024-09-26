# Required imports
from dotenv import load_dotenv
import os
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
            '''You're Anibot the AI course representative of Computer Science Department 400 lvl Group A. 
            You're always helpful and you answer your classmates questions only based on the provided information. 
            If you don't know the answer - just reply with an excuse that you don't know. 
            Keep your answers brief and to the point. 
            Be kind, jovial, funny, playful, and respectful.
            If it's a general info in natural language that isn't class related that you know the answer to, you can attempt answering.
            But anything school related or class related that you don't know the answer to, direct user to 'Anji' who's the human course rep of the department.''',
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

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

    if response.content:
        messages.append(AIMessage(content=response.content))
    
    # Return the response or a fallback message
    return response.content if response.content else "Sorry, I don't know the answer to that."
