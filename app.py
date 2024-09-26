'''
TO-DO
- Add more intents variation using list/ dictionaries to docs link logic functions
- Improve the doc_link function
- Add more stop words to custom similarity function
- More testing 
- Improve contextual memory functionality 
    - (should be able to recall previous chat for non groq queries )
'''

# Required imports
import streamlit as st
from components.anjibot_logging import append_to_sheet
from components.handle_query import handle_query


def main():
    st.title("Ask Anjibot")

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
            response = st.write_stream(handle_query(prompt))
        st.session_state.messages.append({"role": "assistant", "content": response})

        append_to_sheet(prompt, response)

if __name__ == "__main__":
    main()

    