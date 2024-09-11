# Ask Anjibot 

## Anjibot - AI Course Representative for CS Department 400 Level Group A

Anjibot is a Streamlit-based AI chatbot designed to assist Computer Science 400-level students with questions about lecturers, documents, and general queries. Anjibot is powered by Langchain's Groq model (Llama3-8B) and utilizes custom datasets for lecturers and document links to provide personalized responses.

---

## Features

- **Lecturer Queries**: Provides lecturer contact information such as phone numbers and office locations. Even if you're not aware of the lecturer's name, you can still retriev info about their phone number/ office based on course code/ course name.
- **Document Queries**: Supplies links to slides, past questions, and study materials for various courses.
- **General Queries**: Handles general course or class-related questions using a TF-IDF and cosine similarity model.
- **Logs User Interactions**: Records user queries and bot responses into a Google Sheets document for future reference/ updates.

---

## How It Works

### Input Handling
Anjibot handles three types of queries:
1. **Lecturer Queries**: Users can ask for contact details, office information, or lecturer assignments based on the courses they teach.
2. **Document Link Queries**: Users can request slides, past questions, or study smarter links for specific courses.
3. **General Queries**: For general questions, the chatbot responds using a combination of a custom question-answering dataset and the Groq AI model.

### Dataset
The app uses:
- `lecturers.csv` for storing lecturer information (name, course, phone number, office).
- `docs_link.csv` for storing document links (slides, past questions, study smarter flashcards).
- `anjibot_data.json` for storing FAQs as question-answer pairs.

### Intent Recognition
Anjibot determines the intent of the user's query based on the presence of keywords related to lecturers or document links. If none of these keywords are detected, it classifies the query as general and either tries to answer based on the FAQs or passes it to the Groq model for an AI-generated response.

---

## Setup Instructions

### Prerequisites
1. **Python** (3.10 or higher)
2. **Streamlit** for the web interface
3. **Google Cloud API** credentials for Google Sheets logging
4. **Langchain** for the Groq model
5. **Pandas** for data manipulation
6. **scikit-learn** for text similarity
7. **dotenv** for managing environment variables

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/anjibot.git
    cd anjibot
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file in the project root and add your Groq API key and Google Sheets credentials:
    ```bash
    GROQ_API_KEY=<your_groq_api_key>
    
    # Google Sheets API credentials
    [google]
    type = "service_account"
    project_id = "<your_project_id>"
    private_key_id = "<your_private_key_id>"
    private_key = "<your_private_key>"
    client_email = "<your_client_email>"
    client_id = "<your_client_id>"
    auth_uri = "https://accounts.google.com/o/oauth2/auth"
    token_uri = "https://oauth2.googleapis.com/token"
    auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
    client_x509_cert_url = "<your_client_x509_cert_url>"
    
    [app]
    SPREADSHEET_ID = "<your_spreadsheet_id>"
    ```

4. Load your datasets (`lecturers.csv`, `anjibot_data.json` and `docs_link.csv`) into the `Datasets/` folder.

---

## Running the App

To start the Streamlit app, run the following command:
```bash
streamlit run app.py
```

---

## Usage

1. Navigate to the running Streamlit app in your browser.
2. Enter your query into the text input box (e.g., "Who is the lecturer for COSC401?" or "Where can I get slides for CSC405?").
3. Anjibot will respond with the requested information.
4. All queries and responses will be logged into the connected Google Sheet for future reference.

---

## Google Sheets Logging

User queries and chatbot responses are appended to a Google Sheets document specified by the `SPREADSHEET_ID` in the `.env` file. Ensure the correct credentials are used for accessing the Google Sheets API.

---

## Customization

- **Update Datasets**: Add new lecturers or update contact information by modifying `lecturers.csv`. Add more document links or update existing ones by editing `docs_link.csv`.
- **Add More Intents**: Update the `get_intent()` function to add more keywords or phrases to handle additional types of queries.

---

## Future Improvements

- Add more variations to intent detection logic for better response accuracy.
- Update datasets to reflect the current state of courses, lecturers, and documents.
- Integrate broader conversational abilities using more advanced models or larger datasets.

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

## Contact

Feel free to reach me at `anjolaajayi3@gmail.com`.