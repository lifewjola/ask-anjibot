# Required imports
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build

# streamlit secrets
secrets = st.secrets["google"]
creds_info = {
    "type": secrets["type"],
    "project_id": secrets["project_id"],
    "private_key_id": secrets["private_key_id"],
    "private_key": secrets["private_key"].replace('\\n', '\n'),
    "client_email": secrets["client_email"],
    "client_id": secrets["client_id"],
    "auth_uri": secrets["auth_uri"],
    "token_uri": secrets["token_uri"],
    "auth_provider_x509_cert_url": secrets["auth_provider_x509_cert_url"],
    "client_x509_cert_url": secrets["client_x509_cert_url"]
}

# Create credentials and build the service for spreadsheet login
creds = service_account.Credentials.from_service_account_info(creds_info, scopes=['https://www.googleapis.com/auth/spreadsheets'])
service = build('sheets', 'v4', credentials=creds)

# The ID and range of the spreadsheet.
SPREADSHEET_ID = st.secrets["app"]["SPREADSHEET_ID"]
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