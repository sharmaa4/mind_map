# google_drive_sync.py
import streamlit as st
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
import os
import zipfile
import io
import shutil
import json

# --- Authentication ---
def authenticate_gdrive():
    """
    Handles Google authentication.
    - Uses Service Account credentials from Streamlit Secrets when deployed.
    - Uses local webserver auth (OAuth) for local development.
    """
    gauth = GoogleAuth()
    scope = ["https://www.googleapis.com/auth/drive"]

    # Check if running on Streamlit Cloud and secrets are available
    if 'gcp_service_account' in st.secrets:
        print("Authenticating using Streamlit Secrets (Service Account)...")
        try:
            creds = ServiceAccountCredentials.from_json_keyfile_dict(
                st.secrets["gcp_service_account"], scope
            )
            gauth.credentials = creds
        except Exception as e:
            print(f"Service account authentication failed: {e}")
            raise
    else:
        # Fallback to local webserver authentication for local development
        print("Authenticating using local webserver method...")
        gauth.auth_method = 'local'
        gauth.LoadClientConfigFile("client_secrets.json")
        gauth.LoadCredentialsFile("mycreds.txt")

        if gauth.credentials is None:
            gauth.LocalWebserverAuth()
        elif gauth.access_token_expired:
            gauth.Refresh()
        else:
            gauth.Authorize()
        
        gauth.SaveCredentialsFile("mycreds.txt")
    
    return GoogleDrive(gauth)

# --- Core Drive Operations ---
def sync_directory_to_drive(drive, local_path):
    """
    Zips a local directory and uploads it directly into the parent folder 
    specified in Streamlit secrets.
    """
    if not os.path.isdir(local_path):
        print(f"Local path '{local_path}' does not exist. Skipping upload.")
        return

    # The single, top-level folder where all app data is stored.
    parent_folder_id = st.secrets.get("parent_folder_id", "root")
    
    file_name = f"{os.path.basename(local_path)}.zip"
    
    query = f"'{parent_folder_id}' in parents and title='{file_name}' and trashed=false"
    file_list = drive.ListFile({'q': query}).GetList()

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(local_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, local_path)
                zf.write(file_path, arcname)
    zip_buffer.seek(0)

    if file_list:
        gfile = file_list[0]
        print(f"Updating '{file_name}' in Google Drive...")
    else:
        print(f"Uploading new file '{file_name}' to Google Drive...")
        gfile = drive.CreateFile({'title': file_name, 'parents': [{'id': parent_folder_id}]})
    
    gfile.content = zip_buffer
    gfile.Upload()
    print(f"Successfully synced '{local_path}' to Google Drive.")

def sync_directory_from_drive(drive, local_path):
    """
    Downloads and unzips a directory from the parent folder in Google Drive.
    """
    parent_folder_id = st.secrets.get("parent_folder_id", "root")
    file_name = f"{os.path.basename(local_path)}.zip"

    query = f"'{parent_folder_id}' in parents and title='{file_name}' and trashed=false"
    file_list = drive.ListFile({'q': query}).GetList()

    if not file_list:
        print(f"No '{file_name}' found in the designated Drive folder. Starting with a fresh local directory.")
        os.makedirs(local_path, exist_ok=True)
        return False

    gfile = file_list[0]
    print(f"Downloading '{file_name}' from Google Drive...")
    
    download_stream = gfile.GetContentIOBuffer()
    zip_buffer = io.BytesIO(download_stream.read())
    
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    os.makedirs(local_path, exist_ok=True)
    
    with zipfile.ZipFile(zip_buffer, 'r') as zf:
        zf.extractall(local_path)
    print(f"Successfully synced and extracted to '{local_path}'.")
    return True

