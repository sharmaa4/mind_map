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
def get_folder_id(drive, folder_name):
    """
    Finds a folder's ID by its name within the designated parent folder.
    If it doesn't exist, it creates it there.
    """
    # The top-level folder where all app data is stored.
    # For a deployed app, this comes from secrets. For local, it's the root 'My Drive'.
    parent_id = st.secrets.get("parent_folder_id", "root")

    query = f"'{parent_id}' in parents and title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    folder_list = drive.ListFile({'q': query}).GetList()
    
    if folder_list:
        return folder_list[0]['id']
    else:
        print(f"Folder '{folder_name}' not found in parent. Creating it...")
        folder_metadata = {
            'title': folder_name, 
            'mimeType': 'application/vnd.google-apps.folder', 
            'parents': [{'id': parent_id}]
        }
        folder = drive.CreateFile(folder_metadata)
        folder.Upload()
        return folder['id']

def sync_directory_to_drive(drive, local_path, drive_folder_name):
    """
    Zips a local directory and uploads it to a specific folder in Google Drive.
    """
    if not os.path.isdir(local_path):
        print(f"Local path '{local_path}' does not exist. Skipping upload.")
        return

    # This function correctly finds or creates the subfolder (e.g., "MindMapApp_Data")
    # inside the main parent folder specified in your secrets.
    drive_folder_id = get_folder_id(drive, drive_folder_name)
    
    file_name = f"{os.path.basename(local_path)}.zip"
    
    query = f"'{drive_folder_id}' in parents and title='{file_name}' and trashed=false"
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
        gfile = drive.CreateFile({'title': file_name, 'parents': [{'id': drive_folder_id}]})
    
    gfile.content = zip_buffer
    gfile.Upload()
    print(f"Successfully synced '{local_path}' to Google Drive.")

def sync_directory_from_drive(drive, local_path, drive_folder_name):
    """
    Downloads and unzips a directory from Google Drive to a local path.
    """
    drive_folder_id = get_folder_id(drive, drive_folder_name)
    file_name = f"{os.path.basename(local_path)}.zip"

    query = f"'{drive_folder_id}' in parents and title='{file_name}' and trashed=false"
    file_list = drive.ListFile({'q': query}).GetList()

    if not file_list:
        print(f"No '{file_name}' found in Drive folder '{drive_folder_name}'. Starting with a fresh local directory.")
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

