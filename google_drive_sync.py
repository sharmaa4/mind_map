# google_drive_sync.py
import streamlit as st
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os
import zipfile
import io
import shutil
import json

# --- Authentication ---
def authenticate_gdrive():
    """
    Handles Google authentication.
    - Uses Service Account credentials when deployed on Streamlit Cloud.
    - Uses local webserver auth for local development.
    """
    gauth = GoogleAuth()
    
    # Check if running on Streamlit Cloud
    if 'gcp_service_account' in st.secrets:
        print("Authenticating using Streamlit Secrets (Service Account)...")
        # Use service account credentials from st.secrets
        gauth.auth_method = 'service'
        gauth.credentials = gauth.get_credentials_from_service_account_info(
            st.secrets["gcp_service_account"]
        )
    else:
        print("Authenticating using local webserver method...")
        # Use local webserver authentication (requires client_secrets.json)
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

# --- Core Drive Operations (No changes needed below this line) ---
def get_folder_id(drive, folder_name, parent_folder_id='root'):
    """
    Finds a folder's ID by its name within a parent folder.
    If the folder doesn't exist, it creates it.
    """
    query = f"'{parent_folder_id}' in parents and title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    folder_list = drive.ListFile({'q': query}).GetList()
    
    if folder_list:
        return folder_list[0]['id']
    else:
        print(f"Folder '{folder_name}' not found in Google Drive. Creating it...")
        folder_metadata = {
            'title': folder_name, 
            'mimeType': 'application/vnd.google-apps.folder', 
            'parents': [{'id': parent_folder_id}]
        }
        folder = drive.CreateFile(folder_metadata)
        folder.Upload()
        return folder['id']

def sync_directory_to_drive(drive, local_path, drive_folder_name):
    """
    Zips a local directory and uploads it to a specific folder in Google Drive.
    It will overwrite the existing zip file with the same name.
    """
    if not os.path.isdir(local_path):
        print(f"Local path '{local_path}' does not exist. Skipping upload.")
        return

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
        print(f"No '{file_name}' found in Google Drive folder '{drive_folder_name}'. Starting with a fresh local directory.")
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

