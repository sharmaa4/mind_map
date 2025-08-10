# google_drive_sync.py

import streamlit as st
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
import os
import zipfile
import io
import shutil

# The top-level local path where the zip contents will be extracted.
# Use a consistent path that matches database.py's BASE_NOTES_DIR.
LOCAL_BASE_NOTES_DIR = os.path.join(os.getcwd(), "notes")


def authenticate_gdrive():
    """
    Handles Google authentication.
    - Uses Service Account credentials from Streamlit Secrets when deployed.
    - Falls back to interactive PyDrive auth when running locally.
    """
    try:
        # Try service account (expected when deployed)
        if "gdrive_service_account_json" in st.secrets:
            sa_json = st.secrets["gdrive_service_account_json"]
            # write to temp file for oauth lib that expects a file path
            import tempfile, json

            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            tf.write(json.dumps(sa_json).encode("utf-8"))
            tf.flush()
            tf.close()
            scopes = ["https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_name(tf.name, scopes)
            gauth = GoogleAuth()
            gauth.credentials = creds
            drive = GoogleDrive(gauth)
            return drive

        # Fallback: use typical PyDrive interactive auth
        gauth = GoogleAuth()
        # This will attempt to use local settings.yaml or run OAuth flow
        gauth.LocalWebserverAuth()
        drive = GoogleDrive(gauth)
        return drive
    except Exception as e:
        print(f"[google_drive_sync] Error authenticating Google Drive: {e}")
        raise


def sync_directory_from_drive(drive: GoogleDrive, local_path: str = LOCAL_BASE_NOTES_DIR, parent_folder_id: str = None) -> str:
    """
    Downloads <local_path>.zip from Google Drive (top-level folder id from secrets or root)
    and extracts it into local_path. Returns the absolute local_path where files are extracted.
    """
    if parent_folder_id is None:
        parent_folder_id = st.secrets.get("parent_folder_id", "root")

    # the expected Drive filename is <folder_name>.zip
    file_name = f"{os.path.basename(local_path)}.zip"

    query = f"'{parent_folder_id}' in parents and title='{file_name}' and trashed=false"
    try:
        file_list = drive.ListFile({"q": query}).GetList()
    except Exception as e:
        print(f"[google_drive_sync] Drive query failed: {e}")
        raise

    if not file_list:
        print(f"[google_drive_sync] No zip '{file_name}' found in the configured Drive folder.")
        return str(local_path)

    gfile = file_list[0]
    print(f"[google_drive_sync] Downloading '{file_name}' from Google Drive...")
    download_stream = gfile.GetContentIOBuffer()
    zip_buffer = io.BytesIO(download_stream.read())

    # remove existing local_path and extract fresh
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    os.makedirs(local_path, exist_ok=True)

    with zipfile.ZipFile(zip_buffer, "r") as zf:
        zf.extractall(local_path)

    abs_path = os.path.abspath(local_path)
    print(f"[google_drive_sync] Successfully synced and extracted to '{abs_path}'.")
    return abs_path

