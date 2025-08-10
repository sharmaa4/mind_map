import io
import os
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2 import service_account
from database import migrate_database

# Google Drive parent folder ID from secrets.toml
PARENT_FOLDER_ID = st.secrets["parent_folder_id"]

# Database file name
DB_FILENAME = "notes.db"

# Load GCP service account credentials from secrets.toml
def _get_drive_service():
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = service_account.Credentials.from_service_account_info(creds_dict)
    return build("drive", "v3", credentials=creds)

def upload_file_to_gdrive():
    """Uploads the local DB file to Google Drive (overwrites existing)."""
    service = _get_drive_service()

    # Search if file already exists in the GDrive folder
    query = f"'{PARENT_FOLDER_ID}' in parents and name = '{DB_FILENAME}'"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    if files:
        file_id = files[0]["id"]
        media = MediaFileUpload(DB_FILENAME, resumable=True)
        service.files().update(fileId=file_id, media_body=media).execute()
        st.success("✅ Database updated in Google Drive.")
    else:
        file_metadata = {"name": DB_FILENAME, "parents": [PARENT_FOLDER_ID]}
        media = MediaFileUpload(DB_FILENAME, resumable=True)
        service.files().create(body=file_metadata, media_body=media, fields="id").execute()
        st.success("✅ Database uploaded to Google Drive.")

def download_file_from_gdrive():
    """Downloads the DB file from Google Drive and runs migration."""
    service = _get_drive_service()

    query = f"'{PARENT_FOLDER_ID}' in parents and name = '{DB_FILENAME}'"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    if not files:
        st.warning("⚠️ No database file found in Google Drive.")
        return

    file_id = files[0]["id"]
    request = service.files().get_media(fileId=file_id)

    with io.FileIO(DB_FILENAME, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()

    st.success("✅ Database downloaded from Google Drive.")

    # Run migration to ensure schema is up-to-date
    migrate_database()
    st.info("🔄 Database schema upgraded (if needed).")

def sync_with_gdrive():
    """Handles the sync operation."""
    st.subheader("📂 Google Drive Sync")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬆️ Upload to Drive"):
            if os.path.exists(DB_FILENAME):
                upload_file_to_gdrive()
            else:
                st.error("❌ No local database file found to upload.")

    with col2:
        if st.button("⬇️ Download from Drive"):
            download_file_from_gdrive()

