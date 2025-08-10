import io
import os
import json
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2 import service_account

# ---- Google Drive API Setup ----
def get_gdrive_service():
    """Authenticate using service account credentials from Streamlit secrets."""
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        service = build("drive", "v3", credentials=creds)
        return service
    except Exception as e:
        st.error(f"[google_drive_sync] Service-account auth (gcp_service_account) failed: {e}")
        return None

# ---- Download from Google Drive ----
def download_file_from_gdrive(file_id: str, destination_path: str):
    """Download a file from Google Drive by file_id."""
    service = get_gdrive_service()
    if not service:
        return False
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(destination_path, "wb")
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                st.write(f"Download {int(status.progress() * 100)}%.")
        return True
    except Exception as e:
        st.error(f"[google_drive_sync] Download failed: {e}")
        return False

# ---- Upload to Google Drive ----
def upload_file_to_gdrive(local_path: str, parent_folder_id: str):
    """Upload a file to Google Drive into a specific folder."""
    service = get_gdrive_service()
    if not service:
        return None
    try:
        file_metadata = {
            "name": os.path.basename(local_path),
            "parents": [parent_folder_id]
        }
        media = MediaFileUpload(local_path, resumable=True)
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id"
        ).execute()
        return file.get("id")
    except Exception as e:
        st.error(f"[google_drive_sync] Upload failed: {e}")
        return None

# ---- Sync both ways ----
def sync_with_gdrive(local_file_path: str, parent_folder_id: str):
    """
    Sync: download latest from Drive if exists, then upload local version.
    """
    service = get_gdrive_service()
    if not service:
        return False

    # 1. Try downloading the latest
    try:
        results = service.files().list(
            q=f"'{parent_folder_id}' in parents and name='{os.path.basename(local_file_path)}' and trashed=false",
            spaces="drive",
            fields="files(id, name, modifiedTime)",
            orderBy="modifiedTime desc",
            pageSize=1
        ).execute()

        items = results.get("files", [])
        if items:
            file_id = items[0]["id"]
            st.write(f"Downloading latest '{os.path.basename(local_file_path)}' from Google Drive...")
            download_file_from_gdrive(file_id, local_file_path)

    except Exception as e:
        st.error(f"[google_drive_sync] Error fetching latest file: {e}")

    # 2. Upload the local file
    try:
        st.write(f"Uploading '{os.path.basename(local_file_path)}' to Google Drive...")
        upload_file_to_gdrive(local_file_path, parent_folder_id)
        return True
    except Exception as e:
        st.error(f"[google_drive_sync] Sync upload failed: {e}")
        return False

# ---- Backward compatibility aliases ----
sync_from_gdrive = download_file_from_gdrive
sync_to_gdrive = upload_file_to_gdrive

