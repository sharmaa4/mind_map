# google_drive_sync.py
"""
Google Drive sync helpers that support service-account credentials stored in Streamlit secrets.
Accepts service account under st.secrets["gcp_service_account"] (matches your .toml).
Falls back to st.secrets["gdrive_service_account_json"] (legacy) or interactive LocalWebserverAuth()
if a client_secrets.json file is present.
"""

import streamlit as st
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
import os
import zipfile
import io
import shutil
import tempfile
import json

# The top-level local path where the zip contents will be extracted.
LOCAL_BASE_NOTES_DIR = os.path.join(os.getcwd(), "notes")


def authenticate_gdrive():
    """
    Authenticate to Google Drive.

    Priority:
      1. Service account JSON provided in st.secrets["gcp_service_account"] (preferred)
      2. Raw service account JSON in st.secrets["gdrive_service_account_json"] (legacy)
      3. Local interactive OAuth via client_secrets.json + LocalWebserverAuth()

    Returns:
      GoogleDrive instance on success, raises Exception on failure.
    """
    scopes = ["https://www.googleapis.com/auth/drive"]

    def _to_plain(obj):
        """
        Recursively convert AttrDict-like or mapping objects into plain dicts,
        leaving primitive values untouched.
        """
        from collections.abc import Mapping

        if isinstance(obj, Mapping):
            return {k: _to_plain(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_plain(v) for v in obj]
        return obj

    # 1) Service account from st.secrets["gcp_service_account"]
    if "gcp_service_account" in st.secrets:
        raw_sa = st.secrets["gcp_service_account"]
        try:
            sa_info = _to_plain(raw_sa)
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            tf.write(json.dumps(sa_info, ensure_ascii=False).encode("utf-8"))
            tf.flush()
            tf.close()

            creds = ServiceAccountCredentials.from_json_keyfile_name(tf.name, scopes)
            gauth = GoogleAuth()
            gauth.credentials = creds
            drive = GoogleDrive(gauth)

            try:
                os.unlink(tf.name)
            except Exception:
                pass

            return drive
        except Exception as e:
            try:
                os.unlink(tf.name)
            except Exception:
                pass
            raise RuntimeError(f"[google_drive_sync] Service-account auth (gcp_service_account) failed: {e}")

    # 2) Older key name compatibility
    if "gdrive_service_account_json" in st.secrets:
        raw_sa = st.secrets["gdrive_service_account_json"]
        try:
            sa_info = _to_plain(raw_sa)
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            tf.write(json.dumps(sa_info, ensure_ascii=False).encode("utf-8"))
            tf.flush()
            tf.close()

            creds = ServiceAccountCredentials.from_json_keyfile_name(tf.name, scopes)
            gauth = GoogleAuth()
            gauth.credentials = creds
            drive = GoogleDrive(gauth)

            try:
                os.unlink(tf.name)
            except Exception:
                pass

            return drive
        except Exception as e:
            try:
                os.unlink(tf.name)
            except Exception:
                pass
            raise RuntimeError(f"[google_drive_sync] Service-account auth (gdrive_service_account_json) failed: {e}")

    # 3) Local interactive auth using client_secrets.json (fallback)
    try:
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        drive = GoogleDrive(gauth)
        return drive
    except Exception as e:
        raise RuntimeError(f"[google_drive_sync] Interactive Drive auth failed or client_secrets.json missing: {e}")


def sync_directory_from_drive(drive: GoogleDrive, local_path: str = LOCAL_BASE_NOTES_DIR, parent_folder_id: str = None) -> str:
    """
    Downloads <local_path>.zip from Google Drive (top-level folder id from secrets or root)
    and extracts it into local_path. Returns the absolute local_path where files are extracted.
    """
    if parent_folder_id is None:
        parent_folder_id = st.secrets.get("parent_folder_id", "root")

    file_name = f"{os.path.basename(local_path)}.zip"
    query = f"'{parent_folder_id}' in parents and title='{file_name}' and trashed=false"
    try:
        file_list = drive.ListFile({"q": query}).GetList()
    except Exception as e:
        raise RuntimeError(f"[google_drive_sync] Drive listing failed: {e}")

    if not file_list:
        print(f"[google_drive_sync] No zip '{file_name}' found in Drive folder {parent_folder_id}.")
        return os.path.abspath(local_path)

    gfile = file_list[0]
    print(f"[google_drive_sync] Downloading '{file_name}' from Google Drive (id={gfile.get('id')})...")
    download_stream = gfile.GetContentIOBuffer()
    zip_buffer = io.BytesIO(download_stream.read())

    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    os.makedirs(local_path, exist_ok=True)

    with zipfile.ZipFile(zip_buffer, "r") as zf:
        zf.extractall(local_path)

    abs_path = os.path.abspath(local_path)
    print(f"[google_drive_sync] Successfully synced and extracted to '{abs_path}'.")
    return abs_path

