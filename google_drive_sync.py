# google_drive_sync.py
import io
import os
import zipfile
import tempfile
import shutil
import streamlit as st
from pathlib import Path
from typing import Optional
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2 import service_account
from database import migrate_database

# Local paths
LOCAL_NOTES_DIR = Path.cwd() / "notes"
LOCAL_METADATA_DIR = LOCAL_NOTES_DIR / "metadata"
LOCAL_METADATA_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_DB_PATH = LOCAL_METADATA_DIR / "notes_database.db"
LOCAL_ZIP_NAME = "notes.zip"

# Drive config from secrets
PARENT_FOLDER_ID = st.secrets.get("parent_folder_id", None)

# Helper: convert AttrDict-like to plain dict
def _to_plain(obj):
    from collections.abc import Mapping
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj

def _get_drive_service():
    """
    Build a Google Drive service using the service account JSON stored in st.secrets.
    """
    # Try keys in common names
    raw_sa = None
    if "gcp_service_account" in st.secrets:
        raw_sa = st.secrets["gcp_service_account"]
    elif "gdrive_service_account_json" in st.secrets:
        raw_sa = st.secrets["gdrive_service_account_json"]
    else:
        raise RuntimeError("No service account found in st.secrets under 'gcp_service_account' or 'gdrive_service_account_json'.")

    creds_dict = _to_plain(raw_sa)
    try:
        creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=["https://www.googleapis.com/auth/drive"])
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return service
    except Exception as e:
        raise RuntimeError(f"Failed to construct drive service from secrets: {e}")

def _find_file(service, name: str) -> Optional[dict]:
    """
    Return the first file metadata dict for given name inside the configured parent folder.
    """
    if PARENT_FOLDER_ID is None:
        raise RuntimeError("parent_folder_id not set in st.secrets")
    q = f"name = '{name}' and '{PARENT_FOLDER_ID}' in parents and trashed=false"
    resp = service.files().list(q=q, spaces="drive", fields="files(id, name, modifiedTime)", pageSize=1).execute()
    items = resp.get("files", [])
    return items[0] if items else None

def _download_drive_file_by_id(service, file_id: str, dest_path: Path) -> bool:
    try:
        request = service.files().get_media(fileId=file_id)
        with open(dest_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
        return True
    except Exception as e:
        st.error(f"[google_drive_sync] download error: {e}")
        return False

def _upload_file(service, local_path: Path, drive_name: str) -> Optional[str]:
    """
    Upload or update file in the parent folder. Returns file id on success.
    """
    if not local_path.exists():
        st.error(f"[google_drive_sync] Local file not found: {local_path}")
        return None

    # find existing
    existing = None
    try:
        existing = _find_file(service, drive_name)
    except Exception as e:
        st.error(f"[google_drive_sync] Error looking up existing file: {e}")
        return None

    media = MediaFileUpload(str(local_path), resumable=True)
    try:
        if existing:
            file_id = existing["id"]
            service.files().update(fileId=file_id, media_body=media).execute()
            return file_id
        else:
            file_metadata = {"name": drive_name, "parents": [PARENT_FOLDER_ID]}
            new = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
            return new.get("id")
    except Exception as e:
        st.error(f"[google_drive_sync] upload error: {e}")
        return None

def _make_notes_zip(zip_path: Path) -> Path:
    """
    Create a zip of the LOCAL_NOTES_DIR at zip_path (overwrite if exists).
    """
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if LOCAL_NOTES_DIR.exists():
            for root, dirs, files in os.walk(LOCAL_NOTES_DIR):
                for fn in files:
                    full = Path(root) / fn
                    # store with relative path inside zip
                    zf.write(full, arcname=str(full.relative_to(LOCAL_NOTES_DIR.parent)))
    return zip_path

# --------------------------
# Public API expected by app
# --------------------------

def sync_from_gdrive() -> bool:
    """
    Download notes.zip (preferred) or DB file from Drive into the local notes folder / metadata.
    Runs migrate_database() after download/extract.
    Returns True on success, False on failure.
    """
    try:
        service = _get_drive_service()
    except Exception as e:
        st.error(f"[google_drive_sync] Auth failed: {e}")
        return False

    # 1) Try notes.zip
    try:
        item = _find_file(service, LOCAL_ZIP_NAME)
        if item:
            st.info(f"[google_drive_sync] Found '{LOCAL_ZIP_NAME}' on Drive. Downloading and extracting...")
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            tmpf.close()
            ok = _download_drive_file_by_id(service, item["id"], Path(tmpf.name))
            if not ok:
                st.error("[google_drive_sync] Failed to download zip.")
                return False
            # remove existing local notes folder and extract
            if LOCAL_NOTES_DIR.exists():
                shutil.rmtree(LOCAL_NOTES_DIR)
            LOCAL_NOTES_DIR.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(tmpf.name, "r") as zf:
                zf.extractall(path=str(LOCAL_NOTES_DIR.parent))
            os.unlink(tmpf.name)
            st.success("[google_drive_sync] notes.zip downloaded & extracted.")
            # run migration in case DB inside zip is old
            migrate_database()
            return True
    except Exception as e:
        st.warning(f"[google_drive_sync] notes.zip attempt failed (continuing): {e}")

    # 2) Try database files by a set of common names
    candidate_names = ["notes_database.db", "notes.db", "notes_database.sqlite", "notes_database.sqlite3"]
    for name in candidate_names:
        try:
            item = _find_file(service, name)
            if item:
                st.info(f"[google_drive_sync] Found DB '{name}' on Drive. Downloading...")
                ok = _download_drive_file_by_id(service, item["id"], LOCAL_DB_PATH)
                if not ok:
                    st.error("[google_drive_sync] Failed to download DB.")
                    return False
                st.success(f"[google_drive_sync] {name} downloaded to {LOCAL_DB_PATH}.")
                migrate_database()
                return True
        except Exception as e:
            st.warning(f"[google_drive_sync] Checking for {name} failed (continuing): {e}")

    st.warning("[google_drive_sync] No notes.zip or database file found in Drive folder.")
    return False

def sync_to_gdrive() -> bool:
    """
    Upload local DB and a zip of notes to Drive (creates or updates).
    Returns True on success (at least one upload), False otherwise.
    """
    try:
        service = _get_drive_service()
    except Exception as e:
        st.error(f"[google_drive_sync] Auth failed: {e}")
        return False

    success_any = False

    # Upload DB if present
    if LOCAL_DB_PATH.exists():
        st.info(f"[google_drive_sync] Uploading DB {LOCAL_DB_PATH.name} to Drive...")
        uploaded_id = _upload_file(service, LOCAL_DB_PATH, LOCAL_DB_PATH.name)
        if uploaded_id:
            st.success(f"[google_drive_sync] DB uploaded/updated (id={uploaded_id}).")
            success_any = True
    else:
        st.warning(f"[google_drive_sync] Local DB not found at {LOCAL_DB_PATH}. Skipping DB upload.")

    # Create and upload zip of notes folder
    tmp_zip = Path(tempfile.gettempdir()) / LOCAL_ZIP_NAME
    try:
        _make_notes_zip(tmp_zip)
        st.info(f"[google_drive_sync] Uploading notes.zip to Drive...")
        uploaded_id = _upload_file(service, tmp_zip, tmp_zip.name)
        if uploaded_id:
            st.success(f"[google_drive_sync] notes.zip uploaded/updated (id={uploaded_id}).")
            success_any = True
        try:
            tmp_zip.unlink()
        except Exception:
            pass
    except Exception as e:
        st.warning(f"[google_drive_sync] Could not create/upload notes.zip: {e}")

    return success_any

# Backwards-compatible aliases (some parts of app might call these)
download_notes_zip = sync_from_gdrive
upload_notes_zip = sync_to_gdrive

