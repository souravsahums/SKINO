"""Push results to Google Drive from a headless environment (Kaggle/Colab/CI).

Kaggle has no interactive OAuth flow, so this uses a **service account**, which
is the only fully scriptable option:

  1. Google Cloud Console -> create a project -> enable the *Google Drive API*.
  2. Create a Service Account -> Keys -> Add key -> JSON. Download it.
  3. In Google Drive, create the destination folder and **Share** it with the
     service account's email (``...@....iam.gserviceaccount.com``) as *Editor*.
     Without this the service account cannot see the folder.
  4. On Kaggle: Add-ons -> Secrets -> add ``GDRIVE_SA_JSON`` with the JSON
     contents, and ``GDRIVE_FOLDER_ID`` with the folder id from its URL.

If credentials are absent this falls back to writing a zip archive, which you
can download from the Kaggle output pane - so the pipeline never fails just
because Drive is not configured.

Usage:
    python -m track2.sync_drive --results-dir track2/results_paper
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def make_archive(results_dir: str, name: str = "skino_results") -> str:
    out = os.path.join(tempfile.gettempdir(), name)
    path = shutil.make_archive(out, "zip", results_dir)
    size = os.path.getsize(path) / 1e6
    print(f"[archive] {path}  ({size:.1f} MB)")
    return path


def _load_credentials():
    """Service-account JSON from a Kaggle secret, env var, or local file."""
    raw = None
    try:                                     # Kaggle secrets
        from kaggle_secrets import UserSecretsClient
        raw = UserSecretsClient().get_secret("GDRIVE_SA_JSON")
    except Exception:
        pass
    if not raw:
        raw = os.environ.get("GDRIVE_SA_JSON")
    if not raw and os.path.isfile("service_account.json"):
        raw = open("service_account.json").read()
    return json.loads(raw) if raw else None


def _folder_id():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GDRIVE_FOLDER_ID")
    except Exception:
        return os.environ.get("GDRIVE_FOLDER_ID")


def upload(path: str, folder_id: str, info: dict) -> str | None:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    creds = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/drive.file"])
    svc = build("drive", "v3", credentials=creds)
    meta = {"name": os.path.basename(path), "parents": [folder_id]}
    media = MediaFileUpload(path, resumable=True)
    f = svc.files().create(body=meta, media_body=media, fields="id,name").execute()
    print(f"[drive] uploaded {f['name']}  id={f['id']}")
    return f["id"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results_paper"))
    ap.add_argument("--name", default="skino_results")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.results_dir):
        raise SystemExit(f"no results dir: {args.results_dir}")
    archive = make_archive(args.results_dir, args.name)

    info, fid = _load_credentials(), _folder_id()
    if not info or not fid:
        print("[drive] credentials or folder id missing - keeping the local zip only.\n"
              "        set Kaggle secrets GDRIVE_SA_JSON and GDRIVE_FOLDER_ID to enable upload.")
        final = os.path.join("/kaggle/working" if os.path.isdir("/kaggle/working") else ".",
                             os.path.basename(archive))
        if os.path.abspath(final) != os.path.abspath(archive):
            shutil.copy(archive, final)
        print(f"[archive] available at {final}")
        return
    upload(archive, fid, info)


if __name__ == "__main__":
    main()
