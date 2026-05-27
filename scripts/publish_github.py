from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


OWNER = "uers123"
REPO = "tuxiang-wenban"
DEFAULT_BRANCH = "main"
BASE_TAG = "v0.1.0"
COMMIT_MESSAGE = "Implement doc-textify MVP"
RELEASE_NAME = "doc-textify Windows executable"

INCLUDE_PATHS = [
    ".gitignore",
    "ENVIRONMENT.txt",
    "README.md",
    "doc_textify",
    "doc_textify_cli.py",
    "pyproject.toml",
    "requirements-dev.txt",
    "requirements.txt",
    "tests",
]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN is not set.", file=sys.stderr)
        return 1

    files = list_project_files(root)
    print(f"Preparing {len(files)} source files for GitHub upload.")

    head = api_json(token, "GET", f"/repos/{OWNER}/{REPO}/git/ref/heads/{DEFAULT_BRANCH}")
    base_commit_sha = head["object"]["sha"]
    base_commit = api_json(token, "GET", f"/repos/{OWNER}/{REPO}/git/commits/{base_commit_sha}")
    base_tree_sha = base_commit["tree"]["sha"]

    tree_entries = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        blob = api_json(
            token,
            "POST",
            f"/repos/{OWNER}/{REPO}/git/blobs",
            {"content": path.read_text(encoding="utf-8"), "encoding": "utf-8"},
        )
        tree_entries.append({"path": rel, "mode": "100644", "type": "blob", "sha": blob["sha"]})

    tree = api_json(
        token,
        "POST",
        f"/repos/{OWNER}/{REPO}/git/trees",
        {"base_tree": base_tree_sha, "tree": tree_entries},
    )
    commit = api_json(
        token,
        "POST",
        f"/repos/{OWNER}/{REPO}/git/commits",
        {"message": COMMIT_MESSAGE, "tree": tree["sha"], "parents": [base_commit_sha]},
    )
    api_json(
        token,
        "PATCH",
        f"/repos/{OWNER}/{REPO}/git/refs/heads/{DEFAULT_BRANCH}",
        {"sha": commit["sha"], "force": False},
    )
    print(f"Pushed commit {commit['sha']} to {OWNER}/{REPO}:{DEFAULT_BRANCH}.")

    tag = next_available_tag(token)
    release = api_json(
        token,
        "POST",
        f"/repos/{OWNER}/{REPO}/releases",
        {
            "tag_name": tag,
            "target_commitish": commit["sha"],
            "name": RELEASE_NAME,
            "body": "Windows executable for doc-textify. Tesseract remains an optional external OCR dependency.",
            "draft": False,
            "prerelease": False,
        },
    )
    print(f"Published release {tag}.")

    upload_asset(token, release["upload_url"], root / "dist" / "doc-textify.exe", "doc-textify-windows-x64.exe")
    upload_asset(token, release["upload_url"], root / "ENVIRONMENT.txt", "ENVIRONMENT.txt")
    print("Uploaded release assets.")
    print(release["html_url"])
    return 0


def list_project_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for include in INCLUDE_PATHS:
        path = root / include
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(item for item in path.rglob("*") if item.is_file() and "__pycache__" not in item.parts))
    return files


def next_available_tag(token: str) -> str:
    prefix = BASE_TAG.rsplit(".", 1)[0]
    patch = int(BASE_TAG.rsplit(".", 1)[1])
    while True:
        tag = f"{prefix}.{patch}"
        try:
            api_json(token, "GET", f"/repos/{OWNER}/{REPO}/releases/tags/{tag}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return tag
            raise
        patch += 1


def upload_asset(token: str, upload_url: str, path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    endpoint = upload_url.split("{", 1)[0]
    query = urllib.parse.urlencode({"name": name})
    data = path.read_bytes()
    content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
    request = urllib.request.Request(
        f"{endpoint}?{query}",
        data=data,
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": content_type,
            "Content-Length": str(len(data)),
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        response.read()


def api_json(token: str, method: str, path: str, payload: dict | None = None) -> dict:
    data = None
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        f"https://api.github.com{path}",
        data=data,
        method=method,
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
