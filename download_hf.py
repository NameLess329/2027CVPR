import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
from urllib.parse import quote

import requests
from tqdm.auto import tqdm


HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
HF_REPO_TYPES = ("model", "dataset", "space")
CHUNK_SIZE = 1024 * 1024
REQUEST_TIMEOUT = 120


def build_headers(token: str | None = None) -> dict[str, str]:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def encode_repo_id(repo_id: str) -> str:
    return "/".join(quote(part, safe="") for part in repo_id.split("/"))


def build_repo_api_url(repo_id: str, repo_type: str | None = None, revision: str | None = None) -> str:
    encoded_repo = encode_repo_id(repo_id)
    if revision is None:
        if repo_type == "dataset":
            return f"{HF_MIRROR_ENDPOINT}/api/datasets/{encoded_repo}"
        if repo_type == "space":
            return f"{HF_MIRROR_ENDPOINT}/api/spaces/{encoded_repo}"
        return f"{HF_MIRROR_ENDPOINT}/api/models/{encoded_repo}"

    encoded_revision = quote(revision, safe="")
    if repo_type == "dataset":
        return f"{HF_MIRROR_ENDPOINT}/api/datasets/{encoded_repo}/tree/{encoded_revision}"
    if repo_type == "space":
        return f"{HF_MIRROR_ENDPOINT}/api/spaces/{encoded_repo}/tree/{encoded_revision}"
    return f"{HF_MIRROR_ENDPOINT}/api/models/{encoded_repo}/tree/{encoded_revision}"


def request_json(
    url: str,
    token: str | None = None,
    params: dict[str, str | int | bool] | None = None,
) -> requests.Response:
    response = requests.get(
        url,
        headers=build_headers(token),
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response


def resolve_repo_type(
    repo_id: str,
    repo_type: str,
    revision: str | None = None,
    token: str | None = None,
) -> str | None:
    if repo_type != "auto":
        candidate_types = (repo_type,)
    else:
        candidate_types = HF_REPO_TYPES

    errors: list[str] = []

    for candidate in candidate_types:
        try:
            url = build_repo_api_url(repo_id=repo_id, repo_type=None if candidate == "model" else candidate)
            request_json(url=url, token=token)
            if candidate == "dataset":
                return "dataset"
            if candidate == "space":
                return "space"
            return None
        except requests.HTTPError as exc:
            errors.append(f"{candidate}: {exc.response.status_code}")

    checked_types = ", ".join(candidate_types)
    error_text = "; ".join(errors) if errors else "no repository type matched"
    raise ValueError(
        f"Could not access repo '{repo_id}' on {HF_MIRROR_ENDPOINT}. "
        f"Checked repo types: {checked_types}. Details: {error_text}"
    )


def build_file_url(
    repo_id: str,
    file_path: str,
    repo_type: str | None = None,
    revision: str | None = None,
) -> str:
    revision = revision or "main"
    quoted_repo = "/".join(quote(part, safe="") for part in repo_id.split("/"))
    quoted_path = quote(file_path.strip("/\\"), safe="/")

    if repo_type == "dataset":
        return f"{HF_MIRROR_ENDPOINT}/datasets/{quoted_repo}/resolve/{revision}/{quoted_path}"
    if repo_type == "space":
        return f"{HF_MIRROR_ENDPOINT}/spaces/{quoted_repo}/resolve/{revision}/{quoted_path}"
    return f"{HF_MIRROR_ENDPOINT}/{quoted_repo}/resolve/{revision}/{quoted_path}"


def download_file_to_local(
    repo_id: str,
    file_path: str,
    local_dir: str,
    repo_type: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    force_download: bool = False,
    byte_progress_bar: tqdm | None = None,
) -> str:
    local_path = Path(local_dir) / Path(file_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists() and not force_download:
        return str(local_path)

    url = build_file_url(
        repo_id=repo_id,
        file_path=file_path,
        repo_type=repo_type,
        revision=revision,
    )
    response = requests.get(
        url,
        headers=build_headers(token),
        stream=True,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    expected_size = response.headers.get("Content-Length")
    file_progress_bar = None
    if byte_progress_bar is None:
        total = int(expected_size) if expected_size and expected_size.isdigit() else None
        file_progress_bar = tqdm(
            total=total,
            desc=f"Downloading {Path(file_path).name}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
        )

    with open(local_path, "wb") as file_obj:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                file_obj.write(chunk)
                if file_progress_bar is not None:
                    file_progress_bar.update(len(chunk))
                if byte_progress_bar is not None:
                    byte_progress_bar.update(len(chunk))

    if file_progress_bar is not None:
        file_progress_bar.close()

    return str(local_path)


def list_repo_files(
    repo_id: str,
    repo_type: str | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> list[str]:
    file_paths = []
    cursor = None
    tree_url = build_repo_api_url(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision or "main",
    )

    while True:
        params: dict[str, str | int | bool] = {
            "recursive": "true",
            "expand": "false",
            "limit": 1000,
        }
        if cursor:
            params["cursor"] = cursor

        response = request_json(
            url=tree_url,
            token=token,
            params=params,
        )
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(f"Unexpected tree response from {tree_url}: {type(payload).__name__}")

        for entry in payload:
            if entry.get("type") == "file" and entry.get("path"):
                file_paths.append(entry["path"])

        next_link = response.links.get("next", {})
        next_url = next_link.get("url")
        next_cursor = None
        if next_url and "cursor=" in next_url:
            next_cursor = next_url.split("cursor=", 1)[1]

        if not next_cursor:
            break
        cursor = next_cursor

    return file_paths


def download_single_file(
    repo_id: str,
    file_path: str,
    local_dir: str,
    repo_type: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    force_download: bool = False,
) -> str:
    file_path = file_path.strip("/\\")
    return download_file_to_local(
        repo_id=repo_id,
        file_path=file_path,
        local_dir=local_dir,
        repo_type=repo_type,
        revision=revision,
        token=token,
        force_download=force_download,
    )


def download_folder(
    repo_id: str,
    folder_path: str,
    local_dir: str,
    repo_type: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    force_download: bool = False,
    max_workers: int = 8,
) -> str:
    folder_path = folder_path.strip("/\\")
    prefix = f"{folder_path}/"
    repo_files = list_repo_files(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        token=token,
    )
    target_files = [file_path for file_path in repo_files if file_path == folder_path or file_path.startswith(prefix)]

    if not target_files:
        raise FileNotFoundError(f"No files found under folder: {folder_path}")

    files_progress_bar = tqdm(
        total=len(target_files),
        desc="Files",
        unit="file",
        dynamic_ncols=True,
    )
    bytes_progress_bar = tqdm(
        total=None,
        desc="Downloaded",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
    )

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    download_file_to_local,
                    repo_id,
                    file_path,
                    local_dir,
                    repo_type,
                    revision,
                    token,
                    force_download,
                    bytes_progress_bar,
                )
                for file_path in target_files
            ]
            for future in as_completed(futures):
                future.result()
                files_progress_bar.update(1)
    finally:
        files_progress_bar.close()
        bytes_progress_bar.close()

    return str(Path(local_dir) / folder_path)


def download_all(
    repo_id: str,
    local_dir: str,
    repo_type: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    force_download: bool = False,
    max_workers: int = 8,
) -> str:
    repo_files = list_repo_files(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        token=token,
    )

    files_progress_bar = tqdm(
        total=len(repo_files),
        desc="Files",
        unit="file",
        dynamic_ncols=True,
    )
    bytes_progress_bar = tqdm(
        total=None,
        desc="Downloaded",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
    )

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    download_file_to_local,
                    repo_id,
                    file_path,
                    local_dir,
                    repo_type,
                    revision,
                    token,
                    force_download,
                    bytes_progress_bar,
                )
                for file_path in repo_files
            ]
            for future in as_completed(futures):
                future.result()
                files_progress_bar.update(1)
    finally:
        files_progress_bar.close()
        bytes_progress_bar.close()

    return str(Path(local_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face files from hf-mirror.com."
    )
    parser.add_argument("--repo-id", required=True, help="Repo ID, for example: Qwen/Qwen-Image-Layered")
    parser.add_argument("--local-dir", required=True, help="Local download directory")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["file", "folder", "all"],
        help="Download mode: file, folder, or all",
    )
    parser.add_argument(
        "--path",
        help="Path inside the repo. Required for file/folder mode.",
    )
    parser.add_argument(
        "--repo-type",
        choices=["auto", "model", "dataset", "space"],
        default="auto",
        help="Repo type, default: auto",
    )
    parser.add_argument("--revision", help="Branch, tag, or commit id", default=None)
    parser.add_argument("--token", help="Access token for private repos", default=None)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if cached locally",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel workers for folder/all mode, default: 8",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode in {"file", "folder"} and not args.path:
        parser.error("--path is required when --mode is file or folder")

    repo_type = resolve_repo_type(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        token=args.token,
    )

    if args.mode == "file":
        local_path = download_single_file(
            repo_id=args.repo_id,
            file_path=args.path,
            local_dir=args.local_dir,
            repo_type=repo_type,
            revision=args.revision,
            token=args.token,
            force_download=args.force_download,
        )
    elif args.mode == "folder":
        local_path = download_folder(
            repo_id=args.repo_id,
            folder_path=args.path,
            local_dir=args.local_dir,
            repo_type=repo_type,
            revision=args.revision,
            token=args.token,
            force_download=args.force_download,
            max_workers=args.max_workers,
        )
    else:
        local_path = download_all(
            repo_id=args.repo_id,
            local_dir=args.local_dir,
            repo_type=repo_type,
            revision=args.revision,
            token=args.token,
            force_download=args.force_download,
            max_workers=args.max_workers,
        )

    print(f"Download finished. Local path: {local_path}")


if __name__ == "__main__":
    main()
