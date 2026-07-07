import argparse
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from modelscope import dataset_snapshot_download, snapshot_download


MODELSCOPE_BASE_URL = "https://www.modelscope.cn"
MODELSCOPE_REPO_TYPES = ("model", "dataset")


def _repo_exists(repo_id: str, repo_type: str, timeout: int = 10) -> bool:
    owner, name = repo_id.split("/", 1)
    if repo_type == "model":
        url = f"{MODELSCOPE_BASE_URL}/models/{quote(owner)}/{quote(name)}"
    elif repo_type == "dataset":
        url = f"{MODELSCOPE_BASE_URL}/datasets/{quote(owner)}/{quote(name)}"
    else:
        return False

    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 400
    except (HTTPError, URLError, ValueError):
        return False


def resolve_repo_type(repo_id: str, repo_type: str) -> str:
    if "/" not in repo_id:
        raise ValueError("repo_id must be in the format 'owner/name'")

    candidate_types = (repo_type,) if repo_type != "auto" else MODELSCOPE_REPO_TYPES

    for candidate in candidate_types:
        if _repo_exists(repo_id=repo_id, repo_type=candidate):
            return candidate

    checked_types = ", ".join(candidate_types)
    raise ValueError(
        f"Could not access repo '{repo_id}' on {MODELSCOPE_BASE_URL}. "
        f"Checked repo types: {checked_types}"
    )


def _get_downloader(repo_type: str):
    if repo_type == "model":
        return snapshot_download
    if repo_type == "dataset":
        return dataset_snapshot_download
    raise ValueError("repo_type must be 'model' or 'dataset'")


def download_single_file(
    repo_id: str,
    file_path: str,
    local_dir: str,
    repo_type: str = "model",
    revision: str | None = None,
    local_files_only: bool = False,
) -> str:
    downloader = _get_downloader(repo_type)
    file_path = file_path.strip("/\\")

    return downloader(
        repo_id,
        revision=revision,
        local_dir=local_dir,
        allow_file_pattern=file_path,
        local_files_only=local_files_only,
    )


def download_folder(
    repo_id: str,
    folder_path: str,
    local_dir: str,
    repo_type: str = "model",
    revision: str | None = None,
    local_files_only: bool = False,
) -> str:
    downloader = _get_downloader(repo_type)
    folder_path = folder_path.strip("/\\")
    allow_file_pattern = f"{folder_path}/*"

    return downloader(
        repo_id,
        revision=revision,
        local_dir=local_dir,
        allow_file_pattern=allow_file_pattern,
        local_files_only=local_files_only,
    )


def download_all(
    repo_id: str,
    local_dir: str,
    repo_type: str = "model",
    revision: str | None = None,
    local_files_only: bool = False,
) -> str:
    downloader = _get_downloader(repo_type)

    return downloader(
        repo_id,
        revision=revision,
        local_dir=local_dir,
        local_files_only=local_files_only,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download ModelScope files with a unified CLI."
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
        choices=["auto", "model", "dataset"],
        default="auto",
        help="Repo type, default: auto",
    )
    parser.add_argument("--revision", help="Branch, tag, or commit id", default=None)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use local cache and do not access network",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode in {"file", "folder"} and not args.path:
        parser.error("--path is required when --mode is file or folder")

    resolved_repo_type = resolve_repo_type(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
    )

    if args.mode == "file":
        local_path = download_single_file(
            repo_id=args.repo_id,
            file_path=args.path,
            local_dir=args.local_dir,
            repo_type=resolved_repo_type,
            revision=args.revision,
            local_files_only=args.local_files_only,
        )
    elif args.mode == "folder":
        local_path = download_folder(
            repo_id=args.repo_id,
            folder_path=args.path,
            local_dir=args.local_dir,
            repo_type=resolved_repo_type,
            revision=args.revision,
            local_files_only=args.local_files_only,
        )
    else:
        local_path = download_all(
            repo_id=args.repo_id,
            local_dir=args.local_dir,
            repo_type=resolved_repo_type,
            revision=args.revision,
            local_files_only=args.local_files_only,
        )

    print(f"Download finished. Local path: {local_path}")


if __name__ == "__main__":
    main()
