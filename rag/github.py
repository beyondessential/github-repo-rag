"""GitHub API and clone helpers used by scripts/ingest.py and scripts/sync.py."""

import base64
import json
import os
import subprocess
import urllib.error
import urllib.request

# GitHub's compare API limits — beyond these the diff may be truncated.
_MAX_COMMITS = 250
_MAX_FILES = 300


def github_api(path: str, token: str | None) -> dict:
    """Make a GitHub API GET request and return parsed JSON."""
    url = f"https://api.github.com{path}"
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"GitHub API {e.code} for {path}: {body}") from e


def parse_repo_url(repo_url: str) -> tuple[str, str]:
    """Parse 'https://github.com/owner/repo' → ('owner', 'repo')."""
    parts = repo_url.rstrip("/").split("/")
    return parts[-2], parts[-1].removesuffix(".git")


def clone_env(repo_url: str, token: str | None) -> dict[str, str]:
    """Environment for `git clone` that authenticates to github.com with the token.

    The token travels as an HTTP header set through GIT_CONFIG_* variables, so it never
    appears in the command line, git's own output, or a CalledProcessError message.
    """
    env = os.environ.copy()
    if not token or not repo_url.startswith("https://github.com/"):
        return env
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    n = int(env.get("GIT_CONFIG_COUNT", "0"))
    env["GIT_CONFIG_COUNT"] = str(n + 1)
    env[f"GIT_CONFIG_KEY_{n}"] = "http.https://github.com/.extraheader"
    env[f"GIT_CONFIG_VALUE_{n}"] = f"AUTHORIZATION: basic {basic}"
    return env


def shallow_clone(repo_url: str, dest: str, token: str | None, ref: str | None = None) -> None:
    """Shallow-clone a repo (optionally at a branch or tag), authenticating private repos."""
    cmd = ["git", "clone", "--depth=1"]
    if ref:
        cmd += ["--branch", ref]
    subprocess.run([*cmd, repo_url, dest], check=True, env=clone_env(repo_url, token))


def get_latest_sha(owner: str, repo: str, token: str | None) -> str:
    """Return the SHA of the latest commit on the default branch."""
    data = github_api(f"/repos/{owner}/{repo}/commits?per_page=1", token)
    return data[0]["sha"]


def get_changed_files(
    owner: str,
    repo: str,
    base_sha: str,
    head_sha: str,
    token: str | None,
) -> tuple[list[str], list[str], bool]:
    """Compare two commits and return (changed_files, deleted_files, too_large).

    too_large is True when the diff exceeds GitHub's compare limits; the caller
    should fall back to a full reindex in that case.

    changed_files includes added, modified, renamed (new path), and copied files.
    deleted_files includes removed files and the old paths of renames.
    """
    data = github_api(f"/repos/{owner}/{repo}/compare/{base_sha}...{head_sha}", token)

    if data.get("ahead_by", 0) > _MAX_COMMITS or len(data.get("files", [])) >= _MAX_FILES:
        return [], [], True

    changed: list[str] = []
    deleted: list[str] = []
    for f in data.get("files", []):
        status = f["status"]
        if status == "removed":
            deleted.append(f["filename"])
        elif status == "renamed":
            deleted.append(f["previous_filename"])
            changed.append(f["filename"])
        else:  # added, modified, changed, copied
            changed.append(f["filename"])

    return changed, deleted, False
