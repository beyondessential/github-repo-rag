"""Voyage Batch API client (Files + Batches REST endpoints).

The voyageai Python package (0.5.0) has no batch support, so this talks to the
REST API directly. See specs/pipelines/batch-embedding.md.
"""

import json
import os
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator

_API = "https://api.voyageai.com/v1"
MODEL = "voyage-code-3"

# BL-004: Voyage caps a batch at 100K inputs; the file size cap is undocumented (OQ-001)
MAX_LINES = 100_000
MAX_BYTES = 100 * 1024 * 1024

IN_FLIGHT = {"validating", "in_progress", "finalizing", "cancelling"}
FAILED = {"failed", "cancelled"}


def _request(method: str, path: str, body: bytes | None = None, content_type: str | None = None) -> bytes:
    headers = {"Authorization": f"Bearer {os.environ['VOYAGE_API_KEY']}"}
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(f"{_API}{path}", data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"Voyage API {e.code} for {method} {path}: {detail}") from e


def build_input_files(texts: dict[str, str]) -> list[bytes]:
    """Encode {content_hash: text} as JSONL batch input files, split at the line/byte caps."""
    files: list[bytes] = []
    lines: list[bytes] = []
    size = 0
    for h, text in texts.items():
        # BL-003: one text per line, keyed by its content hash
        line = json.dumps({"custom_id": h, "body": {"input": [text]}}, ensure_ascii=False).encode("utf-8") + b"\n"
        if lines and (len(lines) >= MAX_LINES or size + len(line) > MAX_BYTES):
            files.append(b"".join(lines))
            lines, size = [], 0
        lines.append(line)
        size += len(line)
    if lines:
        files.append(b"".join(lines))
    return files


def upload_file(content: bytes, filename: str = "batch.jsonl") -> str:
    """Upload a JSONL batch input file and return its file id."""
    boundary = uuid.uuid4().hex
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/jsonl\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    resp = _request("POST", "/files", body, f"multipart/form-data; boundary={boundary}")
    return json.loads(resp)["id"]


def create_batch(input_file_id: str, metadata: dict[str, str] | None = None) -> dict:
    payload = {
        "endpoint": "/v1/embeddings",
        "completion_window": "12h",
        "request_params": {"model": MODEL, "input_type": "document"},
        "input_file_id": input_file_id,
    }
    if metadata:
        payload["metadata"] = metadata
    return json.loads(_request("POST", "/batches", json.dumps(payload).encode(), "application/json"))


def get_batch(batch_id: str) -> dict:
    return json.loads(_request("GET", f"/batches/{batch_id}"))


def iter_output_embeddings(output_file_id: str) -> Iterator[tuple[str, list[float]]]:
    """Yield (content_hash, embedding) for each successful line of a batch output file."""
    content = _request("GET", f"/files/{output_file_id}/content")
    for raw in content.splitlines():
        if not raw.strip():
            continue
        line = json.loads(raw)
        response = line.get("response") or {}
        if line.get("error") or response.get("status_code") != 200:
            continue
        data = response["body"]["data"]
        yield line["custom_id"], data[0]["embedding"]
