from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from urllib import error, request


@dataclass
class VisionServerClient:
    host: str = "127.0.0.1"
    port: int = 8000
    timeout_sec: float = 2.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _build_url(self, path: str) -> str:
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{self.base_url}{path}"

    def _read_json(self, response) -> dict:
        charset = response.headers.get_content_charset() or "utf-8"
        payload = response.read().decode(charset)
        return json.loads(payload) if payload else {}

    def _request_json(
        self,
        path: str,
        method: str = "GET",
        json_body: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict:
        body = None
        request_headers = dict(headers or {})
        if json_body is not None:
            body = json.dumps(json_body).encode("utf-8")
            request_headers["Content-Type"] = "application/json"

        req = request.Request(
            url=self._build_url(path),
            data=body,
            headers=request_headers,
            method=method,
        )
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as response:
                return self._read_json(response)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} failed: {exc.code} {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"{method} {path} failed: {exc.reason}") from exc

    def check_health(self) -> dict:
        return self._request_json("/health")

    def get_command(self) -> list[str]:
        payload = self._request_json("/command")
        route = payload.get("route", [])
        if not isinstance(route, list):
            raise RuntimeError("GET /command returned invalid route payload.")
        return [str(node) for node in route]

    def update_command(self, route: list[str]) -> dict:
        return self._request_json("/command", method="POST", json_body={"route": route})

    def analyze_image_bytes(
        self,
        image_bytes: bytes,
        filename: str = "frame.jpg",
        field_name: str = "file",
        content_type: str = "image/jpeg",
    ) -> dict:
        boundary = f"----VisionAgvBoundary{uuid.uuid4().hex}"
        body = b"".join(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{field_name}"; '
                    f'filename="{filename}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
                image_bytes,
                b"\r\n",
                f"--{boundary}--\r\n".encode("utf-8"),
            ]
        )
        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        }
        req = request.Request(
            url=self._build_url("/analyze"),
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as response:
                return self._read_json(response)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"POST /analyze failed: {exc.code} {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"POST /analyze failed: {exc.reason}") from exc
