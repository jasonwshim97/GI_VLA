"""annotation — review: consolidated research implementation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import os
import threading
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from pathlib import Path as _ProjectPath
from typing import Any
from urllib.parse import unquote, urlparse

PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


DECISIONS = frozenset({"KEEP", "FIX", "REMOVE"})


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ReviewStore:
    def __init__(self, *, review_root: Path, bbox_root: Path) -> None:
        self.review_root = review_root.resolve()
        self.bbox_root = bbox_root.resolve()
        self.manifest_path = self.review_root / "review_manifest.csv"
        self.decisions_path = self.review_root / "review_decisions.json"
        self.completion_path = self.review_root / "REVIEW_COMPLETE.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.manifest_sha256 = _sha256(self.manifest_path)
        self.lock = threading.RLock()
        self.items = self._load_items()
        self.item_index = {item["qc_index"]: item for item in self.items}
        self.decisions = self._load_decisions()

    def _load_items(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                source_path = (self.bbox_root / row["source_image"]).resolve()
                try:
                    source_path.relative_to(self.bbox_root)
                except ValueError as error:
                    raise ValueError(
                        f"source image escapes bbox root: {source_path}"
                    ) from error
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                center_jump = row.get("normalized_center_jump", "").strip()
                item = {
                    "qc_index": int(row["qc_index"]),
                    "model_split": row["model_split"],
                    "split_index": int(row["split_index"]),
                    "batch": int(row["batch"]),
                    "sample_id": row["sample_id"],
                    "image_id": int(row["image_id"]),
                    "episode": int(row["source_episode_index"]),
                    "frame": int(row["source_frame_index"]),
                    "source_image": row["source_image"],
                    "bbox_xyxy": [
                        float(row["bbox_x1"]),
                        float(row["bbox_y1"]),
                        float(row["bbox_x2"]),
                        float(row["bbox_y2"]),
                    ],
                    "partial": row["partial"].lower() == "true",
                    "occluded": row["occluded"].lower() == "true",
                    "blurred": row["blurred"].lower() == "true",
                    "interpolated_hint": row["interpolated_hint"].lower() == "true",
                    "normalized_center_jump": float(center_jump)
                    if center_jump
                    else None,
                    "width": 640,
                    "height": 480,
                }
                items.append(item)
        items.sort(key=lambda item: int(item["qc_index"]))
        expected_indices = list(range(1, len(items) + 1))
        indices = [int(item["qc_index"]) for item in items]
        if indices != expected_indices:
            raise ValueError("qc_index must be contiguous and start at 1")
        return items

    def _load_decisions(self) -> dict[str, dict[str, Any]]:
        if not self.decisions_path.is_file():
            return {}
        payload = json.loads(self.decisions_path.read_text(encoding="utf-8"))
        if payload.get("manifest_sha256") != self.manifest_sha256:
            raise ValueError(
                "review_decisions.json belongs to a different review manifest"
            )
        raw_decisions = payload.get("decisions", {})
        if not isinstance(raw_decisions, dict):
            raise ValueError("review decisions must be an object")
        decisions: dict[str, dict[str, Any]] = {}
        for raw_index, decision in raw_decisions.items():
            qc_index = int(raw_index)
            if qc_index not in self.item_index:
                raise ValueError(f"decision references unknown qc_index={qc_index}")
            self._validate_decision(self.item_index[qc_index], decision)
            decisions[str(qc_index)] = decision
        return decisions

    @staticmethod
    def _validate_decision(item: dict[str, Any], payload: dict[str, Any]) -> None:
        decision = payload.get("decision")
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {sorted(DECISIONS)}")
        bbox = payload.get("bbox_xyxy")
        if decision == "FIX":
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError("FIX requires bbox_xyxy with four values")
            x1, y1, x2, y2 = (float(value) for value in bbox)
            width, height = (int(item["width"]), int(item["height"]))
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"FIX bbox must be within 0..{width} x 0..{height}")
        elif bbox is not None:
            raise ValueError(f"{decision} must not contain bbox_xyxy")
        notes = payload.get("notes", "")
        if not isinstance(notes, str) or len(notes) > 2000:
            raise ValueError("notes must be a string of at most 2000 characters")

    def _summary_locked(self) -> dict[str, Any]:
        counts = Counter(
            (str(decision["decision"]) for decision in self.decisions.values())
        )
        split_reviewed: Counter[str] = Counter()
        for raw_index in self.decisions:
            split_reviewed[str(self.item_index[int(raw_index)]["model_split"])] += 1
        total = len(self.items)
        reviewed = len(self.decisions)
        return {
            "total": total,
            "reviewed": reviewed,
            "remaining": total - reviewed,
            "complete": reviewed == total,
            "decision_counts": {
                name: counts.get(name, 0) for name in sorted(DECISIONS)
            },
            "split_reviewed": dict(split_reviewed),
        }

    def summary(self) -> dict[str, Any]:
        with self.lock:
            return self._summary_locked()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "schema_version": 1,
                "manifest_sha256": self.manifest_sha256,
                "items": self.items,
                "decisions": self.decisions,
                "summary": self._summary_locked(),
            }

    def _write_locked(self) -> None:
        payload = {
            "schema_version": 1,
            "manifest_sha256": self.manifest_sha256,
            "updated_at": _utc_now(),
            "summary": self._summary_locked(),
            "decisions": self.decisions,
        }
        temporary = self.decisions_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.decisions_path)
        if self.completion_path.exists():
            self.completion_path.unlink()

    def set_decision(self, qc_index: int, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            item = self.item_index.get(qc_index)
            if item is None:
                raise KeyError(f"unknown qc_index={qc_index}")
            normalized = {
                "decision": str(payload.get("decision", "")).upper(),
                "bbox_xyxy": payload.get("bbox_xyxy"),
                "notes": str(payload.get("notes", "")),
                "updated_at": _utc_now(),
            }
            if normalized["decision"] == "FIX":
                normalized["bbox_xyxy"] = [
                    round(float(value), 3) for value in normalized["bbox_xyxy"]
                ]
            self._validate_decision(item, normalized)
            self.decisions[str(qc_index)] = normalized
            self._write_locked()
            return {"decision": normalized, "summary": self._summary_locked()}

    def clear_decision(self, qc_index: int) -> dict[str, Any]:
        with self.lock:
            if qc_index not in self.item_index:
                raise KeyError(f"unknown qc_index={qc_index}")
            self.decisions.pop(str(qc_index), None)
            self._write_locked()
            return {"summary": self._summary_locked()}

    def finalize(self) -> dict[str, Any]:
        with self.lock:
            summary = self._summary_locked()
            if not summary["complete"]:
                raise RuntimeError(
                    f"review is incomplete: {summary['remaining']} items remain"
                )
            payload = {
                "schema_version": 1,
                "manifest_sha256": self.manifest_sha256,
                "completed_at": _utc_now(),
                "summary": summary,
                "decisions_sha256": _sha256(self.decisions_path),
            }
            temporary = self.completion_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.completion_path)
            return payload

    def image_path(self, qc_index: int) -> Path:
        item = self.item_index.get(qc_index)
        if item is None:
            raise KeyError(f"unknown qc_index={qc_index}")
        path = (self.bbox_root / str(item["source_image"])).resolve()
        path.relative_to(self.bbox_root)
        return path


class ReviewHTTPServer(ThreadingHTTPServer):
    def __init__(
        self, address: tuple[str, int], *, store: ReviewStore, static_root: Path
    ) -> None:
        super().__init__(address, ReviewHandler)
        self.store = store
        self.static_root = static_root.resolve()


STATIC_FILES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/app.js": "app.js",
    "/styles.css": "styles.css",
}


class ReviewHandler(BaseHTTPRequestHandler):
    server: ReviewHTTPServer

    def _send_bytes(
        self,
        payload: bytes,
        *,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        cache_control: str = "no-store",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", cache_control)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(
            encoded, content_type="application/json; charset=utf-8", status=status
        )

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/health":
                self._send_json({"ok": True, "summary": self.server.store.summary()})
                return
            if path == "/api/state":
                self._send_json(self.server.store.snapshot())
                return
            if path.startswith("/api/image/"):
                qc_index = int(path.rsplit("/", 1)[-1])
                image_path = self.server.store.image_path(qc_index)
                content_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
                self._send_bytes(
                    image_path.read_bytes(),
                    content_type=content_type,
                    cache_control="private, max-age=86400",
                )
                return
            static_name = STATIC_FILES.get(path)
            if static_name is not None:
                static_path = self.server.static_root / static_name
                content_type = (
                    mimetypes.guess_type(static_name)[0] or "application/octet-stream"
                )
                if static_name.endswith((".html", ".js", ".css")):
                    content_type += "; charset=utf-8"
                self._send_bytes(static_path.read_bytes(), content_type=content_type)
                return
            self._send_error_json(HTTPStatus.NOT_FOUND, "not found")
        except (KeyError, ValueError, FileNotFoundError) as error:
            self._send_error_json(HTTPStatus.NOT_FOUND, str(error))
        except Exception as error:
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))

    def _read_json_body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length is required")
        length = int(raw_length)
        if length < 0 or length > 64 * 1024:
            raise ValueError("request body is too large")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json_body()
            if path == "/api/decision":
                qc_index = int(payload.pop("qc_index"))
                self._send_json(self.server.store.set_decision(qc_index, payload))
                return
            if path == "/api/clear":
                qc_index = int(payload["qc_index"])
                self._send_json(self.server.store.clear_decision(qc_index))
                return
            if path == "/api/finalize":
                self._send_json(self.server.store.finalize())
                return
            self._send_error_json(HTTPStatus.NOT_FOUND, "not found")
        except KeyError as error:
            self._send_error_json(HTTPStatus.NOT_FOUND, str(error))
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(error))
        except RuntimeError as error:
            self._send_error_json(HTTPStatus.CONFLICT, str(error))
        except Exception as error:
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-root",
        type=Path,
        default=Path("annotations/model_used_review"),
    )
    parser.add_argument("--bbox-root", type=Path, default=Path("bbox_vla"))
    parser.add_argument(
        "--static-root", type=Path, default=Path(__file__).with_name("web")
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    review_root = args.review_root.resolve()
    store = ReviewStore(review_root=review_root, bbox_root=args.bbox_root.resolve())
    server = ReviewHTTPServer(
        (args.host, args.port), store=store, static_root=args.static_root.resolve()
    )
    pid_path = review_root / "web_review_server.pid"
    pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    print(
        f"GT review server: http://{args.host}:{args.port} ({len(store.items)} items, {store.summary()['reviewed']} reviewed)",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if pid_path.exists() and pid_path.read_text(encoding="utf-8").strip() == str(
            os.getpid()
        ):
            pid_path.unlink()
    return 0


def cli(argv=None):
    from endo_bbox_vla.utils import run_cli

    return run_cli({"serve": main}, argv)


if __name__ == "__main__":
    raise SystemExit(cli())
