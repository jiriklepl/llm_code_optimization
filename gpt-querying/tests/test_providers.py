from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import urllib.error


GPT_QUERYING_DIR = Path(__file__).resolve().parents[1]
MAIN_PATH = GPT_QUERYING_DIR / "main.py"
RUN_BATCH_PATH = GPT_QUERYING_DIR / "run_batch.py"


def load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


MAIN_MODULE = load_module("gpt_querying_main_test", MAIN_PATH)

RUN_BATCH_MODULE = load_module("gpt_querying_run_batch_test", RUN_BATCH_PATH)


class FakeHTTPResponse:
    def __init__(self, status_code: int, payload: dict[str, object], request_id: str) -> None:
        self._status_code = status_code
        self._payload = json.dumps(payload).encode("utf-8")
        self.headers = {"x-request-id": request_id}

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def getcode(self) -> int:
        return self._status_code

    def read(self) -> bytes:
        return self._payload


class ProviderTests(unittest.TestCase):
    def make_request_line(self, custom_id: str, content: str, temperature: float = 0.7) -> dict[str, object]:
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "google/gemma-4-31b-it",
                "messages": [{"role": "user", "content": content}],
                "temperature": temperature,
            },
        }

    def test_gemma_request_shape(self) -> None:
        provider = MAIN_MODULE.build_request_provider("gemma")
        request = provider.build_request([{"role": "user", "content": "hello"}])

        self.assertEqual(request["model"], MAIN_MODULE.DEFAULT_GEMMA_MODEL)
        self.assertEqual(request["temperature"], MAIN_MODULE.DEFAULT_GEMMA_TEMPERATURE)
        self.assertNotIn("reasoning_effort", request)
        self.assertNotIn("verbosity", request)

    def test_gpt51_request_shape_unchanged(self) -> None:
        provider = MAIN_MODULE.build_request_provider("gpt51")
        request = provider.build_request([{"role": "user", "content": "hello"}])

        self.assertEqual(request["model"], MAIN_MODULE.DEFAULT_GPT51_MODEL)
        self.assertEqual(request["reasoning_effort"], "high")
        self.assertEqual(request["verbosity"], "medium")
        self.assertNotIn("temperature", request)

    def test_gemma_request_execution_and_parse(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            request_file = tmp_path / "requests" / "gemma" / "translation" / "sample.jsonl"
            request_file.parent.mkdir(parents=True, exist_ok=True)
            request_file.write_text(json.dumps(self.make_request_line("hello-01", "hello")) + "\n", encoding="utf-8")

            responses_dir = tmp_path / "responses" / "gemma"
            captured_requests: list[tuple[str, dict[str, object]]] = []

            def fake_urlopen(request):
                payload = json.loads(request.data.decode("utf-8"))
                captured_requests.append((request.full_url, payload))
                return FakeHTTPResponse(
                    200,
                    {
                        "id": "chatcmpl-hello",
                        "object": "chat.completion",
                        "created": 0,
                        "model": payload["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "```cpp\nint answer = 42;\n```",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 2,
                            "total_tokens": 3,
                        },
                    },
                    "req-hello",
                )

            with patch.object(RUN_BATCH_MODULE.urllib.request, "urlopen", side_effect=fake_urlopen):
                RUN_BATCH_MODULE.submit_gemma_request(
                    jsonl_path=request_file,
                    responses_dir=responses_dir,
                    api_base="http://bw01:8000/v1",
                    api_key=None,
                    provider="gemma",
                )

            self.assertEqual(captured_requests[0][0], "http://bw01:8000/v1/chat/completions")
            self.assertEqual(captured_requests[0][1]["temperature"], 0.7)

            output_file = responses_dir / "translation" / "sample.jsonl"
            response_line = json.loads(output_file.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(response_line["custom_id"], "hello-01")
            self.assertEqual(response_line["response"]["status_code"], 200)
            self.assertEqual(response_line["response"]["request_id"], "req-hello")
            self.assertEqual(
                response_line["response"]["body"]["choices"][0]["message"]["content"],
                "```cpp\nint answer = 42;\n```",
            )

            provider = MAIN_MODULE.build_request_provider("gemma")
            parsed_dir = tmp_path / "parsed"
            MAIN_MODULE.parse_batch_output(output_file, parsed_dir, provider=provider)
            self.assertEqual((parsed_dir / "hello-01" / "code.cpp").read_text(encoding="utf-8"), "int answer = 42;")

    def test_gemma_request_failures_are_logged_and_nonzero(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            request_file = tmp_path / "requests" / "gemma" / "optimization" / "sample.jsonl"
            request_file.parent.mkdir(parents=True, exist_ok=True)
            request_file.write_text(
                "\n".join(
                    [
                        json.dumps(self.make_request_line("ok-01", "ok")),
                        json.dumps(self.make_request_line("fail-01", "fail")),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            responses_dir = tmp_path / "responses" / "gemma"
            request_ids: list[str] = []

            def fake_urlopen(request):
                payload = json.loads(request.data.decode("utf-8"))
                request_ids.append(payload["messages"][-1]["content"])
                if payload["messages"][-1]["content"] == "fail":
                    error_payload = io.BytesIO(json.dumps({"error": {"message": "forced failure"}}).encode("utf-8"))
                    raise urllib.error.HTTPError(
                        request.full_url,
                        500,
                        "Internal Server Error",
                        {"x-request-id": "req-fail"},
                        error_payload,
                    )

                return FakeHTTPResponse(
                    200,
                    {
                        "id": "chatcmpl-ok",
                        "object": "chat.completion",
                        "created": 0,
                        "model": payload["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "```cpp\nint answer = 42;\n```"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 2,
                            "total_tokens": 3,
                        },
                    },
                    "req-ok",
                )

            with patch.object(RUN_BATCH_MODULE.urllib.request, "urlopen", side_effect=fake_urlopen):
                with self.assertRaises(SystemExit) as exc_context:
                    RUN_BATCH_MODULE.submit_gemma_request(
                        jsonl_path=request_file,
                        responses_dir=responses_dir,
                        api_base="http://bw01:8000/v1",
                        api_key=None,
                        provider="gemma",
                    )

            self.assertEqual(exc_context.exception.code, 1)
            self.assertEqual(request_ids, ["ok", "fail"])
            output_file = responses_dir / "optimization" / "sample.jsonl"
            error_file = responses_dir / "optimization" / "sample.errors.jsonl"

            success_lines = output_file.read_text(encoding="utf-8").splitlines()
            error_lines = error_file.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(success_lines), 1)
            self.assertEqual(json.loads(success_lines[0])["custom_id"], "ok-01")
            self.assertEqual(len(error_lines), 1)
            self.assertEqual(json.loads(error_lines[0])["custom_id"], "fail-01")

    def test_gemma_receive_reports_no_pending(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            requests_file = Path(tmp_dir) / "requests.gemma.jsonl"
            result = subprocess.run(
                [sys.executable, str(RUN_BATCH_PATH), "--provider", "gemma", "--receive", "--requests-file", str(requests_file)],
                cwd=str(GPT_QUERYING_DIR),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0)
            self.assertIn("No pending", result.stderr)


if __name__ == "__main__":
    unittest.main()
