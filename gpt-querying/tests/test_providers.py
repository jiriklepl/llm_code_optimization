from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENV_KEYS = {
    "GPT_QUERYING_PROVIDER",
    "GPT_QUERYING_MODEL",
    "GPT_QUERYING_MODEL_SNAPSHOT",
    "GPT_QUERYING_OPENAI_MODEL",
    "GPT_QUERYING_GEMINI_MODEL",
    "GPT_QUERYING_ANTHROPIC_MODEL",
    "GPT_QUERYING_REASONING_EFFORT",
    "GPT_QUERYING_THINKING_EFFORT",
    "GPT_QUERYING_VERBOSITY",
    "GPT_QUERYING_GEMINI_THINKING_BUDGET",
    "GPT_QUERYING_GEMINI_INCLUDE_THOUGHTS",
    "GPT_QUERYING_ANTHROPIC_THINKING_BUDGET",
    "GPT_QUERYING_ANTHROPIC_MAX_TOKENS",
    "GPT_QUERYING_ANTHROPIC_BETA_HEADER",
    "GPT_QUERYING_ANTHROPIC_VERSION",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BETA_HEADER",
    "ANTHROPIC_VERSION",
}


def load_module(module_name: str, filename: str):
    module_path = ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def fresh_modules():
    for name in ("providers", "gpt_querying_main_test", "gpt_querying_run_batch_test"):
        sys.modules.pop(name, None)

    providers = load_module("providers", "providers.py")
    main = load_module("gpt_querying_main_test", "main.py")
    run_batch = load_module("gpt_querying_run_batch_test", "run_batch.py")
    return providers, main, run_batch


class ProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {key: "" for key in ENV_KEYS}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.providers, self.main, self.run_batch = fresh_modules()

    def make_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
        ]

    def test_provider_normalization_and_model_resolution(self) -> None:
        self.assertEqual(self.providers.normalize_provider_name("google"), "gemini")
        self.assertEqual(self.providers.normalize_provider_name("claude"), "anthropic")
        self.assertEqual(self.providers.normalize_provider_name("opus"), "anthropic")

        self.assertEqual(self.providers.resolve_provider_model("openai"), "gpt-5.4-2026-03-05")
        self.assertEqual(self.providers.resolve_provider_model("gemini"), "gemini-3.1-pro-preview")
        self.assertEqual(self.providers.resolve_provider_model("anthropic"), "claude-opus-4-6")

        with mock.patch.dict(
            os.environ,
            {
                "GPT_QUERYING_MODEL": "global-model",
                "GPT_QUERYING_GEMINI_MODEL": "gemini-custom",
            },
            clear=False,
        ):
            self.assertEqual(self.providers.resolve_provider_model("openai"), "global-model")
            self.assertEqual(self.providers.resolve_provider_model("gemini"), "gemini-custom")

        self.assertEqual(
            self.providers.resolve_gemini_reasoning_effort("gemini-3.1-pro-preview", "xhigh"),
            "high",
        )
        self.assertEqual(
            self.providers.resolve_anthropic_effort("claude-opus-4-6", "xhigh"),
            "max",
        )
        self.assertIsNone(
            self.providers.resolve_anthropic_thinking_budget(
                reasoning_effort="xhigh",
                model="claude-opus-4-6",
            )
        )
        self.assertEqual(self.providers.DEFAULT_ANTHROPIC_BETA_HEADER, "output-300k-2026-03-24")
        self.assertEqual(self.providers.resolve_anthropic_max_tokens(), 300000)
        self.assertEqual(
            self.run_batch.resolve_beta_header("anthropic", None),
            "output-300k-2026-03-24",
        )

    def test_openai_request_shape(self) -> None:
        provider = self.main.build_batch_provider(
            "openai",
            model="gpt51",
            reasoning_effort="high",
            verbosity="medium",
        )
        request = provider.build_request(self.make_messages())
        line = provider.create_batch_entry("hello-01", request)

        self.assertEqual(line["custom_id"], "hello-01")
        self.assertEqual(line["method"], "POST")
        self.assertEqual(line["url"], "/v1/chat/completions")
        self.assertEqual(line["body"]["model"], "gpt51")
        self.assertEqual(line["body"]["reasoning_effort"], "high")
        self.assertEqual(line["body"]["verbosity"], "medium")
        self.assertEqual(line["body"]["messages"][-1]["content"], "hello")

    def test_gemini_request_shapes(self) -> None:
        provider = self.main.build_batch_provider(
            "gemini",
            model="gemini-3.1-pro-preview",
            reasoning_effort="xhigh",
        )
        request = provider.build_request(self.make_messages())
        self.assertEqual(request["model"], "gemini-3.1-pro-preview")
        self.assertEqual(request["reasoning_effort"], "high")
        self.assertNotIn("extra_body", request)
        self.assertNotIn("verbosity", request)

        provider = self.main.build_batch_provider(
            "gemini",
            model="gemini-3.1-pro-preview",
            reasoning_effort="high",
            gemini_thinking_budget=2048,
            gemini_include_thoughts=True,
        )
        request = provider.build_request(self.make_messages())
        self.assertNotIn("reasoning_effort", request)
        self.assertEqual(
            request["extra_body"]["google"]["thinking_config"],
            {"thinking_budget": 2048, "include_thoughts": True},
        )

        provider = self.main.build_batch_provider(
            "gemini",
            model="gemini-2.5-flash",
            reasoning_effort="xhigh",
        )
        request = provider.build_request(self.make_messages())
        self.assertEqual(request["reasoning_effort"], "high")
        self.assertNotIn("extra_body", request)

    def test_anthropic_request_shapes(self) -> None:
        provider = self.main.build_batch_provider(
            "anthropic",
            model="claude-opus-4-6",
            reasoning_effort="xhigh",
        )
        request = provider.build_request(self.make_messages())
        self.assertEqual(request["max_tokens"], 300000)

        provider = self.main.build_batch_provider(
            "anthropic",
            model="claude-opus-4-6",
            anthropic_max_tokens=32768,
            reasoning_effort="xhigh",
        )
        request = provider.build_request(self.make_messages())
        line = provider.create_batch_entry("hello-01", request)
        self.assertEqual(line["custom_id"], "hello-01")
        self.assertIn("params", line)
        self.assertEqual(request["model"], "claude-opus-4-6")
        self.assertEqual(request["max_tokens"], 32768)
        self.assertEqual(request["system"], "system")
        self.assertEqual(request["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(request["thinking"], {"type": "adaptive"})
        self.assertEqual(request["output_config"], {"effort": "max"})

        provider = self.main.build_batch_provider(
            "anthropic",
            model="claude-opus-4-6",
            anthropic_max_tokens=32768,
            anthropic_thinking_budget=8192,
            reasoning_effort="xhigh",
        )
        request = provider.build_request(self.make_messages())
        self.assertEqual(request["thinking"], {"type": "enabled", "budget_tokens": 8192})
        self.assertEqual(request["output_config"], {"effort": "max"})

    def test_parse_gemini_output_with_errors(self) -> None:
        provider = self.main.build_batch_provider(
            "gemini",
            model="gemini-3.1-pro-preview",
            reasoning_effort="high",
        )

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_file = tmp_path / "responses.jsonl"
            parsed_dir = tmp_path / "parsed"
            response_lines = [
                {
                    "custom_id": "ok-01",
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "content": "```cpp\nint answer = 42;\n```",
                                    }
                                }
                            ],
                            "usageMetadata": {
                                "promptTokenCount": 1000,
                                "candidatesTokenCount": 200,
                                "thoughtsTokenCount": 100,
                            },
                        },
                    },
                },
                {
                    "custom_id": "fail-01",
                    "response": {
                        "status_code": 500,
                        "body": {"error": {"message": "forced failure"}},
                    },
                },
            ]
            output_file.write_text("\n".join(json.dumps(line) for line in response_lines) + "\n", encoding="utf-8")

            self.main.parse_batch_output(output_file, parsed_dir, provider=provider)

            self.assertEqual((parsed_dir / "ok-01" / "code.cpp").read_text(encoding="utf-8"), "int answer = 42;")
            costs = json.loads((parsed_dir / "ok-01" / "costs.json").read_text(encoding="utf-8"))
            self.assertEqual(costs["output_tokens"], 200)
            self.assertEqual(costs["reasoning_tokens"], 100)

            error_payload = json.loads((parsed_dir / "fail-01" / "error.json").read_text(encoding="utf-8"))
            self.assertEqual(error_payload["status_code"], 500)

    def test_parse_anthropic_output_with_errors(self) -> None:
        provider = self.main.build_batch_provider(
            "anthropic",
            model="claude-opus",
            anthropic_max_tokens=32768,
            anthropic_thinking_budget=8192,
        )

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_file = tmp_path / "responses.jsonl"
            parsed_dir = tmp_path / "parsed"
            response_lines = [
                {
                    "custom_id": "ok-01",
                    "result": {
                        "type": "succeeded",
                        "message": {
                            "content": [
                                {"type": "thinking", "text": "internal notes"},
                                {"type": "text", "text": "```cpp\nint answer = 42;\n```"},
                            ],
                            "usage": {
                                "input_tokens": 1000,
                                "output_tokens": 200,
                            },
                        },
                    },
                },
                {
                    "custom_id": "fail-01",
                    "result": {
                        "type": "errored",
                        "error": {"message": "forced failure"},
                    },
                },
            ]
            output_file.write_text("\n".join(json.dumps(line) for line in response_lines) + "\n", encoding="utf-8")

            self.main.parse_batch_output(output_file, parsed_dir, provider=provider)

            self.assertEqual((parsed_dir / "ok-01" / "code.cpp").read_text(encoding="utf-8"), "int answer = 42;")
            costs = json.loads((parsed_dir / "ok-01" / "costs.json").read_text(encoding="utf-8"))
            self.assertEqual(costs["output_tokens"], 200)
            self.assertEqual(costs["reasoning_tokens"], 0)
            self.assertTrue(costs["thinking_token_split_unavailable"])

            error_payload = json.loads((parsed_dir / "fail-01" / "error.json").read_text(encoding="utf-8"))
            self.assertEqual(error_payload["type"], "errored")

    def test_gemini_transport_submit_and_receive(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            request_file = tmp_path / "requests" / "gemini" / "translation" / "sample.jsonl"
            request_file.parent.mkdir(parents=True, exist_ok=True)
            request_file.write_text(
                json.dumps(
                    {
                        "custom_id": "hello-01",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {"model": "gemini-3.1-pro-preview", "messages": [{"role": "user", "content": "hello"}]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            requests_file = tmp_path / "requests.gemini.jsonl"
            responses_dir = tmp_path / "responses" / "gemini"
            response_line = {
                "custom_id": "hello-01",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{"message": {"content": "```cpp\nint answer = 42;\n```"}}],
                    },
                },
            }
            download_bytes = (json.dumps(response_line) + "\n").encode("utf-8")

            fake_openai_client = mock.Mock()
            fake_openai_client.batches.create.return_value = {"id": "batch_1", "status": "validating"}
            fake_openai_client.batches.retrieve.return_value = {
                "id": "batch_1",
                "status": "completed",
                "output_file_id": "files/output-1",
            }
            fake_genai_client = mock.Mock()
            fake_genai_client.files.upload.return_value = {"name": "files/input-1"}
            fake_genai_client.files.download.return_value = download_bytes

            with mock.patch.object(self.run_batch, "OpenAI", return_value=fake_openai_client), \
                 mock.patch.object(self.run_batch, "build_genai_client", return_value=fake_genai_client), \
                 mock.patch.object(self.run_batch, "build_google_upload_file_config", return_value=object()):
                client = self.run_batch.GeminiBatchClient(
                    api_key="test-key",
                    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
                )
                self.run_batch.submit_openai_compatible_request(
                    client=client,
                    jsonl_path=request_file,
                    requests_file=requests_file,
                    responses_dir=responses_dir,
                    completion_window="24h",
                    metadata=None,
                    provider="gemini",
                )
                self.run_batch.receive_openai_compatible_batches(
                    client=client,
                    requests_file=requests_file,
                    responses_dir=responses_dir,
                )

            self.assertEqual(self.run_batch.load_requests(requests_file), [])
            written_output = responses_dir / "translation" / "sample.jsonl"
            self.assertEqual(written_output.read_bytes(), download_bytes)

    def test_anthropic_transport_submit_and_receive(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            request_file = tmp_path / "requests" / "anthropic" / "optimization" / "sample.jsonl"
            request_file.parent.mkdir(parents=True, exist_ok=True)
            request_file.write_text(
                json.dumps(
                    {
                        "custom_id": "hello-01",
                        "params": {
                            "model": "claude-opus",
                            "max_tokens": 32768,
                            "messages": [{"role": "user", "content": "hello"}],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            requests_file = tmp_path / "requests.anthropic.jsonl"
            responses_dir = tmp_path / "responses" / "anthropic"
            result_line = {
                "custom_id": "hello-01",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "content": [{"type": "text", "text": "```cpp\nint answer = 42;\n```"}],
                    },
                },
            }
            download_bytes = (json.dumps(result_line) + "\n").encode("utf-8")

            client = self.run_batch.AnthropicBatchClient(
                api_key="test-key",
                api_base="https://api.anthropic.com/v1",
                anthropic_version="2023-06-01",
                beta_header=self.run_batch.resolve_beta_header("anthropic", None),
            )
            self.assertEqual(client._headers()["anthropic-beta"], "output-300k-2026-03-24")

            with mock.patch.object(
                client,
                "_request_json",
                side_effect=[
                    {"id": "msgbatch_1", "processing_status": "in_progress"},
                    {"id": "msgbatch_1", "processing_status": "ended", "results_url": "https://example.test/results"},
                ],
            ), mock.patch.object(client, "_request_bytes", return_value=download_bytes):
                self.run_batch.submit_anthropic_request(
                    client=client,
                    jsonl_path=request_file,
                    requests_file=requests_file,
                    responses_dir=responses_dir,
                    completion_window="24h",
                    metadata=None,
                    provider="anthropic",
                )
                self.run_batch.receive_anthropic_batches(
                    client=client,
                    requests_file=requests_file,
                    responses_dir=responses_dir,
                )

            self.assertEqual(self.run_batch.load_requests(requests_file), [])
            written_output = responses_dir / "optimization" / "sample.jsonl"
            self.assertEqual(written_output.read_bytes(), download_bytes)

    def test_receive_reports_no_pending(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            requests_file = tmp_path / "requests.gemini.jsonl"
            responses_dir = tmp_path / "responses"
            fake_client = mock.Mock()

            with self.assertLogs("run_batch", level="INFO") as captured:
                self.run_batch.receive_openai_compatible_batches(
                    client=fake_client,
                    requests_file=requests_file,
                    responses_dir=responses_dir,
                )

            self.assertTrue(any("No pending batch requests" in line for line in captured.output))


if __name__ == "__main__":
    unittest.main()
