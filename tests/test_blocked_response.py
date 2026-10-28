"""A blocked response moves to the fallback model at once and skips inline audio."""

import asyncio
import logging
import unittest
from types import SimpleNamespace

from google.genai import types

from src.services import transcriber
from src.utils.retry import FALLBACK_MODEL, ResponseBlockedError

from .support import MODEL, TranscribeRun

TEXT = "这是一段测试转写，用来确认换模型之后能拿到结果，长度也足够通过退化输出检查。"


class _BlockedOn:
    """answer() for TranscribeRun: models in `blocked` block every request."""

    def __init__(self, blocked: set[str]):
        self.blocked = blocked
        self.calls: list[str] = []

    def __call__(self, source, model):
        kind = "inline" if isinstance(source, types.Part) else "file"
        self.calls.append(f"{kind}:{model}")
        if model in self.blocked:
            raise ResponseBlockedError(f"Gemini blocked the response on {model} (block_reason=OTHER)")
        return TEXT


class BlockedResponseTests(unittest.TestCase):
    def test_first_block_switches_to_the_fallback_model(self):
        answer = _BlockedOn({MODEL})
        with self.assertLogs(level=logging.WARNING) as captured:
            self.assertEqual(TranscribeRun(answer).run(), TEXT)
        self.assertEqual(answer.calls, [f"file:{MODEL}", f"file:{FALLBACK_MODEL}"])
        self.assertIn(f"Transcription chunk 1/1: Gemini blocked the response on {MODEL} "
                      f"(block_reason=OTHER); switching to {FALLBACK_MODEL} without retrying",
                      "\n".join(captured.output))

    def test_block_on_both_models_fails_without_the_inline_retry(self):
        answer = _BlockedOn({MODEL, FALLBACK_MODEL})
        with self.assertLogs(level=logging.WARNING) as captured:
            with self.assertRaises(ResponseBlockedError):
                TranscribeRun(answer).run()
        self.assertEqual(answer.calls, [f"file:{MODEL}", f"file:{FALLBACK_MODEL}"])
        self.assertIn(f"Transcription chunk 1/1 failed: Gemini blocked the response on "
                      f"{FALLBACK_MODEL}", "\n".join(captured.output))

    def test_retry_log_names_the_chunk_and_model(self):
        failed: list[str] = []

        def answer(source, model):
            if source.name == "files/chunk_001" and not failed:
                failed.append(source.name)
                raise ValueError("transient")
            return TEXT

        with self.assertLogs(level=logging.WARNING) as captured:
            TranscribeRun(answer, chunks=2).run()
        self.assertIn(f"Transcription chunk 2/2 on {MODEL} failed (attempt 1/6): transient",
                      "\n".join(captured.output))

    def test_block_reason_raises_response_blocked(self):
        response = types.GenerateContentResponse(
            prompt_feedback=types.GenerateContentResponsePromptFeedback(
                block_reason=types.BlockedReason.OTHER))
        client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: response))
        with self.assertRaises(ResponseBlockedError) as caught:
            asyncio.run(transcriber._transcribe_audio(
                client, types.Part.from_text(text="audio"), "gemini-3.8-flash", 1.0, "high"))
        self.assertEqual(str(caught.exception),
                         "Gemini blocked the response on gemini-3.8-flash (block_reason=OTHER)")


if __name__ == "__main__":
    unittest.main()
