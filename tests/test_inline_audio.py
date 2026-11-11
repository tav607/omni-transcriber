"""Audio always travels inline: Vertex AI has no File API. A transient error
climbs the retry ladder and then switches model; a 403 is left as it is."""

import asyncio
import logging
import unittest

from google.genai import errors, types

from src.services import transcriber
from src.utils import gemini
from src.utils.retry import FALLBACK_MODEL

from .support import MODEL, TranscribeRun, inline_chunk_name

TEXT = "这是一段测试转写，用来确认内联路径能拿到结果，长度也足够通过退化输出检查。"


class _Recorder:
    """answer() for TranscribeRun: records each call, raises `error` when set."""

    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[str] = []

    def __call__(self, source, model):
        assert isinstance(source, types.Part) and source.inline_data is not None
        self.calls.append(f"{inline_chunk_name(source)}:{model}")
        if self.error is not None:
            raise self.error
        return TEXT


class InlineAudioTests(unittest.TestCase):
    def test_single_file_is_sent_inline(self):
        answer = _Recorder()
        self.assertEqual(TranscribeRun(answer).run(), TEXT)
        self.assertEqual(answer.calls, [f"audio.mp3:{MODEL}"])

    def test_every_chunk_is_sent_inline(self):
        answer = _Recorder()
        TranscribeRun(answer, chunks=2).run()
        self.assertEqual(sorted(answer.calls), [f"chunk_000.mp3:{MODEL}", f"chunk_001.mp3:{MODEL}"])

    def test_errors_retry_then_fall_back_to_the_other_model(self):
        answer = _Recorder(RuntimeError("503 UNAVAILABLE"))
        with self.assertLogs(level=logging.WARNING):
            with self.assertRaises(RuntimeError):
                TranscribeRun(answer).run()
        self.assertEqual(answer.calls[:12], [f"audio.mp3:{MODEL}"] * 6 + [f"audio.mp3:{FALLBACK_MODEL}"] * 6)


class _Models:
    def __init__(self, error: Exception):
        self.error = error
        self.models_seen: list[str] = []

    def generate_content(self, model, contents, config):
        self.models_seen.append(model)
        raise self.error


class _Client:
    def __init__(self, error: Exception):
        self.models = _Models(error)


class TranscribeAudioCallTests(unittest.TestCase):
    DENIED = errors.ClientError(403, {"error": {
        "code": 403, "message": "denied", "status": "PERMISSION_DENIED"}})

    def test_403_is_left_as_is_and_the_model_is_named_by_full_path(self):
        part = types.Part.from_bytes(data=b"\x00", mime_type="audio/mpeg")
        client = _Client(self.DENIED)
        with self.assertRaises(errors.ClientError):
            asyncio.run(transcriber._transcribe_audio(client, part, MODEL, 1.0, "high"))
        self.assertEqual(client.models.models_seen, [
            f"projects/test-project/locations/global/publishers/google/models/{MODEL}"])


class ModelPathTests(unittest.TestCase):
    def test_a_path_passes_through(self):
        self.assertEqual(gemini.model_path("publishers/google/models/x"), "publishers/google/models/x")
        self.assertEqual(
            gemini.model_path("gemini-4-argon"),
            "projects/test-project/locations/global/publishers/google/models/gemini-4-argon")


if __name__ == "__main__":
    unittest.main()
