"""Upload failures and 403-rejected file handles go straight to inline audio."""

import asyncio
import logging
import unittest

from google.genai import errors, types

from src.services import transcriber
from src.utils.retry import FALLBACK_MODEL

from .support import MODEL, TranscribeRun, inline_chunk_name

TEXT = "这是一段测试转写，用来确认内联路径能拿到结果，长度也足够通过退化输出检查。"


class _Recorder:
    """answer() for TranscribeRun: inline parts succeed, file handles raise."""

    def __init__(self, file_error: Exception | None = None):
        self.file_error = file_error
        self.calls: list[str] = []

    def __call__(self, source, model):
        if isinstance(source, types.Part):
            self.calls.append(f"inline:{inline_chunk_name(source)}:{model}")
            return TEXT
        self.calls.append(f"file:{source.name}:{model}")
        if self.file_error is not None:
            raise self.file_error
        return TEXT


def _refuse(path):
    raise RuntimeError("upload refused")


class InlineFallbackTests(unittest.TestCase):
    def test_failed_upload_is_sent_inline(self):
        answer = _Recorder(AssertionError("no file handle exists"))
        run = TranscribeRun(answer, upload=_refuse)
        with self.assertLogs(level=logging.WARNING):
            self.assertEqual(run.run(), TEXT)
        self.assertEqual(answer.calls, [f"inline:audio.mp3:{MODEL}"])
        self.assertEqual(run.deleted, [])

    def test_one_failed_upload_does_not_fail_the_run(self):
        def upload(path):
            if path.name == "chunk_001.mp3":
                raise RuntimeError("upload refused")
            return types.File(name=f"files/{path.stem}", uri="https://example.invalid/x",
                              mime_type="audio/mpeg")

        answer = _Recorder()
        run = TranscribeRun(answer, chunks=2, upload=upload)
        with self.assertLogs(level=logging.WARNING):
            run.run()
        self.assertEqual(sorted(answer.calls), [
            f"file:files/chunk_000:{MODEL}", f"inline:chunk_001.mp3:{MODEL}"])
        # The upload that did succeed is still cleaned up.
        self.assertEqual(run.deleted, ["files/chunk_000"])

    def test_rejected_file_handle_skips_retries_and_model_fallback(self):
        answer = _Recorder(transcriber.FileHandleRejectedError("403 PERMISSION_DENIED"))
        run = TranscribeRun(answer)
        with self.assertLogs(transcriber.logger, level=logging.WARNING):
            run.run()
        self.assertEqual(answer.calls, [
            f"file:files/audio:{MODEL}", f"inline:audio.mp3:{MODEL}"])

    def test_other_file_errors_still_retry_and_fall_back_first(self):
        answer = _Recorder(RuntimeError("503 UNAVAILABLE"))
        run = TranscribeRun(answer)
        with self.assertLogs(level=logging.WARNING):
            run.run()
        file_calls = [c for c in answer.calls if c.startswith("file:")]
        self.assertEqual(file_calls, [f"file:files/audio:{MODEL}"] * 6
                         + [f"file:files/audio:{FALLBACK_MODEL}"] * 6)
        self.assertEqual(answer.calls[-1], f"inline:audio.mp3:{MODEL}")


class _Models:
    def __init__(self, error: Exception):
        self.error = error

    def generate_content(self, model, contents, config):
        raise self.error


class _Client:
    def __init__(self, error: Exception):
        self.models = _Models(error)


class FileHandle403Tests(unittest.TestCase):
    DENIED = errors.ClientError(403, {"error": {
        "code": 403, "message": "denied", "status": "PERMISSION_DENIED"}})

    def test_403_on_a_file_handle_is_a_rejected_handle(self):
        handle = types.File(name="files/x", uri="https://example.invalid/x", mime_type="audio/mpeg")
        with self.assertRaises(transcriber.FileHandleRejectedError):
            asyncio.run(transcriber._transcribe_audio(
                _Client(self.DENIED), handle, MODEL, 1.0, "high"))

    def test_403_on_inline_audio_is_left_as_is(self):
        part = types.Part.from_bytes(data=b"\x00", mime_type="audio/mpeg")
        with self.assertRaises(errors.ClientError):
            asyncio.run(transcriber._transcribe_audio(
                _Client(self.DENIED), part, MODEL, 1.0, "high"))


if __name__ == "__main__":
    unittest.main()
