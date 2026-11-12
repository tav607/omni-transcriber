"""A recitation block skips the retries and the model switch, and the whole
audio is transcribed again in shorter chunks."""

import asyncio
import shutil
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from google.genai import types

from src.config import TranscriberConfig
from src.services import transcriber
from src.services.transcriber import (
    MAX_DURATION_MS,
    RECITATION_FALLBACK_MS,
    TranscriptionBlockedError,
)
from src.utils import retry
from src.utils.retry import ResponseBlockedError

from .support import MODEL

TEXT = "这是一段测试转写，用来确认重切之后能拿到结果，长度也足够通过退化输出检查。" * 12


def run_transcribe(answer, duration_ms: int):
    """transcribe() over fake audio of duration_ms. The fake split honours
    max_duration_ms, so the fallback's shorter chunks get their own names
    (rec_000.mp3, ...) and answer(chunk_name, model) can tell the two apart."""
    workdir = Path(tempfile.mkdtemp(prefix="omni_test_"))
    audio = workdir / "audio.mp3"
    audio.touch()
    splits: list[int] = []

    def split(audio_path, output_dir, max_duration_ms=MAX_DURATION_MS):
        splits.append(max_duration_ms)
        count = -(-duration_ms // max_duration_ms)
        if count == 1:
            return [audio_path], [duration_ms], duration_ms, [0]
        prefix = "rec" if max_duration_ms == RECITATION_FALLBACK_MS else "chunk"
        paths = [output_dir / f"{prefix}_{i:03d}.mp3" for i in range(count)]
        starts = [i * max_duration_ms for i in range(count)]
        durations = [min(max_duration_ms, duration_ms - s) for s in starts]
        return paths, durations, duration_ms, starts

    async def inline_part(path, temp_dir, mime_type):
        return types.Part.from_bytes(data=path.name.encode(), mime_type="audio/mpeg")

    async def transcribe_audio(client, source, model, temperature, thinking_level,
                               metadata=None, podcast_mode=False, label="chunk", labeling=None):
        return answer(source.inline_data.data.decode(), model)

    try:
        with ExitStack() as stack:
            for name, fake in (("_split_audio", split),
                               ("_inline_audio_part", inline_part),
                               ("_transcribe_audio", transcribe_audio),
                               ("_get_client", lambda key: object())):
                stack.enter_context(mock.patch.object(transcriber, name, fake))
            stack.enter_context(mock.patch.object(retry, "MAX_RETRY_DELAY_MS", 0))
            text = asyncio.run(transcriber.transcribe(
                str(audio), TranscriberConfig(api_key="test-key", model=MODEL)))
        return text, splits
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


class RecitationFallbackTests(unittest.TestCase):
    def test_blocked_chunk_is_not_retried_and_the_audio_is_cut_shorter(self):
        calls: list[tuple[str, str]] = []

        def answer(chunk, model):
            calls.append((chunk, model))
            if chunk == "chunk_001.mp3":
                raise TranscriptionBlockedError("recitation")
            return TEXT

        with self.assertLogs(transcriber.logger, level="WARNING") as captured:
            text, splits = run_transcribe(answer, duration_ms=30 * 60_000)
        self.assertEqual(splits, [MAX_DURATION_MS, RECITATION_FALLBACK_MS])
        # One attempt at the blocked chunk, on the configured model only.
        self.assertEqual([c for c in calls if c[0] == "chunk_001.mp3"], [("chunk_001.mp3", MODEL)])
        self.assertEqual(sorted(c[0] for c in calls if c[0].startswith("rec_")),
                         ["rec_000.mp3", "rec_001.mp3", "rec_002.mp3"])
        self.assertEqual(text.count(TEXT), 3)
        self.assertIn("retrying with 10-minute chunks", "\n".join(captured.output))

    def test_block_wins_over_another_chunks_failure_in_the_same_wave(self):
        def answer(chunk, model):
            if chunk == "chunk_000.mp3":
                raise ResponseBlockedError("blocked on both models")
            if chunk == "chunk_001.mp3":
                raise TranscriptionBlockedError("recitation")
            return TEXT

        with self.assertLogs(level="WARNING"):
            text, splits = run_transcribe(answer, duration_ms=30 * 60_000)
        self.assertEqual(splits, [MAX_DURATION_MS, RECITATION_FALLBACK_MS])
        self.assertEqual(text.count(TEXT), 3)

    def test_audio_already_short_fails_without_a_second_split(self):
        def answer(chunk, model):
            raise TranscriptionBlockedError("recitation")

        with self.assertRaises(TranscriptionBlockedError):
            run_transcribe(answer, duration_ms=8 * 60_000)

    def test_block_in_the_shorter_chunks_fails_the_run(self):
        def answer(chunk, model):
            raise TranscriptionBlockedError("recitation")

        with self.assertLogs(transcriber.logger, level="WARNING"):
            with self.assertRaises(TranscriptionBlockedError):
                run_transcribe(answer, duration_ms=30 * 60_000)


class RecitationDetectionTests(unittest.TestCase):
    """_transcribe_audio reads the block off the real SDK response type."""

    def transcribe(self, response):
        client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: response))
        part = types.Part.from_bytes(data=b"x", mime_type="audio/mpeg")
        return asyncio.run(transcriber._transcribe_audio(client, part, MODEL, 1.0, "high"))

    def test_recitation_finish_reason_raises_the_blocked_error(self):
        response = types.GenerateContentResponse(
            candidates=[types.Candidate(finish_reason=types.FinishReason.RECITATION)])
        with self.assertRaises(TranscriptionBlockedError):
            self.transcribe(response)

    def test_other_empty_response_names_its_finish_reason(self):
        response = types.GenerateContentResponse(
            candidates=[types.Candidate(finish_reason=types.FinishReason.SAFETY)])
        with self.assertRaises(ValueError) as caught:
            self.transcribe(response)
        self.assertNotIsInstance(caught.exception, TranscriptionBlockedError)
        self.assertIn("finish_reason=SAFETY", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
