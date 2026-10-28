"""Fakes that run the transcription and edit pipelines without ffmpeg or Gemini."""

import asyncio
import json
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from google.genai import types

from src.config import EditorConfig, TranscriberConfig
from src.services import editor, transcriber
from src.utils import retry

MODEL = "gemini-3.8-flash"


def inline_chunk_name(part: types.Part) -> str:
    """The chunk file name a fake inline part was built from."""
    return part.inline_data.data.decode()


class TranscribeRun:
    """transcribe() with the audio split, the File API and the model faked.

    upload(path) returns a types.File or raises; answer(source, model) returns
    the transcript for a File handle or an inline types.Part, or raises.
    """

    def __init__(self, answer, chunks: int = 1, upload=None):
        self.answer = answer
        self.chunks = chunks
        self.upload = upload or (lambda path: types.File(
            name=f"files/{path.stem}", uri=f"https://example.invalid/{path.stem}",
            mime_type="audio/mpeg"))
        self.deleted: list[str] = []

    def run(self, podcast_mode: bool = False) -> str:
        """Audio that fits one chunk is sent as the source file, audio.mp3;
        longer audio is split into chunk_000.mp3, chunk_001.mp3, ..."""
        workdir = Path(tempfile.mkdtemp(prefix="omni_test_"))
        audio = workdir / "audio.mp3"
        audio.touch()

        def split(audio_path, output_dir, max_duration_ms=transcriber.MAX_DURATION_MS):
            if self.chunks == 1:
                return [audio_path], [60_000], 60_000, [0]
            # Chunks that meet in pauses: no shared audio at any seam.
            paths = [output_dir / f"chunk_{i:03d}.mp3" for i in range(self.chunks)]
            starts = [i * 60_000 for i in range(self.chunks)]
            return paths, [60_000] * self.chunks, 60_000 * self.chunks, starts

        async def upload(client, file_path, mime_type):
            return self.upload(Path(file_path))

        async def inline_part(path, temp_dir, mime_type):
            return types.Part.from_bytes(data=path.name.encode(), mime_type="audio/mpeg")

        async def transcribe_audio(client, source, model, temperature, thinking_level,
                                   metadata=None, podcast_mode=False, label="chunk"):
            return self.answer(source, model)

        async def delete(client, name):
            self.deleted.append(name)

        try:
            with ExitStack() as stack:
                for name, fake in (("_split_audio", split), ("_upload_file", upload),
                                   ("_inline_audio_part", inline_part),
                                   ("_transcribe_audio", transcribe_audio),
                                   ("_delete_file", delete),
                                   ("_get_client", lambda key: object())):
                    stack.enter_context(mock.patch.object(transcriber, name, fake))
                # Retries back off without sleeping.
                stack.enter_context(mock.patch.object(retry, "MAX_RETRY_DELAY_MS", 0))
                return asyncio.run(transcriber.transcribe(
                    str(audio), TranscriberConfig(api_key="test-key", model=MODEL),
                    podcast_mode=podcast_mode,
                ))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


def _text_response(text: str) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part.from_text(text=text)]))])


def _editor_gemini(edit_chunk, metadata: dict):
    """A stand-in for editor._gemini_generate, so the real chunk editors,
    metadata parsing and glossary step run: schema calls get the glossary or
    the metadata JSON, edit calls get edit_chunk(raw_chunk)."""
    async def generate(client, model, contents, *, system=None, schema=None, **kwargs):
        if schema is not None:
            if system == editor.GLOSSARY_PROMPT:
                return _text_response('{"terms": []}')
            return _text_response(json.dumps(metadata, ensure_ascii=False))
        # "## Raw Transcript Chunk (i/n)", a blank line, then the chunk itself.
        chunk = contents.split("## Raw Transcript Chunk", 1)[1].split("\n", 2)[2]
        return _text_response(edit_chunk(chunk))
    return generate


def run_edit(raw: str, edit_chunk, metadata: dict) -> str:
    """edit() with each edit call answered by edit_chunk(raw_chunk)."""
    with mock.patch.object(editor, "_gemini_generate", _editor_gemini(edit_chunk, metadata)), \
            mock.patch.object(editor, "_get_client", lambda key: object()):
        return asyncio.run(editor.edit(raw, EditorConfig(api_key="test-key", model=MODEL)))


def run_edit_podcast(raw: str, edit_chunk, metadata: dict,
                     episode: "editor.PodcastEpisodeMetadata | None" = None):
    """edit_podcast() with each edit call answered by edit_chunk(raw_chunk)."""
    with mock.patch.object(editor, "_gemini_generate", _editor_gemini(edit_chunk, metadata)), \
            mock.patch.object(editor, "_get_client", lambda key: object()):
        return asyncio.run(editor.edit_podcast(
            raw, EditorConfig(api_key="test-key", model=MODEL), metadata=episode))
