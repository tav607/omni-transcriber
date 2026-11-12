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
    """transcribe() with the audio split and the model faked.

    answer(part, model) returns the transcript for an inline types.Part (see
    inline_chunk_name for which chunk it carries), or raises.
    """

    def __init__(self, answer, chunks: int = 1):
        self.answer = answer
        self.chunks = chunks

    def run(self, podcast_mode: bool = False, labeling=None) -> str:
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

        async def inline_part(path, temp_dir, mime_type):
            return types.Part.from_bytes(data=path.name.encode(), mime_type="audio/mpeg")

        async def transcribe_audio(client, source, model, temperature, thinking_level,
                                   metadata=None, podcast_mode=False, label="chunk", labeling=None):
            return self.answer(source, model)

        try:
            with ExitStack() as stack:
                for name, fake in (("_split_audio", split),
                                   ("_inline_audio_part", inline_part),
                                   ("_transcribe_audio", transcribe_audio),
                                   ("_get_client", lambda key: object())):
                    stack.enter_context(mock.patch.object(transcriber, name, fake))
                # Retries back off without sleeping.
                stack.enter_context(mock.patch.object(retry, "MAX_RETRY_DELAY_MS", 0))
                return asyncio.run(transcriber.transcribe(
                    str(audio), TranscriberConfig(api_key="test-key", model=MODEL),
                    podcast_mode=podcast_mode, labeling=labeling,
                ))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


def _text_response(text: str) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part.from_text(text=text)]))])


def _editor_gemini(edit_chunk, metadata: dict, on_metadata=None, on_edit=None, on_summary=None):
    """A stand-in for editor._gemini_generate, so the real chunk editors,
    metadata parsing and glossary step run: schema calls get the glossary or
    the metadata JSON, edit calls get edit_chunk(raw_chunk)."""
    async def generate(client, model, contents, *, system=None, schema=None, **kwargs):
        if schema is not None:
            if system == editor.GLOSSARY_PROMPT:
                return _text_response('{"terms": []}')
            if on_metadata:
                on_metadata(system, schema)
            if on_summary:
                on_summary(contents)
            return _text_response(json.dumps(metadata, ensure_ascii=False))
        if on_edit:
            on_edit(system, contents)
        # "## Raw Transcript Chunk (i/n)", a blank line, then the chunk itself.
        chunk = contents.split("## Raw Transcript Chunk", 1)[1].split("\n", 2)[2]
        return _text_response(edit_chunk(chunk))
    return generate


def run_edit(raw: str, edit_chunk, metadata: dict, on_metadata=None, on_edit=None,
             on_summary=None, **edit_kwargs) -> str:
    """edit() with each edit call answered by edit_chunk(raw_chunk). on_edit
    and on_metadata see the (system prompt, contents or schema) of each call,
    on_summary the user content of the metadata call; edit_kwargs go to edit()
    (mode, sections, roles, glossary, ...)."""
    fake = _editor_gemini(edit_chunk, metadata, on_metadata, on_edit, on_summary)
    with mock.patch.object(editor, "_gemini_generate", fake), \
            mock.patch.object(editor, "_get_client", lambda key: object()):
        return asyncio.run(editor.edit(
            raw, EditorConfig(api_key="test-key", model=MODEL), **edit_kwargs))


def run_edit_podcast(raw: str, edit_chunk, metadata: dict,
                     episode: "editor.PodcastEpisodeMetadata | None" = None):
    """edit_podcast() with each edit call answered by edit_chunk(raw_chunk)."""
    with mock.patch.object(editor, "_gemini_generate", _editor_gemini(edit_chunk, metadata)), \
            mock.patch.object(editor, "_get_client", lambda key: object()):
        return asyncio.run(editor.edit_podcast(
            raw, EditorConfig(api_key="test-key", model=MODEL), metadata=episode))
