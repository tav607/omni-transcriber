"""Audio transcription using Gemini API with chunking support."""

import asyncio
import re
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from google import genai
from google.genai import types

from ..config import TranscriberConfig
from ..utils.gemini import is_truncated, _gemini_sem
from ..utils.retry import with_retry

logger = logging.getLogger(__name__)

# Module-level client cache: reuse genai.Client to avoid recreating HTTP connection pools
_client_cache: dict[str, genai.Client] = {}


def _get_client(api_key: str) -> genai.Client:
    """Get or create a shared genai.Client for the given API key."""
    if api_key not in _client_cache:
        _client_cache[api_key] = genai.Client(
            api_key=api_key,
            http_options={"base_url": "https://generativelanguage.googleapis.com"},
        )
    return _client_cache[api_key]

# Audio splitting constants
MAX_DURATION_MINUTES = 45
OVERLAP_SECONDS = 10
MAX_DURATION_MS = MAX_DURATION_MINUTES * 60 * 1000  # in milliseconds
OVERLAP_MS = OVERLAP_SECONDS * 1000  # 10 seconds in milliseconds

# Timeout for transcription API calls (15 minutes per chunk, generous for long audio)
TRANSCRIPTION_TIMEOUT_SECONDS = 15 * 60
# Timeout for file upload (5 minutes, should be enough for ~100MB files)
UPLOAD_TIMEOUT_SECONDS = 5 * 60

# Default transcription prompt (for non-podcast sources)
TRANSCRIPTION_PROMPT = (
    "Transcribe this audio verbatim. If the language is Chinese, please use Simplified "
    "Chinese characters. Provide only the direct transcription text without any "
    "introductory phrases. "
    "IMPORTANT: Transcribe exactly what is spoken. Do NOT correct or change "
    "product names, version numbers, or technical terms — even if they seem incorrect or unfamiliar."
)

# Podcast transcription prompt template (with metadata context).
# Speaker labels are assigned here, at the transcription stage, because this is
# the only stage that can hear the voices. The bolded **Name:** format is a
# contract with _SPEAKER_TOKEN_RE below: merge-time seam detection masks these
# labels out, so the prompt and the regex must agree on the format.
PODCAST_TRANSCRIPTION_PROMPT_TEMPLATE = """You are a professional transcriptionist. Transcribe the following audio accurately and verbatim.

## Media Context
{metadata_section}

## Guidelines:
- Transcribe exactly what is said, preserving the original language
- Preserve technical terms, proper nouns, and company names exactly as spoken
- Use appropriate punctuation and paragraph breaks for readability
- If the audio is in Chinese, output in Simplified Chinese (简体中文)
- If the audio is in English, output in English
- Do not add explanations or commentary
- Do not translate - keep the original language
- Label every speaker turn: start the turn with **Name:** when the context or audio identifies the speaker's real name, otherwise **Host:** / **Guest:** by conversational role (the host asks the questions and drives transitions), otherwise **Speaker 1:**, **Speaker 2:** in order of first appearance. Distinguish speakers by their voices and keep each person's label consistent for the whole audio
- For unclear audio, use [inaudible] or [unclear]

Output the complete transcript only."""


@dataclass
class TranscriptionMetadata:
    """
    Metadata passed to transcriber for context.

    Works with both podcast episodes and video sources (YouTube/Bilibili).
    """
    # Generic fields that work for both podcast and video
    source_name: str  # Podcast name or Channel name
    title: str  # Episode title or Video title
    publish_date: str = ""
    description: str = ""  # Episode shownotes or Video description

    # Legacy aliases for backward compatibility
    @property
    def podcast_name(self) -> str:
        return self.source_name

    @property
    def episode_title(self) -> str:
        return self.title

    @property
    def shownotes(self) -> str:
        return self.description


def _build_transcription_prompt(metadata: Optional[TranscriptionMetadata]) -> str:
    """Build transcription prompt with optional metadata context."""
    if metadata is None:
        return TRANSCRIPTION_PROMPT

    # Build metadata section
    metadata_lines = [
        f"- Source: {metadata.source_name}",
        f"- Title: {metadata.title}",
    ]
    if metadata.publish_date:
        metadata_lines.append(f"- Date: {metadata.publish_date}")
    if metadata.description:
        metadata_lines.append(f"- Description: {metadata.description}")

    metadata_section = "\n".join(metadata_lines)
    return PODCAST_TRANSCRIPTION_PROMPT_TEMPLATE.format(metadata_section=metadata_section)

# MIME type mapping
MIME_TYPES = {
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}


def cleanup_repetitive_characters(text: str, max_repeats: int = 10) -> str:
    """
    Clean up repetitive characters in transcription result.
    Remove sequences where the same character repeats more than a threshold.
    """
    if not text:
        return text

    pattern = rf"(.)\1{{{max_repeats},}}"

    def replacer(match: re.Match) -> str:
        char = match.group(1)
        logger.info(
            f"Found repetitive character '{char}' repeated {len(match.group(0))} times, "
            "cleaning to single occurrence"
        )
        return char

    return re.sub(pattern, replacer, text)


def _get_audio_duration(audio_path: Path) -> int:
    """
    Get audio duration in milliseconds using ffprobe.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in milliseconds
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        "--", str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    duration_seconds = float(result.stdout.strip())
    return int(duration_seconds * 1000)


def _split_audio(audio_path: Path, output_dir: Path) -> tuple[List[Path], int]:
    """
    Split audio file into chunks using ffmpeg with stream copy (no re-encoding).

    Args:
        audio_path: Path to original audio file
        output_dir: Directory to save audio chunks

    Returns:
        Tuple of (list of paths to audio chunks, duration in milliseconds).
        If audio is shorter than threshold, returns ([original_path], duration).
    """
    # Get duration first
    duration_ms = _get_audio_duration(audio_path)

    if duration_ms <= MAX_DURATION_MS:
        return [audio_path], duration_ms

    # Defensive check
    if OVERLAP_MS >= MAX_DURATION_MS:
        raise ValueError(
            f"OVERLAP_MS ({OVERLAP_MS}) must be less than MAX_DURATION_MS ({MAX_DURATION_MS})."
        )

    chunks = []
    chunk_index = 0
    start_ms = 0
    ext = audio_path.suffix

    while start_ms < duration_ms:
        end_ms = min(start_ms + MAX_DURATION_MS, duration_ms)
        chunk_duration_ms = end_ms - start_ms

        chunk_path = output_dir / f"chunk_{chunk_index:03d}{ext}"

        # Use ffmpeg with stream copy (very fast, no re-encoding)
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-ss", str(start_ms / 1000),  # Start time in seconds
            "-i", str(audio_path),
            "-t", str(chunk_duration_ms / 1000),  # Duration in seconds
            "-c", "copy",  # Stream copy, no re-encoding
            "-avoid_negative_ts", "1",
            "--", str(chunk_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg split failed: {result.stderr}")

        chunks.append(chunk_path)
        chunk_index += 1

        # If this chunk reached the end, we're done
        if end_ms >= duration_ms:
            break

        # Next chunk starts OVERLAP_SECONDS before the end of current chunk
        start_ms = end_ms - OVERLAP_MS

    logger.info(f"Split audio into {len(chunks)} chunks of ~{MAX_DURATION_MINUTES} minutes each")
    return chunks, duration_ms


# Overlap detection constants
_PUNCT_RE = re.compile(
    r'[\s\u3000。，、！？：；「」『』（）【】\u201c\u201d\u2018\u2019'
    r'\u2026\u2014\u00b7\.\,\!\?\:\;\"\'\(\)\[\]\-\n\r\t>]'
)
# Safety cap: never skip more than this many characters (prevents data loss
# from false-positive overlap detection)
MAX_OVERLAP_SKIP = 1500

# Speaker-label tokens (**Name:** / **Host:** / **Speaker 1:**). Adjacent audio
# chunks label the same overlap speech independently and often differently, so
# label text must not enter the n-gram matching or it poisons the seam detection.
_SPEAKER_TOKEN_RE = re.compile(r'\*\*[^*\n]{1,40}[:：]\*\*')


def _strip_for_matching_with_map(text: str) -> tuple[str, list[int]]:
    """Strip speaker labels, whitespace, and punctuation for fuzzy overlap
    matching. Returns (clean_text, mapping) where mapping[i] is the index in
    `text` of clean_text[i], so match positions can be projected back."""
    masked = _SPEAKER_TOKEN_RE.sub(lambda m: ' ' * len(m.group(0)), text)
    clean_chars: list[str] = []
    mapping: list[int] = []
    for idx, ch in enumerate(masked):
        if not _PUNCT_RE.match(ch):
            clean_chars.append(ch)
            mapping.append(idx)
    return ''.join(clean_chars), mapping


def _strip_for_matching(text: str) -> str:
    """Strip whitespace, punctuation, and speaker labels for fuzzy overlap matching."""
    return _strip_for_matching_with_map(text)[0]


def _find_overlap_length(prev_text: str, next_text: str) -> int:
    """
    Detect how many leading characters of *next_text* duplicate content
    from near the end of *prev_text*.

    Uses 8-character n-gram density in a sliding window, which is robust
    to the minor wording and punctuation differences that Gemini produces
    when the same audio appears in two overlapping chunks.

    Returns the number of characters to skip from the start of next_text
    (0 if no overlap is detected).
    """
    SEARCH_LEN = 2000       # chars to inspect in each chunk
    NGRAM_SIZE = 8           # n-gram length (8 CJK chars ≈ very specific)
    WINDOW_SIZE = 80         # sliding-window width in stripped chars
    WINDOW_STEP = 40         # step size
    MATCH_THRESHOLD = 0.3    # 30 % of n-grams must hit

    prev_tail = prev_text[-SEARCH_LEN:] if len(prev_text) > SEARCH_LEN else prev_text
    next_head = next_text[:SEARCH_LEN] if len(next_text) > SEARCH_LEN else next_text

    prev_clean = _strip_for_matching(prev_tail)
    next_clean, next_map = _strip_for_matching_with_map(next_head)

    if len(prev_clean) < NGRAM_SIZE or len(next_clean) < WINDOW_SIZE:
        return 0  # not enough text for reliable n-gram matching

    # Build n-gram index from previous chunk's tail
    prev_ngrams: set[str] = {
        prev_clean[j:j + NGRAM_SIZE]
        for j in range(len(prev_clean) - NGRAM_SIZE + 1)
    }

    # Fix 3: Anchor check — overlap must start at the beginning of next chunk.
    # If the first window doesn't match, there is no boundary overlap.
    first_window = next_clean[:WINDOW_SIZE]
    first_num = max(1, len(first_window) - NGRAM_SIZE + 1)
    first_hits = sum(
        1 for k in range(first_num)
        if first_window[k:k + NGRAM_SIZE] in prev_ngrams
    )
    if first_hits / first_num < MATCH_THRESHOLD:
        return 0

    # Scan next chunk's head: while n-gram density stays high, we are in
    # the overlap region.  Once it drops, new content has begun.
    overlap_end_clean = 0
    for start in range(0, max(1, len(next_clean) - WINDOW_SIZE + 1), WINDOW_STEP):
        window = next_clean[start:start + WINDOW_SIZE]
        # Fix 1: use actual window length, not fixed WINDOW_SIZE
        window_len = len(window)
        num_ngrams = max(1, window_len - NGRAM_SIZE + 1)
        hits = sum(
            1 for k in range(num_ngrams)
            if window[k:k + NGRAM_SIZE] in prev_ngrams
        )
        if hits / num_ngrams >= MATCH_THRESHOLD:
            overlap_end_clean = start + window_len
        elif overlap_end_clean > 0:
            break  # transition from overlap → new content

    if overlap_end_clean == 0:
        return 0

    # Clamp to actual cleaned text length
    overlap_end_clean = min(overlap_end_clean, len(next_clean))

    # Project the clean-text match end back to a position in next_head via the
    # mapping (label chars were removed from the clean text, so a simple
    # punctuation-skipping recount would drift).
    orig_pos = next_map[overlap_end_clean - 1] + 1

    # Fix 2: Snap forward with a tight tolerance (50 chars) for a clean cut.
    # Larger jumps risk discarding non-duplicated content.
    SNAP_LIMIT = 50
    for j in range(orig_pos, min(orig_pos + SNAP_LIMIT, len(next_text))):
        if (next_text[j] == '\n'
                and j + 1 < len(next_text)
                and next_text[j + 1] == '\n'):
            return j + 2
    for j in range(orig_pos, min(orig_pos + SNAP_LIMIT, len(next_text))):
        if next_text[j] in '。.！!？?\n':
            return j + 1

    return orig_pos


def _merge_transcriptions(transcripts: List[str]) -> str:
    """
    Merge multiple transcript chunks, removing duplicated content at boundaries.

    Uses character n-gram density matching, which is robust to the minor
    transcription variations that Gemini produces when the same audio segment
    appears in overlapping chunks.

    Args:
        transcripts: List of transcript strings to merge

    Returns:
        Merged transcript string
    """
    if len(transcripts) == 1:
        return transcripts[0]

    merged = transcripts[0]

    for i in range(1, len(transcripts)):
        current = transcripts[i]
        raw_skip = _find_overlap_length(merged, current)
        if raw_skip > MAX_OVERLAP_SKIP:
            logger.warning(
                f"Overlap skip {raw_skip} exceeds safety limit {MAX_OVERLAP_SKIP} "
                f"at chunk {i}, ignoring to prevent data loss"
            )
            skip = 0
        else:
            skip = raw_skip
        if skip > 0:
            current = current[skip:].lstrip()
            logger.info(f"Removed {skip} chars of overlapping content at chunk {i}")
        elif raw_skip == 0:
            # A capped skip (raw_skip > limit) is a different, already-warned case;
            # only a genuine miss means the ~OVERLAP_SECONDS overlap survived.
            logger.warning(
                f"No overlap detected at chunk seam {i}; "
                f"~{OVERLAP_SECONDS}s of audio may be duplicated there"
            )
        merged = merged + "\n\n" + current

    return merged


async def transcribe(
    audio_path: str,
    config: TranscriberConfig,
    metadata: Optional[TranscriptionMetadata] = None,
    on_status: Callable[[str], None] | None = None,
) -> str:
    """
    Transcribe audio file using Gemini API.
    Supports automatic chunking for audio longer than 45 minutes.

    Args:
        audio_path: Path to the audio file to transcribe
        config: Transcriber configuration
        metadata: Optional metadata for podcast mode (provides context for better transcription)
        on_status: Optional callback to report status updates

    Returns:
        The transcribed text
    """
    if not config.api_key:
        raise ValueError("Transcriber API key is not configured")

    logger.info("Starting audio transcription processing...")

    # Get shared client (reuses HTTP connection pool)
    client = _get_client(config.api_key)

    audio_path_obj = Path(audio_path)
    if not audio_path_obj.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    chunk_paths: List[Path] = []
    temp_dir = None
    uploaded_files: List[types.File] = []

    try:
        # Split audio if needed (uses ffmpeg, very fast)
        temp_dir = Path(tempfile.mkdtemp(prefix="omni_chunks_"))
        chunk_paths, duration_ms = _split_audio(audio_path_obj, temp_dir)
        duration_minutes = duration_ms // 60000
        needs_splitting = len(chunk_paths) > 1

        if needs_splitting:
            if on_status:
                on_status(f"Audio is {duration_minutes} minutes, split into {len(chunk_paths)} chunks")
            logger.info(f"Split audio into {len(chunk_paths)} chunks")
        else:
            # No splitting needed, use original file directly
            chunk_paths = [audio_path_obj]
            if on_status:
                on_status(f"Audio is {duration_minutes} minutes, processing as single file...")

        # Configure thinking based on level
        thinking_budget = 1024 if config.thinking_level == "low" else 8192

        # Get MIME type based on file extension
        ext = audio_path_obj.suffix.lower()
        mime_type = MIME_TYPES.get(ext, "audio/mpeg")

        # Step 1: Upload all chunks in parallel
        if on_status:
            if len(chunk_paths) > 1:
                on_status(f"Uploading {len(chunk_paths)} audio chunks...")
            else:
                on_status("Uploading audio...")

        logger.info("Uploading audio to Gemini File API...")

        async def upload_chunk(chunk_path: Path) -> types.File:
            chunk_mime = MIME_TYPES.get(chunk_path.suffix.lower(), mime_type)

            # Acquire the gate inside the retried attempt but around the call, so
            # the queue wait sits outside _upload_file's own wait_for budget.
            async def _attempt():
                async with _gemini_sem:
                    return await _upload_file(client, str(chunk_path), chunk_mime)

            return await with_retry(
                _attempt,
                max_attempts=3,
                base_delay_ms=1000,
                context=f"File upload {chunk_path.name}",
            )

        upload_tasks = [upload_chunk(p) for p in chunk_paths]
        uploaded_files = await asyncio.gather(*upload_tasks)
        logger.info(f"Uploaded {len(uploaded_files)} files")

        # Step 2: Transcribe all chunks in parallel
        if on_status:
            if len(chunk_paths) > 1:
                on_status(f"Transcribing {len(chunk_paths)} chunks in parallel...")
            else:
                on_status("Transcribing...")

        async def transcribe_chunk(uploaded_file: types.File) -> str:
            # Gate around the call, outside _transcribe_audio's own wait_for, so
            # queue time doesn't burn the per-chunk timeout budget.
            async def _attempt():
                async with _gemini_sem:
                    return await _transcribe_audio(
                        client,
                        uploaded_file,
                        config.model,
                        config.temperature,
                        thinking_budget,
                        metadata,
                    )

            return await with_retry(
                _attempt,
                max_attempts=3,
                base_delay_ms=1000,
                context="Transcription",
            )

        transcribe_tasks = [transcribe_chunk(f) for f in uploaded_files]
        transcripts = await asyncio.gather(*transcribe_tasks)
        logger.info(f"Transcribed {len(transcripts)} chunks")

        # Step 3: Clean up each transcript
        transcripts = [cleanup_repetitive_characters(t) for t in transcripts]

        # Step 4: Merge if multiple chunks
        if len(transcripts) > 1:
            if on_status:
                on_status("Merging transcriptions...")
            full_text = _merge_transcriptions(transcripts)
        else:
            full_text = transcripts[0]

        logger.info(f"Transcription completed, text length: {len(full_text)}")

        if on_status:
            on_status("Transcription complete")

        logger.info("Transcription process completed!")
        return full_text

    finally:
        # Always clean up uploaded files
        logger.info("Cleaning up uploaded files...")
        if on_status:
            on_status("Cleaning up...")

        for uploaded_file in uploaded_files:
            try:
                await _delete_file(client, uploaded_file.name)
            except Exception:
                logger.warning(
                    f"Failed to clean up uploaded file {uploaded_file.name}"
                )

        # Cleanup temp directory if created
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")


async def _upload_file(
    client: genai.Client, file_path: str, mime_type: str
) -> types.File:
    """Upload a file to Gemini File API."""
    path = Path(file_path)
    original_name = path.name

    async def _do_upload(upload_path: str) -> types.File:
        """Perform upload with timeout."""
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    client.files.upload,
                    file=upload_path,
                    config=types.UploadFileConfig(mime_type=mime_type),
                ),
                timeout=UPLOAD_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"File upload timed out after {UPLOAD_TIMEOUT_SECONDS // 60} minutes."
            )

    # Check if filename contains non-ASCII characters
    try:
        original_name.encode("ascii")
        # ASCII-safe, upload directly
        return await _do_upload(file_path)
    except UnicodeEncodeError:
        # Non-ASCII filename, copy to temp file with safe name
        logger.debug(f"Non-ASCII filename detected: {original_name}, using temp file")
        with tempfile.NamedTemporaryFile(
            suffix=path.suffix, prefix="upload_", delete=False
        ) as tmp:
            tmp_path = tmp.name
        try:
            shutil.copy2(file_path, tmp_path)
            return await _do_upload(tmp_path)
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


async def _transcribe_audio(
    client: genai.Client,
    uploaded_file: types.File,
    model: str,
    temperature: float,
    thinking_budget: int,
    metadata: Optional[TranscriptionMetadata] = None,
) -> str:
    """Transcribe audio using Gemini model."""
    logger.info(f"Processing transcription for {uploaded_file.name}...")

    # Build prompt with or without metadata context
    prompt = _build_transcription_prompt(metadata)

    def _generate():
        return client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_uri(
                            file_uri=uploaded_file.uri,
                            mime_type=uploaded_file.mime_type,
                        ),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                temperature=temperature,
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            ),
        )

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(_generate),
            timeout=TRANSCRIPTION_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise TimeoutError(
            f"Transcription timed out after {TRANSCRIPTION_TIMEOUT_SECONDS // 60} minutes. "
            "The audio chunk may be too long or the API is unresponsive."
        )

    # Validate response
    text = response.text
    if not text or text.strip() == "":
        error_msg = "Transcription returned empty result."
        if hasattr(response, "prompt_feedback") and response.prompt_feedback:
            if hasattr(response.prompt_feedback, "block_reason"):
                error_msg += f" Block reason: {response.prompt_feedback.block_reason}"
        raise ValueError(error_msg)

    # Warn only: a long chunk hitting the output cap is not fixable by a retry,
    # and the partial transcript is still worth keeping.
    if is_truncated(response):
        logger.warning("Transcription hit max output tokens; chunk result may be truncated")

    return text


async def _delete_file(client: genai.Client, file_name: str) -> None:
    """Delete a file from Gemini File API."""
    await asyncio.to_thread(client.files.delete, name=file_name)
