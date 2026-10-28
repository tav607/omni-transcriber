"""Audio transcription using Gemini API with chunking support."""

import array
import asyncio
import re
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from google import genai
from google.genai import types

from ..config import TranscriberConfig
from ..utils.chinese import _to_simplified, _zh_fold
from ..utils.gemini import is_truncated, _check_served_model, _gemini_sem
from ..utils.retry import ResponseBlockedError, with_retry, with_model_fallback

logger = logging.getLogger(__name__)

# Module-level client cache: reuse genai.Client to avoid recreating HTTP connection pools
_client_cache: dict[str, genai.Client] = {}


def _get_client(api_key: str) -> genai.Client:
    """Get or create a shared genai.Client for the given API key.

    base_url is left unset so the google-genai SDK's own GOOGLE_GEMINI_BASE_URL
    environment variable takes effect (its default matches the official
    endpoint, so behaviour is unchanged unless that variable is set to route
    Gemini through a relay/proxy).
    """
    if api_key not in _client_cache:
        _client_cache[api_key] = genai.Client(api_key=api_key)
    return _client_cache[api_key]

# Audio splitting constants. 15 minutes per chunk: transcription accuracy
# degrades late in long chunks (2026-08 benchmarks in the transcribe skill:
# ~4.3% word error at 15-min chunks vs 6.5-8.6% as a single 45-min chunk,
# errors climbing from ~minute 18). Don't raise this.
MAX_DURATION_MINUTES = 15
OVERLAP_SECONDS = 10
MAX_DURATION_MS = MAX_DURATION_MINUTES * 60 * 1000  # in milliseconds
OVERLAP_MS = OVERLAP_SECONDS * 1000  # 10 seconds in milliseconds

# Chunks are cut in a pause (backported from the transcribe skill, which
# measured the numbers below). Within the minute before each 15-minute mark the
# longest stretch of near-silence wins and the cut goes through its middle, so
# no word is split and no audio is shared. Only a minute without a pause of
# SPLIT_MIN_PAUSE_SECONDS falls back to cutting at the mark with
# OVERLAP_SECONDS of shared audio, for the merge to align as before. Shared
# audio is transcribed twice, and when the two chunks word it differently the
# merge cannot align it: in five saved calls (2026-10-27) that left a duplicate
# at 3 of 15 seams, all in expert calls (a question cut off in one chunk and
# whole in the next; "感谢" against "周末愉快"). A chunk that opens mid-sentence
# also invites an invented opening: both Gemini passes of one Dewu chunk began
# with a sentence nobody said ("这两个满打满算也就几十万吧", "这两、两个场景我们
# 就都讲了"; the skill's MAI transcript and a pause-cut run of the same audio
# have neither), and with that seam unaligned it stayed in the transcript.
# Every one of the 15 seams had a pause of 0.6 s or more in the minute before
# the mark; re-run with pause cuts, the Dewu and Maybank calls cut all 7 seams
# in a pause and kept no duplicate.
SPLIT_SEARCH_SECONDS = 60
SPLIT_MIN_PAUSE_SECONDS = 0.3
# Near-silence is judged against the stretch itself, since a phone call and a
# far-field meeting sit 30 dB apart: 20 ms frames quieter than the 5th
# percentile plus 30% of the way to the 95th. A fixed -35 dB took a far-field
# meeting's quiet speech for a 30-second pause.
_PAUSE_RATE = 8000
_PAUSE_FRAME = 160
_PAUSE_FRAME_S = _PAUSE_FRAME / _PAUSE_RATE

# Timeout for transcription API calls (15 minutes per chunk, generous for long audio)
TRANSCRIPTION_TIMEOUT_SECONDS = 15 * 60
# Timeout for file upload (5 minutes, should be enough for ~100MB files)
UPLOAD_TIMEOUT_SECONDS = 5 * 60

# Degenerate-output floor: far below real speech density (Chinese ~3-5,
# English ~12-15 chars/sec), so only content-free responses trip it.
MIN_TRANSCRIPT_CHARS_PER_SEC = 0.5
# Retries for a suspiciously-short chunk transcript before accepting it.
TRANSCRIPTION_SHORT_RETRIES = 3

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
{speaker_guideline}{date_anchor}
- For unclear audio, use [inaudible] or [unclear]

Output the complete transcript only."""

# Strong speaker labeling for conversational audio (podcasts, meetings, uploaded
# recordings). The bolded **Name:** format pairs with _SPEAKER_TOKEN_RE at merge
# time, so seam detection can mask the labels out.
# The example names are placeholders, not real people: gemini-3.7-flash was caught
# lifting a real-sounding example name out of this rule and using it as the actual
# speaker label in 3 of 9 Chinese runs (2026-08-14), so the format illustration must
# not look like someone the model could plausibly be hearing.
PODCAST_SPEAKER_GUIDELINE = (
    "- Label every speaker turn: start the turn with **Name:** when the context or "
    "audio identifies the speaker's real name, otherwise **Host:** / **Guest:** by "
    "conversational role (the host asks the questions and drives transitions), "
    "otherwise **Speaker 1:**, **Speaker 2:** in order of first appearance. "
    "Distinguish speakers by their voices and keep each person's label consistent "
    "for the whole audio. Once a speaker's full name is known from the context or "
    "audio, use the FULL name in every label (**Jane Doe:**, not **Jane:**; "
    "first names can collide). Write names in the language of the audio: for "
    "Chinese audio use Chinese characters (e.g. **张三:**, never pinyin or other "
    "romanization), for English audio use the Latin spelling. The names in these "
    "examples are placeholders showing the format only; never use them as a "
    "speaker label"
)
# Gentle labeling for single-track video (YouTube etc.), where forcing role or
# numbered labels onto a single-presenter talk adds noise.
GENERIC_SPEAKER_GUIDELINE = "- Include speaker changes when clearly identifiable"


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


# A YouTube/Bilibili video whose metadata reads like a podcast (interview,
# panel, 访谈, ...) should transcribe in podcast mode so it gets the strong
# speaker labeling. Keyword set copied verbatim from the transcribe skill's
# _is_podcast heuristic.
_PODCAST_VIDEO_KEYWORDS = re.compile(
    r"podcast|播客|访谈|对谈|圆桌|panel|interview|conversation|talk\s*show|episode\s*\d",
    re.IGNORECASE,
)


def is_podcast_video(title: str, channel: str = "", description: str = "") -> bool:
    """True if a video's title/channel/description reads like a podcast, so the
    transcription should use podcast-mode speaker labeling instead of the gentle
    single-track guideline."""
    haystack = f"{title or ''} {channel or ''} {(description or '')[:500]}"
    return bool(_PODCAST_VIDEO_KEYWORDS.search(haystack))


def _build_transcription_prompt(
    metadata: Optional[TranscriptionMetadata],
    podcast_mode: bool = False,
) -> str:
    """Build the transcription prompt.

    podcast_mode selects the speaker-labeling guideline (strong role/name labels
    vs. gentle) by content form, independently of whether metadata is present:
    audio with no metadata still gets speaker labels, while a plain video does
    not get the forced role/numbered labels.
    """
    if metadata is None and not podcast_mode:
        return TRANSCRIPTION_PROMPT

    # Build metadata section
    metadata_lines = []
    if metadata is not None:
        metadata_lines.append(f"- Source: {metadata.source_name}")
        metadata_lines.append(f"- Title: {metadata.title}")
        if metadata.publish_date:
            metadata_lines.append(f"- Date: {metadata.publish_date}")
        if metadata.description:
            metadata_lines.append(f"- Description: {metadata.description}")
    if not metadata_lines:
        metadata_lines.append("- (no metadata available)")

    # Year anchor (backported from the transcribe skill, 2026-09-11): without it
    # a pass can rewrite every spoken year to an earlier one from training memory
    # and drag product names and titles back a generation with it (Netflix Q2
    # 2026: 2026 -> 2024, current slate -> Bridgerton/Under Paris/Golf; CRWV:
    # Vera Rubin NVL72 -> GB200 NVL72). Measured on gemini-3.7-flash, 6 passes
    # per variant: no metadata 4/12 drifted, title + date lines alone 4/12,
    # this explicit rule 0/12. The date line by itself does not protect.
    date_anchor = ""
    if metadata is not None and metadata.publish_date:
        date_anchor = (
            f"\n- This recording is dated {metadata.publish_date}. Transcribe every spoken year exactly as heard; "
            f"never shift a year, quarter, or product generation to an earlier one from memory. "
            f"Names and products that sound unfamiliar are current as of {metadata.publish_date}; write them as heard."
        )

    return PODCAST_TRANSCRIPTION_PROMPT_TEMPLATE.format(
        metadata_section="\n".join(metadata_lines),
        speaker_guideline=PODCAST_SPEAKER_GUIDELINE if podcast_mode else GENERIC_SPEAKER_GUIDELINE,
        date_anchor=date_anchor,
    )

# MIME type mapping
MIME_TYPES = {
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}


# ffmpeg and ffprobe echo the source file's tags into stderr, and those tags
# are not always UTF-8: Chinese podcast mp3s routinely carry GBK-encoded ID3
# frames. Under a strict text=True the decode raises before the returncode is
# ever inspected, so the whole run dies on a metadata byte with a traceback
# naming neither the file nor ffmpeg. Replacement characters in a diagnostic
# string cost nothing; callers only match on returncode or log stderr (the
# duration probe parses stdout, which holds nothing but ffprobe's ASCII number).
_FFMPEG_TEXT = {"text": True, "errors": "replace"}


class FileHandleRejectedError(Exception):
    """A request referencing an uploaded file came back 403 PERMISSION_DENIED.

    On 2026-08-14 every file_uri request failed this way on every 3.1+ model
    for about 11 hours while the same audio sent inline worked. Retrying and
    switching models only burned about two minutes per chunk before the inline
    fallback, so this skips straight to it."""


# Inlining the audio caps the whole request at 20MB, and chunks are cut with
# ffmpeg stream copy so they inherit the source bitrate: 15 minutes of 320kbps
# audio is ~36MB. Anything near the ceiling is re-encoded to 64kbps mono, which
# fits 15 minutes into ~7MB and stays well above what speech recognition needs.
INLINE_AUDIO_MAX_BYTES = 18_000_000


async def _inline_audio_part(path: Path, temp_dir: Path, mime_type: str) -> types.Part:
    """Read a chunk into an inline request part, re-encoding if it is too big."""
    src = path
    if path.stat().st_size > INLINE_AUDIO_MAX_BYTES:
        src = temp_dir / f"{path.stem}_inline.mp3"
        if not src.exists():
            result = await asyncio.to_thread(
                subprocess.run,
                ["ffmpeg", "-y", "-v", "error", "-i", str(path),
                 "-ac", "1", "-b:a", "64k", "--", str(src)],
                capture_output=True, **_FFMPEG_TEXT,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Inline fallback: mp3 re-encode failed: {result.stderr[-300:]}"
                )
        size = src.stat().st_size
        if size > INLINE_AUDIO_MAX_BYTES:
            raise RuntimeError(
                f"Inline fallback: chunk is still {size / 1e6:.0f}MB after "
                "re-encoding, too large to send inline"
            )
        mime_type = "audio/mpeg"
    return types.Part.from_bytes(data=src.read_bytes(), mime_type=mime_type)


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
    result = subprocess.run(cmd, capture_output=True, **_FFMPEG_TEXT, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    duration_seconds = float(result.stdout.strip())
    return int(duration_seconds * 1000)


def _frame_levels_db(audio_path: Path, start_s: float, length_s: float) -> list[float]:
    """Loudness of each 20 ms frame of a stretch of audio, in dBFS ([] on failure)."""
    cmd = [
        "ffmpeg", "-v", "error", "-ss", f"{start_s:.3f}", "-t", f"{length_s:.3f}",
        "-i", str(audio_path), "-ac", "1", "-ar", str(_PAUSE_RATE), "-f", "s16le", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        return []
    samples = array.array("h")
    samples.frombytes(result.stdout[: len(result.stdout) // 2 * 2])
    if sys.byteorder == "big":
        samples.byteswap()
    levels = []
    for i in range(0, len(samples) - _PAUSE_FRAME + 1, _PAUSE_FRAME):
        power = sum(x * x for x in samples[i:i + _PAUSE_FRAME]) / _PAUSE_FRAME
        levels.append(10 * math.log10(power / 32768 ** 2) if power else -120.0)
    return levels


def _pause_cut(audio_path: Path, start_s: float, length_s: float) -> float | None:
    """Seconds into the file at the middle of the longest pause in a stretch of
    audio, or None when it has no pause of SPLIT_MIN_PAUSE_SECONDS."""
    levels = _frame_levels_db(audio_path, start_s, length_s)
    if len(levels) < 20:
        return None
    ordered = sorted(levels)
    floor, loud = ordered[len(ordered) // 20], ordered[len(ordered) * 19 // 20]
    threshold = floor + 0.3 * (loud - floor)
    best_len, best_first, run_first = 0, 0, None
    for i, level in enumerate(levels + [math.inf]):
        if level < threshold:
            if run_first is None:
                run_first = i
        elif run_first is not None:
            if i - run_first > best_len:
                best_len, best_first = i - run_first, run_first
            run_first = None
    if best_len * _PAUSE_FRAME_S < SPLIT_MIN_PAUSE_SECONDS:
        return None
    return start_s + (best_first + best_len / 2) * _PAUSE_FRAME_S


def _split_audio(
    audio_path: Path,
    output_dir: Path,
    max_duration_ms: int = MAX_DURATION_MS,
) -> tuple[list[Path], list[int], int, list[int]]:
    """Split audio into chunks using ffmpeg stream copy if needed.

    Returns (chunk_paths, chunk_durations_ms, total_duration_ms, chunk_starts_ms);
    durations and starts come from the split arithmetic itself, so no extra
    ffprobe calls are needed. Each cut goes through a pause where one is found
    (no shared audio); otherwise the next chunk starts OVERLAP_MS before the cut.
    max_duration_ms is a parameter so tests (and the skill's RECITATION
    fallback, which omni does not have) can split at other lengths."""
    duration_ms = _get_audio_duration(audio_path)
    if duration_ms <= max_duration_ms:
        return [audio_path], [duration_ms], duration_ms, [0]

    spans: list[tuple[int, int]] = []
    start_ms = 0
    in_pauses = 0
    while True:
        mark_ms = start_ms + max_duration_ms
        if mark_ms >= duration_ms:
            spans.append((start_ms, duration_ms))
            break
        search_s = min(SPLIT_SEARCH_SECONDS, max_duration_ms / 2000)
        cut = _pause_cut(audio_path, mark_ms / 1000 - search_s, search_s)
        if cut is not None:
            spans.append((start_ms, int(cut * 1000)))
            start_ms = int(cut * 1000)
            in_pauses += 1
        else:
            spans.append((start_ms, mark_ms))
            start_ms = mark_ms - OVERLAP_MS

    chunks = []
    durations: list[int] = []
    starts: list[int] = []
    ext = audio_path.suffix
    for chunk_index, (start_ms, end_ms) in enumerate(spans):
        chunk_duration_ms = end_ms - start_ms
        chunk_path = output_dir / f"chunk_{chunk_index:03d}{ext}"

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_ms / 1000),
            "-i", str(audio_path),
            "-t", str(chunk_duration_ms / 1000),
            "-c", "copy",
            "-avoid_negative_ts", "1",
            "--", str(chunk_path),
        ]
        result = subprocess.run(cmd, capture_output=True, **_FFMPEG_TEXT, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg split failed: {result.stderr}")

        chunks.append(chunk_path)
        durations.append(chunk_duration_ms)
        starts.append(start_ms)

    logger.info(f"Split audio into {len(chunks)} chunks of ~{max_duration_ms // 60000} minutes, "
                f"{in_pauses} of {len(chunks) - 1} cut(s) in a pause")
    return chunks, durations, duration_ms, starts


def _seam_overlaps_s(durations_ms: list[int], starts_ms: list[int]) -> list[float]:
    """Seconds of audio each pair of neighbouring chunks shares (0 for a pause cut)."""
    return [round((s + d - n) / 1000, 3) for s, d, n in zip(starts_ms, durations_ms, starts_ms[1:])]


# Overlap detection for chunk merging (the skill's endpoint-anchored alignment)
_PUNCT_RE = re.compile(
    r'[\s\u3000。，、！？：；「」『』（）【】\u201c\u201d\u2018\u2019'
    r'\u2026\u2014\u00b7\.\,\!\?\:\;\"\'\(\)\[\]\-\n\r\t>]'
)
_OVERLAP_PREV_CLEAN_LIMIT = 600
_OVERLAP_NEXT_CLEAN_LIMIT = 350
_OVERLAP_MIN_LATIN_CHARS = 40
_OVERLAP_MIN_CJK_CHARS = 12
_OVERLAP_MIN_IDENTITY = 0.65
_OVERLAP_MIN_NORMALIZED_SCORE = 0.30
_OVERLAP_TERMINAL_WINDOW = 24
_OVERLAP_MIN_TERMINAL_IDENTITY = 0.60
# Only unambiguous closing punctuation: straight quotes are excluded because
# they cannot be told apart from the opening quote of the NEXT sentence.
_OVERLAP_BOUNDARY_CHARS = ' \t\r\n。.!！?？,，;；:：”’」』)）]】'
# An accepted skip of OVERLAP_WARNING_SKIP characters or more is logged as a
# warning, and a skip over MAX_OVERLAP_SKIP is ignored: a duplicated seam is
# better than deleted content.
OVERLAP_WARNING_SKIP = 250
MAX_OVERLAP_SKIP = 400

# Speaker-label tokens (**Name:** / **Host:** / **Speaker 1:**). Adjacent audio
# chunks label the same overlap speech independently and often differently, so
# label text must not enter the matching or it poisons the seam detection;
# each label is matched as one sentinel character instead.
_SPEAKER_TOKEN_RE = re.compile(r'\*\*[^*\n]{1,40}[:：]\*\*')
_SPEAKER_BOUNDARY = '\0'

# Any **Name:** turn label at the start of a line.
_NAMED_LABEL_RE = re.compile(r'^\*\*([^*\n]+?)\s*[:：]\*\*', re.MULTILINE)
# Role words that look like single-token names but must never be merged into a
# full-name label.
_ROLE_LABEL_WORDS = {
    "host", "guest", "operator", "speaker", "moderator", "interviewer",
    "interviewee", "narrator", "unknown",
}


def _unify_short_name_labels(transcript: str) -> str:
    """Merge first-name-only labels into the matching full-name label.

    Chunks are labeled independently, so a chunk that never hears the speaker
    introductions may label **Jane:** where other chunks used **Jane Doe:**.
    Merge only on an exact, case-insensitive first-token match that is unique
    across all full-name labels; prefix matching would conflate similar names
    (e.g. Spence Neumann vs Spencer Wang). Chinese names have no spaces, so this
    is a no-op for them (the transcription prompt pins their script instead).
    """
    names: list[str] = []
    seen: set[str] = set()
    for m in _NAMED_LABEL_RE.finditer(transcript):
        name = m.group(1).strip()
        key = name.casefold()
        if key not in seen:
            seen.add(key)
            names.append(name)

    multi = [n for n in names if len(n.split()) > 1]
    single = [
        n for n in names
        if len(n.split()) == 1 and n.casefold() not in _ROLE_LABEL_WORDS
    ]
    if not multi or not single:
        return transcript

    remap: dict[str, str] = {}
    for s in single:
        matches = [m for m in multi if m.split()[0].casefold() == s.casefold()]
        if len(matches) == 1:
            remap[s.casefold()] = matches[0]

    if not remap:
        return transcript

    def _sub(m: re.Match) -> str:
        full = remap.get(m.group(1).strip().casefold())
        return f"**{full}:**" if full else m.group(0)

    return _NAMED_LABEL_RE.sub(_sub, transcript)


def _unify_prefix_name_labels(transcript: str) -> str:
    """Merge full-name labels that differ only by a shortened first name.

    The same person can come back as **Spencer Neumann:** in one chunk and
    **Spence Neumann:** in another (Netflix Q2 2026 run in the transcribe
    skill, 2026-09-11), which the first-name merge above does not cover. Two
    labels merge when everything after the first name matches exactly and one
    first name is a prefix of the other; the label with more turns wins. A
    surname carried by a third label is ambiguous and left alone.
    """
    counts: dict[str, int] = {}
    spelling: dict[str, str] = {}
    for m in _NAMED_LABEL_RE.finditer(transcript):
        name = m.group(1).strip()
        key = name.casefold()
        spelling.setdefault(key, name)
        counts[key] = counts.get(key, 0) + 1

    by_surname: dict[str, list[str]] = {}
    for key in spelling:
        parts = key.split()
        if len(parts) >= 2 and parts[0] not in _ROLE_LABEL_WORDS:
            by_surname.setdefault(" ".join(parts[1:]), []).append(key)

    remap: dict[str, str] = {}
    for keys in by_surname.values():
        if len(keys) != 2:
            continue
        a, b = keys
        fa, fb = a.split()[0], b.split()[0]
        if fa == fb or not (fa.startswith(fb) or fb.startswith(fa)):
            continue
        winner, loser = (a, b) if counts[a] >= counts[b] else (b, a)
        remap[loser] = spelling[winner]
    if not remap:
        return transcript

    def _sub(m: re.Match) -> str:
        full = remap.get(m.group(1).strip().casefold())
        return f"**{full}:**" if full else m.group(0)

    return _NAMED_LABEL_RE.sub(_sub, transcript)


def _strip_for_matching_with_map(text: str) -> tuple[str, list[int]]:
    """Normalize text for fuzzy matching while retaining speaker boundaries.

    All speaker labels become one sentinel, so independently assigned names can
    still align, while a transition into a new speaker cannot be mistaken for
    ordinary text. Mapping entries project each normalized character back to
    its source position, including characters expanded by Unicode casefolding.
    """
    clean_chars: list[str] = []
    mapping: list[int] = []

    def append_text(start: int, end: int) -> None:
        for idx in range(start, end):
            ch = text[idx]
            if not _PUNCT_RE.match(ch):
                folded = ch.casefold()
                # Adjacent chunks can come back in different scripts (one
                # traditional, one simplified), and an unfolded seam then
                # finds no overlap and keeps both copies of it.
                if _zh_fold is not None and "㐀" <= ch <= "鿿":
                    simple = _zh_fold(ch, "zh-cn")
                    if len(simple) == 1:
                        folded = simple
                clean_chars.extend(folded)
                mapping.extend([idx] * len(folded))

    cursor = 0
    for match in _SPEAKER_TOKEN_RE.finditer(text):
        append_text(cursor, match.start())
        clean_chars.append(_SPEAKER_BOUNDARY)
        mapping.append(match.end() - 1)
        cursor = match.end()
    append_text(cursor, len(text))
    return ''.join(clean_chars), mapping


def _strip_for_matching(text: str) -> str:
    """Strip whitespace, punctuation, and speaker labels for fuzzy overlap matching."""
    return _strip_for_matching_with_map(text)[0]


def _is_cjk_heavy(text: str) -> bool:
    if not text:
        return False
    cjk = sum('㐀' <= ch <= '鿿' for ch in text)
    return cjk >= 8 and cjk / len(text) >= 0.4


def _find_overlap_length(prev_text: str, next_text: str) -> int:
    """Return the raw prefix length in next_text that overlaps prev_text's suffix.

    The alignment is semi-global: it may skip any prefix of prev_text for free,
    but it must consume a prefix of next_text and end at prev_text's final clean
    character. This endpoint constraint prevents repeated operator scripts from
    matching unrelated text elsewhere in the previous chunk.
    """
    prev_clean = _strip_for_matching(prev_text)[-_OVERLAP_PREV_CLEAN_LIMIT:]
    next_clean, next_map = _strip_for_matching_with_map(next_text)
    next_clean = next_clean[:_OVERLAP_NEXT_CLEAN_LIMIT]
    next_map = next_map[:len(next_clean)]

    min_overlap = _OVERLAP_MIN_CJK_CHARS
    if len(prev_clean) < min_overlap or len(next_clean) < min_overlap:
        return 0

    match_score = 2
    edit_penalty = -2
    rows = len(prev_clean) + 1
    cols = len(next_clean) + 1
    scores = [[0] * cols for _ in range(rows)]
    moves = [bytearray(cols) for _ in range(rows)]

    # Free deletion of prev_clean's prefix lets the alignment start at any
    # suffix position. Consuming next_clean before prev_clean starts is costly.
    for j in range(1, cols):
        scores[0][j] = scores[0][j - 1] + edit_penalty
        moves[0][j] = 2  # consume next character
    for i in range(1, rows):
        moves[i][0] = 1  # consume previous character

    for i in range(1, rows):
        prev_ch = prev_clean[i - 1]
        for j in range(1, cols):
            diagonal = scores[i - 1][j - 1] + (
                match_score if prev_ch == next_clean[j - 1] else edit_penalty
            )
            consume_prev = scores[i - 1][j] + edit_penalty
            consume_next = scores[i][j - 1] + edit_penalty
            best = max(diagonal, consume_prev, consume_next)
            scores[i][j] = best
            moves[i][j] = (
                0 if diagonal == best
                else 1 if consume_prev == best
                else 2
            )

    best_end = 0
    best_rank: tuple[float, float, int, int] | None = None
    last_row = rows - 1
    for end in range(min_overlap, cols):
        candidate = next_clean[:end]
        required_overlap = (
            _OVERLAP_MIN_CJK_CHARS if _is_cjk_heavy(candidate)
            else _OVERLAP_MIN_LATIN_CHARS
        )
        if end < required_overlap:
            continue
        i = last_row
        j = end
        matches = 0
        edits = 0
        columns = 0
        terminal_columns = 0
        terminal_matches = 0
        speaker_violation = False
        while j > 0:
            move = moves[i][j]
            is_match = False
            if move == 0:
                next_ch = next_clean[j - 1]
                is_match = prev_clean[i - 1] == next_ch
                # A speaker boundary inside the overlap must align with a
                # boundary on the previous side. A boundary consumed against
                # ordinary text marks the start of a NEW turn, and deleting
                # past it would swallow the new speaker's speech. The leading
                # boundary of next (j == 1) is the overlap's own label and is
                # exempt, mirroring the old label-masking behavior.
                if next_ch == _SPEAKER_BOUNDARY and j > 1 and not is_match:
                    speaker_violation = True
                    break
                if is_match:
                    matches += 1
                else:
                    edits += 1
                i -= 1
                j -= 1
            elif move == 1:
                edits += 1
                i -= 1
            else:
                if next_clean[j - 1] == _SPEAKER_BOUNDARY and j > 1:
                    speaker_violation = True
                    break
                edits += 1
                j -= 1
            if terminal_columns < _OVERLAP_TERMINAL_WINDOW:
                terminal_columns += 1
                if is_match:
                    terminal_matches += 1
            columns += 1

        if speaker_violation or columns == 0:
            continue
        identity = matches / columns
        terminal_identity = terminal_matches / terminal_columns
        normalized_score = scores[last_row][end] / (match_score * columns)
        if (identity < _OVERLAP_MIN_IDENTITY
                or terminal_identity < _OVERLAP_MIN_TERMINAL_IDENTITY
                or normalized_score < _OVERLAP_MIN_NORMALIZED_SCORE):
            continue

        # Identity is the primary confidence signal. Score and length only
        # break ties, so a second repeated template cannot win merely by being
        # longer than the actual suffix-prefix overlap.
        rank = (identity, normalized_score, scores[last_row][end], -end)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_end = end

    if best_end == 0:
        return 0

    # Mapping projects the clean-text boundary back into the original string,
    # preserving speaker labels and punctuation. Consume only punctuation and
    # whitespace immediately following the matched character. Do not search for
    # a later sentence boundary, which could delete the beginning of new speech.
    orig_pos = next_map[best_end - 1] + 1
    while orig_pos < len(next_text) and next_text[orig_pos] in _OVERLAP_BOUNDARY_CHARS:
        orig_pos += 1
    return orig_pos


def _merge_transcriptions(transcripts: list[str], seam_overlaps_s: list[float] | None = None) -> str:
    """
    Merge multiple transcript chunks, removing duplicated content at boundaries.

    seam_overlaps_s gives each seam's shared audio (see _seam_overlaps_s): a
    seam cut in a pause (0) shares nothing and is joined as is. Every other
    seam, and all of them when seam_overlaps_s is None, is an endpoint-anchored
    alignment of the previous chunk's tail against the next chunk's head (see
    _find_overlap_length), which tolerates the minor wording differences Gemini
    produces when the same audio appears in two overlapping chunks.

    Args:
        transcripts: List of transcript strings to merge
        seam_overlaps_s: Seconds of audio each pair of neighbouring chunks shares

    Returns:
        Merged transcript string
    """
    if len(transcripts) == 1:
        return transcripts[0]

    merged = transcripts[0]

    for i in range(1, len(transcripts)):
        current = transcripts[i]
        if seam_overlaps_s is not None and i - 1 < len(seam_overlaps_s) and not seam_overlaps_s[i - 1]:
            merged = merged + "\n\n" + current
            continue
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
            message = f"Removed {skip} chars of overlapping content at chunk {i}"
            if skip >= OVERLAP_WARNING_SKIP:
                logger.warning(message)
            else:
                logger.info(message)
        elif raw_skip == 0:
            # A capped skip (raw_skip > limit) is a different, already-warned case;
            # only a genuine miss means the ~OVERLAP_SECONDS overlap survived.
            logger.warning(
                f"No overlap detected at chunk seam {i}; "
                f"~{OVERLAP_SECONDS}s of audio may be duplicated there"
            )
        merged = merged + "\n\n" + current

    # Chunks label speakers independently, so a later chunk may fall back to a
    # first name or shorten one (Spence / Spencer); unify to full names after
    # every seam is stitched.
    return _unify_prefix_name_labels(_unify_short_name_labels(merged))


async def transcribe(
    audio_path: str,
    config: TranscriberConfig,
    metadata: Optional[TranscriptionMetadata] = None,
    on_status: Callable[[str], None] | None = None,
    podcast_mode: bool = False,
) -> str:
    """
    Transcribe audio file using Gemini API.
    Supports automatic chunking for audio longer than 45 minutes.

    Args:
        audio_path: Path to the audio file to transcribe
        config: Transcriber configuration
        metadata: Optional metadata for podcast mode (provides context for better transcription)
        on_status: Optional callback to report status updates
        podcast_mode: Content form of the source. True for conversational audio
            (podcasts, meetings, uploaded recordings), which gets strong speaker
            labels even without metadata; False for single-track video.

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
        # Split audio if needed (ffmpeg stream copy, cut in pauses)
        temp_dir = Path(tempfile.mkdtemp(prefix="omni_chunks_"))
        # The pause search decodes a minute of audio before each cut, so the
        # split runs off the event loop the bot and the Dropbox watcher share.
        chunk_paths, chunk_durations_ms, duration_ms, chunk_starts_ms = await asyncio.to_thread(
            _split_audio, audio_path_obj, temp_dir
        )
        chunk_durations_s = [d / 1000 for d in chunk_durations_ms]
        seam_overlaps_s = _seam_overlaps_s(chunk_durations_ms, chunk_starts_ms)
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
        # Settle every upload before proceeding: a bare gather propagates the
        # first failure while sibling uploads are still in flight, so any that
        # land afterward would never be recorded in uploaded_files and the
        # finally block could not delete them (they leak on Gemini until the 48h
        # auto-expiry, counting against the storage quota). Collect the
        # successes so cleanup can reach them; a chunk whose upload failed is
        # sent with its audio inline instead (None in wave_files).
        upload_results = await asyncio.gather(*upload_tasks, return_exceptions=True)
        wave_files: List[types.File | None] = []
        cancelled = None
        for chunk_path, result in zip(chunk_paths, upload_results):
            # Classify by BaseException, not Exception: a CancelledError is a
            # BaseException and must not slip through as a "successful" upload
            # (it would be appended and then blow up in the finally cleanup).
            # Record every real upload first so cleanup can reach it, then
            # re-raise a cancellation as-is.
            if isinstance(result, asyncio.CancelledError):
                cancelled = result
            elif isinstance(result, BaseException):
                logger.warning(
                    f"Upload of {chunk_path.name} failed after retries ({result}); "
                    "this chunk will be sent inline instead"
                )
                wave_files.append(None)
            else:
                uploaded_files.append(result)
                wave_files.append(result)
        if cancelled is not None:
            raise cancelled
        logger.info(f"Uploaded {len(uploaded_files)} of {len(chunk_paths)} files")

        # Step 2: Transcribe all chunks in parallel
        if on_status:
            if len(chunk_paths) > 1:
                on_status(f"Transcribing {len(chunk_paths)} chunks in parallel...")
            else:
                on_status("Transcribing...")

        async def transcribe_chunk(
            uploaded_file: types.File | None, chunk_path: Path, duration_s: float, label: str
        ) -> str:
            # `label` ("chunk 2/4") goes into every log line, so a failure names
            # the request it came from. Gate around the call, outside
            # _transcribe_audio's own wait_for, so queue time doesn't burn the
            # per-chunk timeout budget.
            def _attempts_from(audio_source):
                def _attempt_on(model_id: str):
                    async def _attempt():
                        async with _gemini_sem:
                            return await _transcribe_audio(
                                client,
                                audio_source,
                                model_id,
                                config.temperature,
                                config.thinking_level,
                                metadata,
                                podcast_mode,
                                label=label,
                            )
                    return _attempt
                return _attempt_on

            async def _inline_pass() -> str:
                part = await _inline_audio_part(
                    chunk_path, temp_dir,
                    MIME_TYPES.get(chunk_path.suffix.lower(), mime_type),
                )
                return await with_model_fallback(
                    _attempts_from(part), config.model,
                    context=f"Transcription {label} (inline)",
                )

            async def _one_pass() -> str:
                if uploaded_file is None:
                    return await _inline_pass()
                try:
                    return await with_model_fallback(
                        _attempts_from(uploaded_file), config.model,
                        context=f"Transcription {label}",
                        non_retryable_exceptions=(FileHandleRejectedError,),
                    )
                except ResponseBlockedError:
                    # Both models blocked it. Inlined audio was blocked the same
                    # way, since the block follows the request, not the handle.
                    raise
                except FileHandleRejectedError as error:
                    logger.warning(
                        f"Transcription {label}: the uploaded file was rejected with 403 ({error}); "
                        "sending the audio inline instead"
                    )
                    return await _inline_pass()
                except Exception as error:
                    # A File API handle can be rejected independently of the
                    # model: on 2026-08-14 every uploaded-file request returned
                    # 403 PERMISSION_DENIED on both models while the same audio
                    # sent inline transcribed fine. Swapping models cannot fix
                    # that, so the last resort drops the file reference entirely.
                    logger.warning(
                        f"Transcription {label}: the File API path failed on every model "
                        f"({error}); retrying with the audio inlined in the request"
                    )
                    return await _inline_pass()

            # Degenerate-output guard: a chunk that transcribes to almost nothing
            # (e.g. just a speaker's name) is retried. Measured on the cleaned
            # length so a repetitive-character loop can't inflate past the floor.
            # with_retry handles empty responses (it raises and retries); this
            # outer loop handles the non-empty-but-degenerate case, and accepts
            # the last short result so legitimately sparse audio (music, long
            # silence) doesn't fail the whole run.
            min_chars = max(10, int(duration_s * MIN_TRANSCRIPT_CHARS_PER_SEC))
            text = ""
            for attempt in range(1, TRANSCRIPTION_SHORT_RETRIES + 1):
                try:
                    text = await _one_pass()
                except Exception as error:
                    # Logged here because the gather below raises only the first error.
                    logger.error(f"Transcription {label} failed: {error}")
                    raise
                cleaned_len = len(cleanup_repetitive_characters(text).strip())
                if duration_s <= 0 or cleaned_len >= min_chars:
                    return text
                if attempt < TRANSCRIPTION_SHORT_RETRIES:
                    logger.warning(
                        f"Transcript {label} suspiciously short ({cleaned_len} chars for "
                        f"{duration_s:.0f}s of audio, expected >= {min_chars}); "
                        f"retrying ({attempt}/{TRANSCRIPTION_SHORT_RETRIES})"
                    )
            logger.warning(
                f"Transcript {label} still short after {TRANSCRIPTION_SHORT_RETRIES} attempts; "
                "accepting it (audio may be legitimately sparse)"
            )
            return text

        transcribe_tasks = [
            transcribe_chunk(f, p, d, f"chunk {i + 1}/{len(chunk_paths)}")
            for i, (f, p, d) in enumerate(zip(wave_files, chunk_paths, chunk_durations_s))
        ]
        # Settle every chunk before raising: a bare gather propagates the first
        # failure while sibling transcriptions are still in flight, and the
        # finally block would then delete their uploaded files out from under
        # them (each orphan burns its retries against a now-deleted file while
        # holding a shared Gemini slot). return_exceptions=True lets them all
        # finish first; a cancellation propagates, otherwise the first real
        # error is raised (one chunk failing still fails the whole run).
        results = await asyncio.gather(*transcribe_tasks, return_exceptions=True)
        cancelled = next(
            (r for r in results if isinstance(r, asyncio.CancelledError)), None
        )
        if cancelled is not None:
            raise cancelled
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            raise errors[0]
        transcripts = list(results)
        logger.info(f"Transcribed {len(transcripts)} chunks")

        # Step 3: Clean up each transcript. Chunks are folded to simplified script
        # before the merge, so a chunk that came back traditional cannot leave
        # the transcript mixing scripts.
        transcripts = [_to_simplified(cleanup_repetitive_characters(t)) for t in transcripts]

        # Step 4: Merge if multiple chunks
        if len(transcripts) > 1:
            if on_status:
                on_status("Merging transcriptions...")
            # About 80ms of pure-Python alignment per seam, so run it off the
            # event loop the bot and the Dropbox watcher share.
            full_text = await asyncio.to_thread(_merge_transcriptions, transcripts, seam_overlaps_s)
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
    uploaded_file: types.File | types.Part,
    model: str,
    temperature: float,
    thinking_level: str,
    metadata: Optional[TranscriptionMetadata] = None,
    podcast_mode: bool = False,
    label: str = "chunk",
) -> str:
    """Transcribe audio using Gemini model."""
    source = getattr(uploaded_file, "name", None) or "inline audio"
    logger.info(f"Transcribing {label} ({source}) on {model}...")

    # Build prompt with or without metadata context
    prompt = _build_transcription_prompt(metadata, podcast_mode)
    # Either a File API handle or an already-built inline part, so the inline
    # fallback reuses this function rather than duplicating prompt assembly.
    is_file_ref = not isinstance(uploaded_file, types.Part)
    audio_part = (
        types.Part.from_uri(file_uri=uploaded_file.uri, mime_type=uploaded_file.mime_type)
        if is_file_ref else uploaded_file
    )

    def _generate():
        return client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt), audio_part],
                )
            ],
            config=types.GenerateContentConfig(
                temperature=temperature,
                # Gemini 3.x effort dial (low/medium/high, mutually exclusive
                # with the legacy thinking_budget).
                thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
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
    except Exception as e:
        if is_file_ref and getattr(e, "code", None) == 403:
            raise FileHandleRejectedError(str(e)) from e
        raise
    _check_served_model(model, getattr(response, "model_version", None))

    # Validate response
    text = response.text
    if not text or text.strip() == "":
        block_reason = getattr(response.prompt_feedback, "block_reason", None)
        if block_reason is not None:
            raise ResponseBlockedError(
                f"Gemini blocked the response on {model} "
                f"(block_reason={getattr(block_reason, 'name', None) or block_reason})"
            )
        raise ValueError("Transcription returned empty result.")

    # Warn only: a long chunk hitting the output cap is not fixable by a retry,
    # and the partial transcript is still worth keeping.
    if is_truncated(response):
        logger.warning(f"Transcription {label} hit max output tokens on {model}; it may be truncated")

    return text


async def _delete_file(client: genai.Client, file_name: str) -> None:
    """Delete a file from Gemini File API."""
    await asyncio.to_thread(client.files.delete, name=file_name)
