"""Transcript editor using Gemini API with two-step processing."""

import asyncio
import json
import re
import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

from google import genai
from google.genai import types

from ..config import EditorConfig
from ..utils.gemini import is_truncated
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


# Chunk size for transcript editing (characters)
CHUNK_SIZE = 32000  # 32K characters per chunk for raw transcript editing
CHUNK_OVERLAP = 500  # Overlap to avoid cutting mid-sentence
MIN_MEANINGFUL_CONTENT = 50  # Minimum chars of actual text (excluding timestamps/sound effects)

# Text-only Gemini calls (edit/metadata/translate). The SDK sets no request
# timeout of its own, so without this a hung connection stalls forever (the
# same class of hang that f14ebf4 fixed on the transcription side).
GENERATION_TIMEOUT_SECONDS = 10 * 60


async def _gemini_generate(
    client: genai.Client,
    model: str,
    contents,
    *,
    system: str | None = None,
    temperature: float = 1.0,
    thinking_budget: int = 1024,
    max_tokens: int | None = None,
    schema: dict | None = None,
    timeout: float = GENERATION_TIMEOUT_SECONDS,
):
    """Shared generate_content wrapper: one place for temperature, thinking,
    schema plumbing, and a hard timeout. Post-call validation stays at the
    call sites, where the fallback semantics differ per stage."""
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        max_output_tokens=max_tokens,
        response_mime_type="application/json" if schema else None,
        response_schema=schema,
    )

    def _generate():
        return client.models.generate_content(model=model, contents=contents, config=config)

    try:
        return await asyncio.wait_for(asyncio.to_thread(_generate), timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Gemini call timed out after {int(timeout) // 60} minutes")


@dataclass
class VideoEditorMetadata:
    """Metadata for video sources (YouTube/Bilibili) used in editing."""
    title: str
    channel: str
    description: str = ""
    upload_date: str = ""
    webpage_url: str = ""


def _has_meaningful_content(chunk: str) -> bool:
    """Check if a chunk has meaningful content beyond just timestamps/sound effects.

    Args:
        chunk: The transcript chunk to check

    Returns:
        True if chunk has enough meaningful text content
    """
    # Remove timestamps like [00:00:00] or [HH:MM:SS]
    text = re.sub(r'\[\d{1,2}:\d{2}:\d{2}\]', '', chunk)
    # Remove sound effects like [MUSIC PLAYING], [APPLAUSE], etc.
    text = re.sub(r'\[[A-Z][A-Z\s]+\]', '', text)
    # Remove speaker labels like **Name:**
    text = re.sub(r'\*\*[^*]+\*\*:', '', text)
    # Remove whitespace
    text = text.strip()

    return len(text) >= MIN_MEANINGFUL_CONTENT


def _normalize_quotes(text: str) -> str:
    """Normalize smart/curly quotes to ASCII equivalents.

    Gemini models output smart quotes (U+2018/2019/201C/201D) which render
    as overly wide characters in monospace fonts and PDFs.
    """
    return text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201C", '"').replace("\u201D", '"')


USER_PROMPT_PREFIX = "Here's the transcript:\n\n"

# Default sections for metadata generation (used when no CONTEXT.md override)
DEFAULT_SECTIONS = """Output a JSON object with these fields:
- "title": string - 简短的中文标题，概括内容主题 (5-15个中文字符，仅中文、数字和基本标点)
- "summary": string - 中文摘要，概括核心内容和关键结论 (不超过300字)
- "key_points": array of strings - 中文要点列表，关注可执行事项、决策和重要信息 (最多20条)"""

# System prompt template for metadata generation (JSON output)
METADATA_SYSTEM_PROMPT_TEMPLATE = """You are a professional transcript analyst. Given an edited transcript, generate structured metadata in JSON format.

## Required Output

{sections}

## Language Rules
- Unless specified otherwise in the field instructions, output title/summary/key_points in Chinese (简体中文)

## Rules
- Output ONLY a valid JSON object
- Do NOT wrap in markdown code blocks or any other formatting
- Ensure all strings are properly escaped
- Arrays should contain strings only
"""

# Section heading mappings (field name → display heading)
SECTION_HEADINGS = {
    "summary": "Summary",
    "key_points": "Key Points",
    "decisions": "Decisions",
    "action_items": "Action Items",
    "key_concepts": "Key Concepts",
    "questions": "Questions",
    "highlights": "Highlights",
    "homework": "Homework",
    "qa": "Q&A Highlights",
}

# Translation prompt for translating transcript text
TRANSLATION_SYSTEM_PROMPT = """You are a professional translator. Add inline Chinese translations to the following transcript.

## Rules
- First determine if the text is primarily in Chinese. If yes, return it UNCHANGED.
- For non-Chinese text: after each paragraph, add a blockquote with the Chinese translation
- Format:
  Original paragraph text.
  > 中文翻译。

  Next paragraph.
  > 下一段翻译。

- Translate meaning accurately, not word-for-word
- Keep proper nouns, technical terms, brand names unchanged
- Output ONLY the processed text, no explanations or commentary
"""


async def _generate_normal_metadata(
    client: genai.Client,
    transcript: str,
    sections: str,
    model: str,
    temperature: float,
    thinking_budget: int,
    metadata: Optional["VideoEditorMetadata"] = None,
    background: str = "",
) -> dict:
    """Generate metadata JSON from edited transcript.

    Args:
        client: Gemini API client
        transcript: The edited transcript text
        sections: Section definitions (from CONTEXT.md or DEFAULT_SECTIONS)
        model: Model to use
        temperature: Temperature setting
        thinking_budget: Thinking budget
        metadata: Optional video metadata for additional context
        background: Optional background context from CONTEXT.md

    Returns:
        Dictionary with metadata fields (title, summary, key_points, etc.)
    """
    logger.info("Generating metadata from edited transcript...")

    # Build user content with context
    parts = []

    if metadata:
        parts.append("## Source Information")
        parts.append(f"- Channel/Source: {metadata.channel}")
        parts.append(f"- Title: {metadata.title}")
        if metadata.upload_date:
            parts.append(f"- Date: {metadata.upload_date}")
        if metadata.description:
            parts.append(f"- Description: {metadata.description}")
        parts.append("")
    elif background:
        parts.append("## Background Information")
        parts.append(background)
        parts.append("")

    parts.append("## Transcript")
    parts.append("")

    # Truncate very long transcripts for metadata generation
    max_transcript_chars = 500000
    if len(transcript) > max_transcript_chars:
        logger.warning(f"Transcript too long ({len(transcript)} chars), truncating for metadata generation")
        transcript_truncated = transcript[:max_transcript_chars] + "\n\n[... transcript truncated ...]"
        parts.append(transcript_truncated)
    else:
        parts.append(transcript)

    user_content = "\n".join(parts)

    # Build system prompt with sections
    system_prompt = METADATA_SYSTEM_PROMPT_TEMPLATE.format(sections=sections)

    # Gemini 3 counts thinking tokens against max_output_tokens, so the cap
    # must hold the thinking budget (8192 on this path) PLUS the JSON itself.
    response = await _gemini_generate(
        client, model, user_content,
        system=system_prompt, temperature=temperature,
        thinking_budget=thinking_budget, max_tokens=16384,
    )

    if not response.text:
        raise ValueError("Empty metadata response")
    if is_truncated(response):
        raise ValueError("Metadata generation hit max output tokens; JSON is incomplete")

    # Parse JSON from response
    response_text = response.text.strip()

    # Strategy 1: Try parsing entire response directly (ideal case - LLM followed instructions)
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from ```json ... ``` code block
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: Extract from ``` ... ``` code block
    json_match = re.search(r'```\s*([\{\[].*?[\}\]])\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 4: Find first '{' and try parsing progressively longer substrings
    # This handles cases where LLM adds explanation after JSON
    first_brace = response_text.find('{')
    if first_brace != -1:
        # Try parsing from first '{' to each subsequent '}'
        brace_count = 0
        for i, char in enumerate(response_text[first_brace:], start=first_brace):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found matching closing brace
                    candidate = response_text[first_brace:i+1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        continue  # Try next closing brace

    raise ValueError(f"No valid JSON found in metadata output: {response_text[:500]}")


def _build_normal_result(
    metadata_dict: dict,
    edited_transcript: str,
) -> str:
    """Build final Markdown from metadata dict and edited transcript.

    Args:
        metadata_dict: Dictionary with title, summary, key_points, etc.
        edited_transcript: The edited transcript text

    Returns:
        Final Markdown string
    """
    parts = []

    # Extract title (special handling - becomes h1)
    title = metadata_dict.get("title", "Transcript")
    parts.append(f"# {title}")
    parts.append("")

    # Process remaining fields in order
    for key, value in metadata_dict.items():
        # Skip title (already handled) and internal/private fields
        if key == "title" or key.startswith("_"):
            continue

        # Get display heading
        heading = SECTION_HEADINGS.get(key, key.replace("_", " ").title())
        parts.append(f"## {heading}")
        parts.append("")

        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    # Handle list of dicts: join all key-values with <br> followed by two spaces
                    kv_parts = [f"**{k.upper()}**: {v}" for k, v in item.items()]
                    parts.append(f"- {'<br>  '.join(kv_parts)}")
                else:
                    parts.append(f"- {item}")
        elif isinstance(value, dict):
            # Handle nested dicts as key: value pairs
            for k, v in value.items():
                parts.append(f"- **{k}**: {v}")
        else:
            parts.append(str(value))

        parts.append("")

    # Add transcript section
    parts.append("## Transcript")
    parts.append("")
    parts.append(edited_transcript)

    return "\n".join(parts)


async def _translate_transcript(
    client: genai.Client,
    transcript: str,
    model: str,
    temperature: float,
    thinking_budget: int,
) -> str:
    """Add inline Chinese translations to non-Chinese transcript.

    Args:
        client: Gemini API client
        transcript: The transcript text to translate
        model: Model to use
        temperature: Temperature setting
        thinking_budget: Thinking budget

    Returns:
        Transcript with inline translations (or unchanged if already Chinese)
    """
    logger.info("Adding inline translations to transcript...")

    response = await _gemini_generate(
        client, model, transcript,
        system=TRANSLATION_SYSTEM_PROMPT, temperature=temperature,
        thinking_budget=thinking_budget, max_tokens=65536,
    )

    if not response.text:
        logger.warning("Empty translation response, returning original transcript")
        return transcript
    # A truncated translation replaces the full transcript with a shorter one,
    # silently dropping the tail; raise so with_retry gets a chance.
    if is_truncated(response):
        raise ValueError("Translation hit max output tokens; output is incomplete")

    return response.text.strip()

# Raw transcript editing prompt - corrects errors and removes fillers
RAW_EDIT_SYSTEM_PROMPT = """You are a professional transcript editor. Your task is to clean up this raw transcript chunk.

## Your Tasks

1. **修正转录错误**：
   - 修正明显的语音转文字错误（音近字、同音字）
   - 修正专业术语、公司名称、人名的转录错误
   - 保留专有名词、产品名称、版本号、技术术语的原样

2. **移除填充词**：
   - 移除中文填充词：嗯、啊、呃、额、哦、唔、那个、就是、然后、对对对、是是是
   - 移除英文填充词：um, uh, like, you know, I mean, sort of, kind of, basically
   - 保留有实际语义的语气词和感叹词

3. **格式规范**：
   - 适当分段提高可读性
   - 按说话人变化或话题变化自然分段
   - 移除重复的口吃表达

4. **保持完整**：
   - 输出完整内容，绝对不要截断或省略任何部分
   - 保持原语言，不要翻译
   - 保留技术术语、专有名词

## Output Format

直接输出编辑后的转录文本，不要添加任何标题、说明或JSON。
只输出编辑后的内容。

## CRITICAL RULES
- 必须输出完整内容，不能省略任何部分
- 不要添加任何前言或总结
- 直接输出编辑后的转录文本
"""

# Raw transcript editing prompt WITH context (for YouTube/Bilibili with metadata)
RAW_EDIT_WITH_CONTEXT_SYSTEM_PROMPT = """You are a professional transcript editor. Your task is to clean up this raw transcript chunk.

## Video Context
Use the video information provided to help identify correct speaker names, channel hosts, and proper nouns.

## Your Tasks

1. **修正转录错误**：
   - 修正明显的语音转文字错误（音近字、同音字）
   - 修正说话人名字错误（根据元数据中的频道名和描述推断正确名字）
   - 修正专业术语、公司名称、人名的转录错误
   - 保留专有名词、产品名称、版本号、技术术语的原样

2. **移除填充词**：
   - 移除中文填充词：嗯、啊、呃、额、哦、唔、那个、就是、然后、对对对、是是是
   - 移除英文填充词：um, uh, like, you know, I mean, sort of, kind of, basically
   - 保留有实际语义的语气词和感叹词

3. **格式规范**：
   - 适当分段提高可读性
   - 按说话人变化或话题变化自然分段
   - 移除重复的口吃表达

4. **保持完整**：
   - 输出完整内容，绝对不要截断或省略任何部分
   - 保持原语言，不要翻译
   - 保留技术术语、专有名词

## Output Format

直接输出编辑后的转录文本，不要添加任何标题、说明或JSON。
只输出编辑后的内容。

## CRITICAL RULES
- 必须输出完整内容，不能省略任何部分
- 不要添加任何前言或总结
- 直接输出编辑后的转录文本
"""

TRANSLATION_PROMPT_ADDITION = """

## Translation Mode (ENABLED)
Since translation mode is enabled, you must add inline Chinese translations to the Transcript section:

1. **Detect language**: First determine if the transcript is primarily in Chinese
2. **If NOT Chinese**: After each paragraph in the Transcript section, add a blockquote with the Chinese translation
3. **If Chinese**: No translation needed, output normally

### Translation Format
For non-Chinese transcripts, format each paragraph like this:
```
Original paragraph text here.
> 这里是中文翻译。

Next paragraph in original language.
> 下一段的中文翻译。
```

### Translation Requirements
- Translate the meaning accurately, not word-for-word
- Maintain the same paragraph structure
- Use `> ` (blockquote) for all translations
- Keep translations natural and readable in Chinese
"""


def _split_into_chunks(transcript: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    Split transcript into manageable chunks for editing.

    Tries to split at dialogue boundaries (speaker changes) to avoid
    cutting in the middle of a speaker's turn. Falls back to paragraph
    boundaries if no speaker patterns found.

    Args:
        transcript: The transcript text to split
        chunk_size: Maximum size of each chunk

    Returns:
        List of transcript chunks
    """
    if len(transcript) <= chunk_size:
        return [transcript]

    chunks = []

    # Pattern to match speaker labels like "**Speaker Name:**" or "Speaker Name:"
    speaker_pattern = re.compile(r'\n(?=(?:\*\*)?[^*\n:]{1,30}(?:\*\*)?:\s)')

    # Find all speaker change positions
    speaker_positions = [0] + [m.start() + 1 for m in speaker_pattern.finditer(transcript)]
    speaker_positions.append(len(transcript))

    if len(speaker_positions) <= 2:
        # No speaker labels found, fall back to paragraph splitting
        paragraphs = re.split(r'\n\n+', transcript)
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para) + 2  # +2 for \n\n

            if current_size + para_size > chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
    else:
        # Split at speaker boundaries
        current_start = 0
        current_chunk_end = 0

        for pos in speaker_positions[1:]:
            segment_end = pos

            if segment_end - current_start > chunk_size:
                if current_chunk_end > current_start:
                    # Save current chunk up to previous speaker boundary
                    chunks.append(transcript[current_start:current_chunk_end].strip())
                    current_start = current_chunk_end
                else:
                    # Single segment exceeds chunk_size, force split at paragraph
                    search_start = current_start + chunk_size - 500
                    search_end = min(current_start + chunk_size + 500, len(transcript))
                    paragraph_break = transcript.rfind('\n\n', search_start, search_end)
                    if paragraph_break > current_start:
                        chunks.append(transcript[current_start:paragraph_break].strip())
                        current_start = paragraph_break + 2
                    else:
                        # No paragraph break, split at chunk_size
                        chunks.append(transcript[current_start:current_start + chunk_size].strip())
                        current_start = current_start + chunk_size

            current_chunk_end = segment_end

        # Don't forget the last chunk
        if current_start < len(transcript):
            remaining = transcript[current_start:].strip()
            if remaining:
                chunks.append(remaining)

    # Filter out empty chunks
    chunks = [c for c in chunks if c.strip()]

    logger.info(f"Split transcript into {len(chunks)} chunks")
    return chunks


async def edit(
    transcript: str,
    config: EditorConfig,
    sections_override: str | None = None,
    enable_translation: bool = False,
    metadata: Optional[VideoEditorMetadata] = None,
    background: str = "",
    on_status: Callable[[str], None] | None = None,
) -> str:
    """
    Edit and format a transcript using Gemini API with multi-step processing.

    Step 1: Edit raw transcript in chunks (parallel) - fix errors, remove fillers
    Step 2: Generate metadata JSON (title, summary, key_points) from edited transcript
    Step 2b (optional): Add inline translations if enabled
    Step 3: Assemble final Markdown from metadata + edited transcript

    Args:
        transcript: The raw transcript text to edit
        config: Editor configuration
        sections_override: Optional override for section definitions (from CONTEXT.md)
        enable_translation: If True, add inline Chinese translations for non-Chinese transcripts
        metadata: Optional video metadata for context (helps with speaker names, proper nouns)
        background: Optional background context from CONTEXT.md (for Dropbox watcher mode)
        on_status: Optional callback to report status updates

    Returns:
        The edited and formatted transcript as Markdown
    """
    if not config.api_key:
        raise ValueError("Editor API key is not configured")

    logger.info("Starting transcript editing (two-step processing)...")

    # Get shared client (reuses HTTP connection pool)
    client = _get_client(config.api_key)

    # Configure thinking budgets for different steps
    # Raw editing and translation use low thinking (simple cleanup tasks)
    # Metadata generation uses high thinking (needs deep reasoning for summarization)
    thinking_budget_low = 1024
    thinking_budget_high = 8192

    # ============================================
    # Step 1: Edit raw transcript in chunks (parallel) - uses LOW thinking
    # ============================================
    if on_status:
        on_status("Editing raw transcript...")

    chunks = _split_into_chunks(transcript)
    logger.info(f"Split transcript into {len(chunks)} chunks for editing")

    if len(chunks) > 1:
        if on_status:
            on_status(f"Editing {len(chunks)} chunks in parallel...")

        # Process all chunks in parallel
        async def edit_chunk(chunk: str, index: int) -> str:
            return await with_retry(
                lambda: _edit_raw_chunk(
                    client,
                    chunk,
                    config.model,
                    config.temperature,
                    thinking_budget_low,  # Raw editing uses low thinking
                    index,
                    len(chunks),
                    metadata,
                    background,
                ),
                max_attempts=3,
                base_delay_ms=1000,
                context=f"Editing chunk {index + 1}",
            )

        edit_tasks = [edit_chunk(chunk, i) for i, chunk in enumerate(chunks)]
        edited_chunks = await asyncio.gather(*edit_tasks)
        logger.info(f"Edited {len(edited_chunks)} chunks")

        # Combine edited chunks
        edited_transcript = "\n\n".join(edited_chunks)
    else:
        # Single chunk, edit directly
        if on_status:
            on_status("Editing transcript...")

        edited_transcript = await with_retry(
            lambda: _edit_raw_chunk(
                client,
                chunks[0],
                config.model,
                config.temperature,
                thinking_budget_low,  # Raw editing uses low thinking
                0,
                1,
                metadata,
                background,
            ),
            max_attempts=3,
            base_delay_ms=1000,
            context="Editing transcript",
        )

    logger.info(f"Edited transcript: {len(transcript)} -> {len(edited_transcript)} chars")

    # ============================================
    # Step 2: Generate metadata JSON
    # ============================================
    if on_status:
        on_status("Generating summary and key points...")

    # Use section definitions from override or default
    sections = sections_override or DEFAULT_SECTIONS

    # Generate metadata (title, summary, key_points, etc.)
    metadata_dict = await with_retry(
        lambda: _generate_normal_metadata(
            client,
            edited_transcript,
            sections,
            config.model,
            config.temperature,
            thinking_budget_high,  # Metadata generation uses high thinking
            metadata,
            background,
        ),
        max_attempts=3,
        base_delay_ms=1000,
        context="Generating metadata",
    )

    logger.info(f"Generated metadata: title='{metadata_dict.get('title', 'N/A')}', "
                f"{len(metadata_dict.get('key_points', []))} key points")

    # ============================================
    # Step 2b (optional): Add inline translations
    # ============================================
    final_transcript = edited_transcript

    if enable_translation:
        if on_status:
            on_status("Adding inline translations...")
        logger.info("Translation mode enabled, adding inline translations")

        final_transcript = await with_retry(
            lambda: _translate_transcript(
                client,
                edited_transcript,
                config.translation_model,
                config.temperature,
                thinking_budget_low,  # Translation uses low thinking
            ),
            max_attempts=3,
            base_delay_ms=1000,
            context="Adding translations",
        )
        logger.info(f"Translated transcript: {len(edited_transcript)} -> {len(final_transcript)} chars")

    # ============================================
    # Step 3: Assemble final Markdown
    # ============================================
    final_output = _build_normal_result(metadata_dict, final_transcript)
    final_output = _normalize_quotes(final_output)

    logger.info(f"Editing completed, output length: {len(final_output)}")

    if on_status:
        on_status("Editing complete")

    return final_output


async def _edit_raw_chunk(
    client: genai.Client,
    chunk: str,
    model: str,
    temperature: float,
    thinking_budget: int,
    chunk_index: int,
    total_chunks: int,
    metadata: Optional[VideoEditorMetadata] = None,
    background: str = "",
) -> str:
    """Edit a raw transcript chunk (fix errors, remove fillers)."""
    logger.info(f"Processing chunk {chunk_index + 1}/{total_chunks}...")

    # Build user content with optional context (metadata or background)
    parts = []
    if metadata:
        parts.append("## Video Information (用于推断正确的说话人名字和专有名词)")
        parts.append(f"- Channel: {metadata.channel}")
        parts.append(f"- Title: {metadata.title}")
        if metadata.upload_date:
            parts.append(f"- Date: {metadata.upload_date}")
        if metadata.description:
            parts.append(f"- Description: {metadata.description}")
        parts.append("")
    elif background:
        parts.append("## Background Information (用于推断正确的说话人名字和专有名词)")
        parts.append(background)
        parts.append("")

    parts.append(f"## Raw Transcript Chunk ({chunk_index + 1}/{total_chunks})")
    parts.append("")
    parts.append(chunk)

    user_content = "\n".join(parts)

    # Use context-aware prompt if metadata or background provided
    system_prompt = RAW_EDIT_WITH_CONTEXT_SYSTEM_PROMPT if (metadata or background) else RAW_EDIT_SYSTEM_PROMPT

    response = await _gemini_generate(
        client, model, user_content,
        system=system_prompt, temperature=temperature,
        thinking_budget=thinking_budget, max_tokens=65536,
    )

    # Validate response
    text = response.text
    if not text or text.strip() == "":
        logger.warning(f"Empty response for chunk {chunk_index + 1}, using original")
        return chunk

    return text.strip()


async def _generate_final_output(
    client: genai.Client,
    user_content: str,
    system_prompt: str,
    model: str,
    temperature: float,
    thinking_budget: int,
) -> str:
    """Generate final formatted output with Title/Summary/Key Points/Transcript."""
    logger.info("Generating final formatted output...")

    response = await _gemini_generate(
        client, model, user_content,
        system=system_prompt, temperature=temperature,
        thinking_budget=thinking_budget,
    )

    # Validate response
    text = response.text
    if not text or text.strip() == "":
        error_msg = "Final output generation returned empty result."
        if hasattr(response, "prompt_feedback") and response.prompt_feedback:
            if hasattr(response.prompt_feedback, "block_reason"):
                error_msg += f" Block reason: {response.prompt_feedback.block_reason}"
        raise ValueError(error_msg)

    return text


# Legacy function for backwards compatibility
async def _edit_transcript(
    client: genai.Client,
    user_content: str,
    system_prompt: str,
    model: str,
    temperature: float,
    thinking_budget: int,
) -> str:
    """Edit transcript using Gemini model (legacy single-step)."""
    return await _generate_final_output(
        client, user_content, system_prompt, model, temperature, thinking_budget
    )


# ============================================================================
# Podcast Mode - Specialized processing for Apple Podcasts
# ============================================================================

@dataclass
class PodcastEpisodeMetadata:
    """Metadata about a podcast episode for editor context."""
    podcast_name: str
    episode_title: str
    episode_link: str = ""
    publish_date: str = ""
    shownotes: str = ""  # Episode description for context


@dataclass
class PodcastEditedTranscript:
    """Structured output for podcast mode editing."""
    title: str  # Title line: podcast - episode - date
    info: dict  # Info section: podcast, episode, link, publish_time
    summary: str  # Chinese summary
    takeaways: List[str]  # Key takeaways (Chinese)
    qa_pairs: List[dict]  # Q&A pairs: [{"question": ..., "answer": ...}]
    highlights: List[str]  # Highlight quotes
    full_transcript: str  # Full edited transcript
    markdown: str  # Complete markdown output


# Podcast raw transcript editing prompt (with Episode Context)
PODCAST_RAW_EDIT_SYSTEM_PROMPT = """You are a professional transcript editor. Your task is to clean up this raw transcript chunk.

## Episode Context
Use the episode information provided to help identify correct speaker names and proper nouns.

## Your Tasks

1. **修正转录错误**：
   - 修正明显的语音转文字错误（音近字、同音字）
   - 修正说话人名字错误（根据元数据中的播客名和描述推断正确名字）
   - 例如：如果播客名是"张小珺商业访谈录"，说话人应该是"张小珺"而非"张小军"、"张小钧"等
   - 修正专业术语、公司名称、人名的转录错误

2. **移除填充词**：
   - 移除中文填充词：嗯、啊、呃、额、哦、唔、那个、就是、然后、对对对、是是是
   - 移除英文填充词：um, uh, like, you know, I mean, sort of, kind of, basically
   - 保留有实际语义的语气词和感叹词

3. **格式规范**：
   - 按说话人分段，格式为 **说话人名:** 后跟内容
   - 移除时间戳（如 **00:00**、**01:23** 等）
   - 适当分段提高可读性，每个说话人的发言可以分为多段

4. **保持完整**：
   - 输出完整内容，绝对不要截断或省略任何部分
   - 保持原语言，不要翻译
   - 保留技术术语、专有名词、公司名称

## Output Format

直接输出编辑后的转录文本，不要添加任何标题、说明或JSON。
只输出编辑后的内容，从第一个说话人开始。

## CRITICAL RULES
- 必须输出完整内容，不能省略任何部分
- 不要添加任何前言、解释或总结
- 不要输出 JSON
- 直接输出编辑后的转录文本
- 绝对不要输出类似 "It appears that" 或 "Please provide" 的错误提示
- 即使输入看起来不完整，也要直接编辑并输出可用内容
"""

# Podcast metadata generation prompt (outputs JSON)
PODCAST_METADATA_SYSTEM_PROMPT = """You are a professional editor specializing in podcast transcripts. Your task is to analyze the transcript and generate structured metadata.

## Output Format

Output ONLY a JSON block (wrapped in ```json ... ```):

```json
{
  "summary": "中文摘要，200-400字，概括主题、嘉宾背景、核心讨论内容和关键洞见",
  "takeaways": [
    "要点1：完整表达一个核心观点或洞见",
    "要点2：...",
    "..."
  ],
  "qa_pairs": [
    {
      "q": "问题1（保持原语言）",
      "a": "回答1，可以是多段落（保持原语言）"
    }
  ],
  "highlights": [
    "精彩语句1（保持原语言）",
    "精彩语句2",
    "..."
  ]
}
```

## Guidelines

1. **summary**: 必须是中文（简体），200-400字，详细概括
2. **takeaways**: 必须是中文（简体），10-15条核心观点
   - 每条是完整的观点陈述
   - 英文内容翻译成中文，保留术语原文如：深度学习（Deep Learning）
3. **qa_pairs**: 8-12个最有价值的问答
   - 保持原语言（中文用中文，英文用英文）
   - 回答应完整，可包含多段落
4. **highlights**: 10-20条精彩语句或金句
   - 保持原语言
   - 选择有洞察力、启发性的句子

## CRITICAL RULES
- 只输出 JSON 块，不要输出其他内容
- JSON 必须是有效的 JSON 格式
- summary 和 takeaways 必须是中文
- qa_pairs 和 highlights 保持原语言
"""


async def edit_podcast(
    transcript: str,
    config: EditorConfig,
    metadata: Optional[PodcastEpisodeMetadata] = None,
    on_status: Callable[[str], None] | None = None,
) -> PodcastEditedTranscript:
    """
    Edit and format a podcast transcript using two-step processing.

    This is the podcast-specific version that:
    1. Uses Episode Context for better speaker name detection
    2. Outputs structured metadata (summary, takeaways, Q&A, highlights)
    3. Returns a PodcastEditedTranscript with rich structured data

    Args:
        transcript: The raw transcript text to edit
        config: Editor configuration
        metadata: Episode metadata (podcast name, title, link, date, shownotes)
        on_status: Optional callback to report status updates

    Returns:
        PodcastEditedTranscript with structured content
    """
    if not config.api_key:
        raise ValueError("Editor API key is not configured")

    if not transcript or not transcript.strip():
        raise ValueError("Empty transcript provided")

    logger.info("Starting podcast transcript editing (two-step processing)...")

    # Get shared client (reuses HTTP connection pool)
    client = _get_client(config.api_key)

    # Configure thinking budgets for different steps
    # Raw editing uses low thinking (simple cleanup tasks)
    # Metadata generation uses high thinking (needs deep reasoning for summarization)
    thinking_budget_low = 1024
    thinking_budget_high = 8192

    # ============================================
    # Step 1: Edit raw transcript in chunks (parallel) - uses LOW thinking
    # ============================================
    if on_status:
        on_status("Editing raw transcript...")

    chunks = _split_into_chunks(transcript)
    logger.info(f"Split transcript into {len(chunks)} chunks for editing")

    if on_status:
        if len(chunks) > 1:
            on_status(f"Editing {len(chunks)} chunks in parallel...")
        else:
            on_status("Editing transcript...")

    # Process all chunks in parallel
    async def edit_chunk(chunk: str, index: int) -> str:
        return await with_retry(
            lambda: _edit_podcast_raw_chunk(
                client,
                chunk,
                metadata,
                config.model,
                config.temperature,
                thinking_budget_low,  # Raw editing uses low thinking
                index,
                len(chunks),
            ),
            max_attempts=3,
            base_delay_ms=1000,
            context=f"Editing chunk {index + 1}",
        )

    edit_tasks = [edit_chunk(chunk, i) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*edit_tasks, return_exceptions=True)

    # Handle results, using original chunk if editing failed
    edited_chunks = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Failed to edit chunk {i + 1}/{len(chunks)}: {result}")
            # Use original chunk as fallback
            edited_chunks.append(chunks[i])
        elif result:  # Skip empty strings (from skipped chunks)
            edited_chunks.append(result)

    logger.info(f"Edited {len(edited_chunks)} chunks")

    # Combine edited chunks (filter out any remaining empty strings)
    edited_transcript = "\n\n".join(c for c in edited_chunks if c.strip())
    logger.info(f"Edited transcript: {len(transcript)} -> {len(edited_transcript)} chars")

    # ============================================
    # Step 2: Generate metadata from EDITED transcript
    # ============================================
    if on_status:
        on_status("Generating metadata (summary, takeaways, Q&A, highlights)...")

    logger.info("Generating metadata from edited transcript...")
    metadata_dict = await with_retry(
        lambda: _generate_podcast_metadata(
            client,
            edited_transcript,
            metadata,
            config.model,
            config.temperature,
            thinking_budget_high,  # Metadata generation uses high thinking
        ),
        max_attempts=3,
        base_delay_ms=1000,
        context="Generating metadata",
    )

    logger.info(f"Generated metadata: {len(metadata_dict.get('takeaways', []))} takeaways, "
               f"{len(metadata_dict.get('qa_pairs', []))} Q&A, "
               f"{len(metadata_dict.get('highlights', []))} highlights")

    # ============================================
    # Step 3: Build final result
    # ============================================
    result = _build_podcast_result(metadata_dict, edited_transcript, metadata)

    if on_status:
        on_status("Editing complete")

    logger.info(f"Podcast editing completed: '{result.title}' "
               f"({len(result.takeaways)} takeaways, {len(result.qa_pairs)} Q&A, "
               f"{len(result.highlights)} highlights, {len(result.full_transcript)} chars)")

    return result


async def _edit_podcast_raw_chunk(
    client: genai.Client,
    chunk: str,
    metadata: Optional[PodcastEpisodeMetadata],
    model: str,
    temperature: float,
    thinking_budget: int,
    chunk_index: int,
    total_chunks: int,
) -> str:
    """Edit a raw transcript chunk with podcast context (fix errors, remove fillers, format speakers)."""
    # Pre-validation: skip chunks without meaningful content
    if not _has_meaningful_content(chunk):
        logger.warning(
            f"Chunk {chunk_index + 1}/{total_chunks}: Skipping - "
            f"no meaningful content (only timestamps/sound effects)"
        )
        return ""  # Return empty string, will be filtered out when joining

    logger.info(f"Processing podcast chunk {chunk_index + 1}/{total_chunks}...")

    # Build user content with full metadata context
    parts = []
    if metadata:
        parts.append("## Episode Information (用于推断正确的说话人名字和专有名词)")
        parts.append(f"- Podcast Name: {metadata.podcast_name}")
        parts.append(f"- Episode Title: {metadata.episode_title}")
        if metadata.publish_date:
            parts.append(f"- Publish Date: {metadata.publish_date}")
        if metadata.shownotes:
            # Truncate shownotes for context
            shownotes_truncated = metadata.shownotes[:2000]
            if len(metadata.shownotes) > 2000:
                shownotes_truncated += "..."
            parts.append(f"- Episode Description: {shownotes_truncated}")
        parts.append("")

    parts.append(f"## Raw Transcript Chunk ({chunk_index + 1}/{total_chunks})")
    parts.append("")
    parts.append(chunk)

    user_content = "\n".join(parts)

    response = await _gemini_generate(
        client, model, user_content,
        system=PODCAST_RAW_EDIT_SYSTEM_PROMPT, temperature=temperature,
        thinking_budget=thinking_budget, max_tokens=65536,
    )

    # Validate response
    text = response.text
    if not text or text.strip() == "":
        logger.warning(f"Empty response for chunk {chunk_index + 1}, using original")
        return chunk

    return text.strip()


async def _generate_podcast_metadata(
    client: genai.Client,
    transcript: str,
    metadata: Optional[PodcastEpisodeMetadata],
    model: str,
    temperature: float,
    thinking_budget: int,
) -> dict:
    """Generate metadata (summary, takeaways, Q&A, highlights) from edited transcript."""
    logger.info("Generating podcast metadata...")

    # Build user content with metadata context
    parts = []
    if metadata:
        parts.append("## Episode Information")
        parts.append(f"- Podcast Name: {metadata.podcast_name}")
        parts.append(f"- Episode Title: {metadata.episode_title}")
        if metadata.publish_date:
            parts.append(f"- Publish Date: {metadata.publish_date}")
        if metadata.shownotes:
            shownotes_truncated = metadata.shownotes[:2000]
            if len(metadata.shownotes) > 2000:
                shownotes_truncated += "..."
            parts.append(f"- Episode Description: {shownotes_truncated}")
        parts.append("")

    parts.append("## Transcript")
    parts.append("")

    # Protective truncation for very long transcripts
    max_transcript_chars = 500000
    if len(transcript) > max_transcript_chars:
        logger.warning(f"Transcript too long ({len(transcript)} chars), truncating to {max_transcript_chars} chars")
        transcript_truncated = transcript[:max_transcript_chars] + "\n\n[... transcript truncated for metadata generation ...]"
        parts.append(transcript_truncated)
    else:
        parts.append(transcript)

    user_content = "\n".join(parts)

    # Gemini 3 counts thinking tokens against max_output_tokens, so the cap
    # must hold the thinking budget (8192 on this path) PLUS the JSON itself;
    # the podcast JSON (Q&A, highlights) runs long.
    response = await _gemini_generate(
        client, model, user_content,
        system=PODCAST_METADATA_SYSTEM_PROMPT, temperature=temperature,
        thinking_budget=thinking_budget, max_tokens=24576,
    )

    if not response.text:
        raise ValueError("Empty metadata response")
    if is_truncated(response):
        raise ValueError("Metadata generation hit max output tokens; JSON is incomplete")

    # Parse JSON from response
    response_text = response.text

    # Pattern 1: ```json ... ```
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)

    # Pattern 2: ``` ... ```
    if not json_match:
        json_match = re.search(r'```\s*([\{\[].*?[\}\]])\s*```', response_text, re.DOTALL)

    # Pattern 3: Raw JSON object
    if not json_match:
        json_match = re.search(r'(\{[\s\S]*\})', response_text)

    if not json_match:
        raise ValueError("No JSON block found in metadata output")

    json_str = json_match.group(1).strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse metadata JSON: {e}")
        raise ValueError(f"Invalid JSON in metadata output: {e}")


def _build_podcast_result(
    metadata_dict: dict,
    edited_transcript: str,
    episode_metadata: Optional[PodcastEpisodeMetadata] = None,
) -> PodcastEditedTranscript:
    """Build PodcastEditedTranscript from metadata dict and edited transcript."""
    # Build title and info from episode metadata
    if episode_metadata:
        title = f"{episode_metadata.podcast_name} - {episode_metadata.episode_title}"
        if episode_metadata.publish_date:
            title += f" - {episode_metadata.publish_date}"
        info = {
            "podcast": episode_metadata.podcast_name,
            "episode": episode_metadata.episode_title,
            "link": episode_metadata.episode_link,
            "publish_time": episode_metadata.publish_date,
        }
    else:
        title = "Podcast Transcript"
        info = {}

    summary = metadata_dict.get("summary", "")
    takeaways = metadata_dict.get("takeaways", [])
    highlights = metadata_dict.get("highlights", [])

    # Normalize qa_pairs format
    qa_pairs = []
    for qa in metadata_dict.get("qa_pairs", []):
        if isinstance(qa, dict):
            qa_pairs.append({
                "question": qa.get("q", qa.get("question", "")),
                "answer": qa.get("a", qa.get("answer", ""))
            })

    # Generate final markdown
    markdown = _render_podcast_markdown(
        title=title,
        info=info,
        summary=summary,
        takeaways=takeaways,
        qa_pairs=qa_pairs,
        highlights=highlights,
        transcript=edited_transcript,
    )

    return PodcastEditedTranscript(
        title=title,
        info=info,
        summary=summary,
        takeaways=takeaways,
        qa_pairs=qa_pairs,
        highlights=highlights,
        full_transcript=edited_transcript,
        markdown=_normalize_quotes(markdown),
    )


def _get_link_text(url: str) -> str:
    """Get appropriate link text based on URL domain.

    Args:
        url: The episode URL

    Returns:
        Appropriate link text for the platform
    """
    if not url:
        return "Listen"
    url_lower = url.lower()
    if "xiaoyuzhoufm.com" in url_lower:
        return "Listen on Xiaoyuzhoufm"
    elif "podcasts.apple.com" in url_lower:
        return "Listen on Apple Podcasts"
    elif "spotify.com" in url_lower:
        return "Listen on Spotify"
    elif "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "Watch on YouTube"
    else:
        return "Listen"


def _render_podcast_markdown(
    title: str,
    info: dict,
    summary: str,
    takeaways: list,
    qa_pairs: list,
    highlights: list,
    transcript: str,
) -> str:
    """Render structured podcast data to final markdown."""
    lines = [f"# {title}", ""]

    # Info section
    if info:
        lines.append("## Info")
        lines.append("")
        if info.get("podcast"):
            lines.append(f"- **Podcast**: {info['podcast']}")
        if info.get("episode"):
            lines.append(f"- **Episode**: {info['episode']}")
        if info.get("link"):
            link_text = _get_link_text(info["link"])
            lines.append(f"- **Link**: [{link_text}]({info['link']})")
        if info.get("publish_time"):
            lines.append(f"- **Publish Time**: {info['publish_time']}")
        lines.append("")

    # Summary
    if summary:
        lines.append("## Summary")
        lines.append("")
        lines.append(summary)
        lines.append("")

    # Takeaways
    if takeaways:
        lines.append("## Takeaways")
        lines.append("")
        for point in takeaways:
            lines.append(f"- {point}")
        lines.append("")

    # Q&A
    if qa_pairs:
        lines.append("## Q & A")
        lines.append("")
        for i, qa in enumerate(qa_pairs):
            if i > 0:
                lines.append("---")
                lines.append("")
            lines.append(f"**Q: {qa['question']}**")
            lines.append("")
            lines.append(f"A: {qa['answer']}")
            lines.append("")

    # Highlights
    if highlights:
        lines.append("## Highlights")
        lines.append("")
        for h in highlights:
            lines.append(f"- {h}")
        lines.append("")

    # Transcript
    if transcript:
        lines.append("## Transcript")
        lines.append("")
        lines.append(transcript)

    return "\n".join(lines)
