"""Simplified/traditional Chinese script folding (backported from the transcribe skill)."""

import re

# Gemini can answer Taiwanese-accented audio in traditional script wholesale,
# even though every prompt asks for simplified. Optional import: folding
# degrades to identity if the dependency is missing, so a deploy that skipped
# `uv sync` still runs.
try:
    from zhconv import convert as _zh_fold
except ImportError:
    _zh_fold = None
else:
    # zhconv builds its conversion table on first use, without a lock, and the
    # edit guard folds tokens from worker threads; build it once here (about
    # 50ms) so concurrent callers only ever read it.
    _zh_fold("", "zh-cn")

_KANA_RE = re.compile(r"[぀-ヿ]")


def _to_simplified(text: str) -> str:
    """Fold Chinese text to simplified script.

    The prompt asks for simplified Chinese, but a pass on Taiwanese-accented
    audio can come back traditional wholesale, so a single-pass transcript used
    to mix scripts chunk by chunk (2 of 63 Chinese transcripts in the vault,
    one 85% traditional; two October expert calls converted by hand). Text with
    Japanese kana is left alone: folding would rewrite its kanji.
    """
    if _zh_fold is None or not text:
        return text
    cjk = sum("㐀" <= ch <= "鿿" for ch in text)
    if cjk == 0 or len(_KANA_RE.findall(text)) > 0.05 * cjk:
        return text
    return _zh_fold(text, "zh-cn")
