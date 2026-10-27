"""Edit fidelity guard: put back what the LLM editor dropped or reworded.

Backported from the transcribe skill (tav607/my-claude-skills b2ae63f). The
tokenizer and number folding come from the skill's consensus reconcile, which
is why they keep its _cons_ names: the two copies stay easy to diff.
"""

import difflib
import re

from ..utils.chinese import _zh_fold
# The transcription prompt's bold **Name:** label contract lives with these.
from .transcriber import _NAMED_LABEL_RE, _SPEAKER_TOKEN_RE

# Bold **00:00** timestamps (the podcast edit prompt removes them) and
# [HH:MM:SS] variants.
_TIMESTAMP_RE = re.compile(r'\*\*\d{1,2}:\d{2}(?::\d{2})?\*\*|\[\d{1,2}:\d{2}:\d{2}\]')
# Uppercase bracketed sound effects like [MUSIC PLAYING]. Lowercase [inaudible]
# / [unclear] are real transcript content and deliberately not masked here.
_SOUND_EFFECT_RE = re.compile(r'\[[A-Z][A-Z\s]+\]')

# Single CJK chars first (Chinese has no word spacing, so alignment is per
# character, the CER convention), then unicode word runs excluding CJK (so
# accented names like Condé stay whole), keeping internal apostrophes and
# decimal points.
_CONS_TOKEN_RE = re.compile(r"[㐀-鿿]|[^\W_㐀-鿿]+(?:['’.][^\W_㐀-鿿]+)*")

# Hesitation markers never enter the token stream: the edit prompts strip
# them, so dropping one is never a loss.
_CONS_FILLERS = {"uh", "um", "mhm", "mm", "hmm", "ah", "er", "erm",
                 "呃", "嗯", "唉", "诶", "欸", "呐"}
_CONS_WORD_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
}


def _cons_tokenize(text: str) -> tuple[list[tuple[str, int, int]], list[tuple[int, int]]]:
    """Tokenize for alignment. Returns (tokens, label_spans).

    Tokens are (normalized, start, end) char spans, lowercased and folded to
    simplified script so a script flip is not a change; line-start speaker
    labels are excluded from the token stream and returned separately.
    """
    label_spans = [(m.start(), m.end()) for m in _NAMED_LABEL_RE.finditer(text)]
    tokens = []
    for m in _CONS_TOKEN_RE.finditer(text):
        if any(ls <= m.start() < le for ls, le in label_spans):
            continue
        norm = m.group().lower().replace("’", "'").rstrip(".")
        if norm and _zh_fold is not None and "㐀" <= norm <= "鿿":
            norm = _zh_fold(norm, "zh-cn")
        if norm and norm not in _CONS_FILLERS:
            tokens.append((norm, m.start(), m.end()))
    return tokens, label_spans


# Chinese numerals are the CJK half of the spoken-vs-digit formatting split
# _CONS_WORD_NUM covers for English: an editor may write "468亿" / "62%" where
# the transcript has "四百六十八亿" / "百分之六十二", which is formatting, not a
# change of content.
_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
# 万 and 亿 are deliberately outside the run: "468亿" keeps the scale character
# too, so folding only the numeral head lines the two up without having to
# agree on whether the result is spelled out in full.
_CN_NUM_RUN = re.compile(r"[零〇一二两三四五六七八九十百千]+(?:点[零〇一二两三四五六七八九]+)?")


def _cn_run_to_digits(run: str) -> str:
    """One Chinese numeral run (no 万/亿, no decimal point) as digits."""
    if not any(ch in _CN_UNITS for ch in run):
        # No unit character: read as a digit string ("二零二六" for a year)
        # rather than as a positional number, which would collapse it to 6.
        return "".join(str(_CN_DIGITS[ch]) for ch in run)
    total = tmp = 0
    for ch in run:
        if ch in _CN_UNITS:
            total += (tmp or 1) * _CN_UNITS[ch]
            tmp = 0
        else:
            tmp = _CN_DIGITS[ch]
    return str(total + tmp)


def _cn_numerals_to_digits(text: str) -> str:
    def convert(m: re.Match) -> str:
        head, _, frac = m.group().partition("点")
        digits = _cn_run_to_digits(head)
        if frac:
            digits += "." + "".join(str(_CN_DIGITS[ch]) for ch in frac)
        return digits
    return _CN_NUM_RUN.sub(convert, text)


def _cons_formatting_variants(a_norm: tuple, b_norm: tuple) -> bool:
    """True when two readings differ only in formatting, not in content:
    spacing ("light shed" vs "lightshed"), spoken vs digit numbers ("five"
    vs "5"), or the Chinese equivalent ("四百六十八" vs "468").

    The percent marker is dropped rather than folded because "%" is not a word
    character and never survives _CONS_TOKEN_RE: "62%" already arrives here as
    ("62",), so "百分之六十二" can only be matched by discarding 百分之 on the
    other side.
    """
    def canon(span):
        joined = "".join(_CONS_WORD_NUM.get(t, t) for t in span)
        return _cn_numerals_to_digits(joined.replace("百分之", ""))
    return canon(a_norm) == canon(b_norm)


# Edit fidelity guard. The editor is told to keep everything, yet in 12 of 23
# October 2026 expert calls it compressed passages, dropped sentences or
# reworded claims ("the cost difference only because of the database is 45%,
# it's higher" came back as "Oracle was 45% higher"). Re-edited, a 59-minute
# Chinese call lost 14% of its distinct content words on gemini-3.8-flash. So
# each edited chunk is aligned against its raw input on consensus tokens, and
# only cleanup survives: fillers, stutters, false starts restated within the
# next few words, small function-word trims, formatting variants. Every other
# deletion, insertion or substitution is put back as the raw text. Numbers and
# negations never count as cleanup.
def _edit_phrases(words: list[str]) -> list[tuple[str, ...]]:
    phrases = [tuple(w) if "一" <= w[0] <= "鿿" else tuple(w.split()) for w in words]
    return sorted(phrases, key=len, reverse=True)


_EDIT_FILLERS = _edit_phrases([
    "那个", "就是说", "就是", "然后", "这个", "对对对", "对对", "是是是", "是是", "的话", "怎么说呢",
    "怎么说", "那么", "其实", "反正", "啊", "哦", "噢", "吧", "呢", "嘛", "啦", "呀", "哈", "哎",
    "额", "唔", "对", "是", "好", "那", "就",
    "you know", "i mean", "sort of", "kind of", "uhm", "oh", "like", "so", "yeah", "yes", "right",
    "okay", "ok", "well", "actually", "basically", "really", "just", "and",
])
_EDIT_FUNCTION_WORDS = _edit_phrases([
    "这种", "那种", "这样", "那样", "这些", "那些", "这边", "那边", "所以", "的", "了", "也", "都",
    "还", "这", "在", "有", "去", "来", "说", "我们", "我", "你", "他", "它",
    "the", "a", "an", "to", "of", "that", "it", "is", "was", "are", "were", "we", "i", "you",
    "they", "this", "there", "in", "on", "at", "for", "with", "as", "be", "have", "has", "do",
])
# English words that carry no content of their own when deciding whether a
# dropped run was restated; "only", "because" and the like are kept out on
# purpose, so a run of them alone still counts as content.
_EDIT_STOPWORDS = frozenset(
    "a an and are as at be been being but by can could did do does for from had has have he her "
    "here him his how i if in into is it its me my of on or our she so some than that the their "
    "them then there these they this those to was we were what when where which who will with "
    "would you your yeah okay ok uh um like well really actually basically know mean sort kind "
    "right oh going get got thing things lot also".split()
)
_EDIT_NUMERALS = frozenset("零〇一二两三四五六七八九十百千万亿")
_EDIT_NEGATIONS = frozenset("不没别无非未莫勿") | {"not", "no", "never", "nor", "none", "nothing", "without"}


def _strip_phrases(run: list[str], table: list[tuple[str, ...]]) -> list[str]:
    out, i = [], 0
    while i < len(run):
        for p in table:
            if tuple(run[i:i + len(p)]) == p:
                i += len(p)
                break
        else:
            out.append(run[i])
            i += 1
    return out


def _is_cjk_token(t: str) -> bool:
    return len(t) == 1 and "㐀" <= t <= "鿿"


def _edit_risky(tokens: list[str]) -> bool:
    return any(t in _EDIT_NEGATIONS or t.endswith("n't") or t in _EDIT_NUMERALS
               or any(ch.isdigit() for ch in t) for t in tokens)


def _edit_numeric(token: str) -> bool:
    return any(ch.isdigit() for ch in token) or (bool(token) and all(ch in _EDIT_NUMERALS for ch in token))


def _edit_separator(gap: str) -> str:
    """The punctuation in the text between two tokens, units and spaces aside."""
    return re.sub(r"[\s%‰°]", "", gap)


_EDIT_BRACKETS = {"(": ")", "（": "）", "[": "]", "【": "】", "「": "」", "『": "』", "《": "》", "“": "”"}


_EDIT_GAP_FILLERS = re.compile(
    "[" + "".join(f for f in _CONS_FILLERS if len(f) == 1) + "]|\\b(?:"
    + "|".join(sorted((f for f in _CONS_FILLERS if len(f) > 1), key=len, reverse=True)) + ")\\b",
    re.IGNORECASE)


def _edit_gap(gap: str) -> str:
    """A raw separator without the hesitation words the tokenizer leaves in it."""
    out = _EDIT_GAP_FILLERS.sub("", gap)
    out = re.sub(r"([，,、；;])(?:\s*[，,、；;])+", r"\1", out)
    # "，嗯。" leaves "，。": the stronger mark wins.
    out = re.sub(r"[，,、]\s*(?=[。.！!？?；;])|(?<=[。.！!？?；;])\s*[，,、]", "", out)
    return re.sub(r" {2,}", " ", out)


def _edit_closers(text: str, gap: str) -> str:
    """The leading closing brackets of `gap` that brackets opened in `text` need."""
    out = ""
    for ch in gap:
        if ch == " ":
            out += ch
            continue
        opener = next((o for o, c in _EDIT_BRACKETS.items() if c == ch), None)
        if opener is None or (text + out).count(opener) <= (text + out).count(ch):
            break
        out += ch
    return out.rstrip(" ") if out.strip() else ""


def _edit_stutter(norms: list[str], i1: int, i2: int) -> bool:
    """The run repeats the words right before or after it ("我们我们要", "the the")."""
    run = _strip_phrases(norms[i1:i2], _EDIT_FILLERS) or norms[i1:i2]
    for k in range(1, min(len(run), 8) + 1):
        for unit in (norms[i2:i2 + k], norms[max(0, i1 - k):i1]):
            if len(unit) == k and len(run) % k == 0 and run == unit * (len(run) // k):
                return True
    return False


def _edit_units(tokens: list[str]) -> set[str]:
    left = _strip_phrases(_strip_phrases(tokens, _EDIT_FILLERS), _EDIT_FUNCTION_WORDS)
    cjk = "".join(t for t in left if _is_cjk_token(t))
    units = {cjk[i:i + 2] for i in range(len(cjk) - 1)} if len(cjk) >= 2 else set(cjk)
    return units | {t for t in left if not _is_cjk_token(t) and t not in _EDIT_STOPWORDS}


def _edit_restated(norms: list[str], i1: int, i2: int) -> bool:
    """A dropped false start: three quarters of its content units reappear within
    a short window of the raw text ("价格战的这个，就是大家打价格战")."""
    units = _edit_units(norms[i1:i2])
    if not units:
        return False
    w = max(15, 3 * (i2 - i1))
    near = _edit_units(norms[max(0, i1 - w):i1]) | _edit_units(norms[i2:i2 + w])
    return len(units & near) / len(units) >= 0.75


def _edit_free_deletion(norms: list[str], i1: int, i2: int) -> bool:
    left = _strip_phrases(norms[i1:i2], _EDIT_FILLERS)
    if not left or _edit_stutter(norms, i1, i2):
        return True
    if _edit_risky(left):
        return False
    left = _strip_phrases(left, _EDIT_FUNCTION_WORDS)
    if not left or (len(left) <= 2 and all(_is_cjk_token(t) for t in left)):
        return True
    return _edit_restated(norms, i1, i2)


def _edit_free_insertion(tokens: list[str]) -> bool:
    left = _strip_phrases(tokens, _EDIT_FILLERS)
    return not left or (not _edit_risky(left) and not _strip_phrases(left, _EDIT_FUNCTION_WORDS))


def _edit_free_replacement(a: list[str], b: list[str]) -> bool:
    if _cons_formatting_variants(tuple(a), tuple(b)):
        return True
    ka, kb = _strip_phrases(a, _EDIT_FILLERS), _strip_phrases(b, _EDIT_FILLERS)
    if ka == kb:
        return True
    if _edit_risky(ka) or _edit_risky(kb):
        return False
    ka, kb = _strip_phrases(ka, _EDIT_FUNCTION_WORDS), _strip_phrases(kb, _EDIT_FUNCTION_WORDS)
    if ka == kb:
        return True
    # A one- or two-character homophone fix, or a respelled word ("Ziliz" -> "Zilliz").
    if all(_is_cjk_token(t) for t in ka + kb) and len(ka) <= 2 and len(kb) <= 2:
        return True
    return (len(ka) == len(kb) == 1 and not _is_cjk_token(ka[0])
            and difflib.SequenceMatcher(None, ka[0], kb[0]).ratio() >= 0.75)


def _restore_edit_losses(raw: str, edited: str) -> tuple[str, int]:
    """Put back what the editor dropped or reworded beyond cleanup.

    Returns (text, number of restored spans). Timestamps and sound-effect tags
    are masked out of the comparison because the edit prompts remove them on
    purpose, and a substitution that would move a speaker label is left alone.
    """
    def mask(text: str) -> str:
        # Labels too: once a timestamp is blanked, the label after it no longer
        # starts a line, and the tokenizer would read its name as content.
        blank = lambda m: " " * len(m.group())
        for pattern in (_TIMESTAMP_RE, _SOUND_EFFECT_RE, _SPEAKER_TOKEN_RE):
            text = pattern.sub(blank, text)
        return text

    def through_symbols(text: str, end: int) -> int:
        # "%" and friends are not token characters but belong to the number.
        while end < len(text) and text[end] in "%‰°":
            end += 1
        return end

    def plain(gap: str) -> bool:
        # Punctuation and spaces only: no line break, label, timestamp or tag.
        return "\n" not in gap and mask(gap) == gap

    rt, _ = _cons_tokenize(mask(raw))
    et, _ = _cons_tokenize(mask(edited))
    if not rt or not et:
        return edited, 0
    rn, en = [t[0] for t in rt], [t[0] for t in et]
    # Any label, not only one opening a line: an editor can leave a turn
    # mid-line, and a patch must never cut through one.
    labels = [m.start() for m in _SPEAKER_TOKEN_RE.finditer(edited)]
    label_ends = [m.end() for m in _SPEAKER_TOKEN_RE.finditer(edited)]
    patches: list[tuple[int, int, str]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, rn, en, autojunk=False).get_opcodes():
        if tag == "equal":
            # Punctuation is not a token, so two numbers the editor fused by
            # dropping what stood between them read as equal: a restart
            # "六……600多" came back as "六600多", which reads as 6600.
            for i, j in zip(range(i1, i2 - 1), range(j1, j2 - 1)):
                if not (_edit_numeric(rn[i]) and _edit_numeric(rn[i + 1])):
                    continue
                gap = raw[rt[i][2]:rt[i + 1][1]]
                if (_edit_separator(gap) and re.fullmatch(r"[^\w\n]{1,8}", gap)
                        and not _edit_separator(edited[et[j][2]:et[j + 1][1]])):
                    patches.append((et[j][2], et[j + 1][1], gap))
        elif tag == "delete" and not _edit_free_deletion(rn, i1, i2):
            # Put the run back after the previous word with the raw separator
            # before it, plus the closing brackets its own brackets need (taking
            # the separator alone left "（大概是三十亿美元" for "（大概是三十亿美
            # 元）"). A separator that holds a turn boundary is not copied: the
            # run joins the next word instead, or comes back as a whole turn.
            run_end = through_symbols(raw, rt[i2 - 1][2])
            run = raw[rt[i1][1]:run_end]
            gap_a = _edit_gap(raw[rt[i1 - 1][2]:rt[i1][1]]) if i1 else ""
            gap_b = _edit_gap(raw[run_end:rt[i2][1]] if i2 < len(rt) else raw[run_end:])
            ea = et[j1 - 1][2] if j1 else None
            eb = et[j1][1] if j1 < len(et) else None
            if ea is not None and plain(gap_a):
                # After the previous word with the raw separator, less a closing
                # bracket the edit kept right there.
                kept = len(gap_a) - len(gap_a.lstrip("".join(_EDIT_BRACKETS.values())))
                kept = kept if kept and edited.startswith(gap_a[:kept], ea) else 0
                tail = _edit_closers(gap_a + run, gap_b)
                # A restored restart "六" must not fuse with the "600" after it.
                if (i2 < len(rt) and _edit_numeric(rn[i2 - 1]) and _edit_numeric(rn[i2])
                        and plain(gap_b) and eb is not None and not _edit_separator(edited[ea:eb])):
                    tail = gap_b
                patches.append((ea + kept, ea + kept, gap_a[kept:] + run + tail))
            elif eb is not None and plain(gap_b):
                # The run opened a turn: keep it inside that turn.
                patches.append((eb, eb, run + gap_b))
            else:
                at = ea if ea is not None else et[0][1]
                text = (gap_a if ea is not None else "") + run + _edit_closers(gap_a + run, gap_b)
                # A whole lost turn comes back with its label, but not with the
                # timestamps and tags the edit removes everywhere.
                text = _SOUND_EFFECT_RE.sub("", _TIMESTAMP_RE.sub("", text))
                patches.append((at, at, re.sub(r"(?<=\S) {2,}(?=\S)", " ", text)))
        elif tag == "insert" and not _edit_free_insertion(en[j1:j2]):
            start, end = (et[j1 - 1][2] if j1 else et[j1][1]), through_symbols(edited, et[j2 - 1][2])
            inside = [le for ls, le in zip(labels, label_ends) if start <= ls < end]
            if inside:  # keep the label of the turn the inserted words open
                start = max(inside)
                while start < end and edited[start] == " ":
                    start += 1
                # and the single space after it
                while end < len(edited) and edited[end] == " " and edited[start - 1] == " ":
                    end += 1
            patches.append((start, end, ""))
        elif tag == "replace" and not _edit_free_replacement(rn[i1:i2], en[j1:j2]):
            start, end = et[j1][1], through_symbols(edited, et[j2 - 1][2])
            r_start, r_end = rt[i1][1], through_symbols(raw, rt[i2 - 1][2])
            if (any(start <= ls < end for ls in labels)
                    or _SPEAKER_TOKEN_RE.search(raw[r_start:r_end])):
                continue
            patches.append((start, end, raw[r_start:r_end]))
    for start, end, text in sorted(patches, reverse=True):
        edited = edited[:start] + text + edited[end:]
    return edited, len(patches)
