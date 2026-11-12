"""Output sections of an edited transcript, as data.

A section is one field the metadata call fills in and one H2 of the final
Markdown. The same list drives the prompt text, the response schema and the
rendering, so the three cannot drift apart. It replaces free-text section
definitions that were parsed back into a schema with a regex: a definition the
regex could not read (an object array, a '*' bullet) dropped the whole call
to unconstrained JSON extraction, and headings came from title-casing the
field name.
"""

from dataclasses import dataclass
from typing import Sequence

SECTION_TYPES = ("text", "list", "qa")
MODES = ("generic", "meeting", "earnings")

# Control field of earnings mode: where the transcript's Q&A session starts.
# Never rendered; _build_normal_result splits the Transcript section there.
QA_START_QUOTE = "_qa_start_quote"
_QA_START_QUOTE_PROMPT = (
    "Q&A环节第一段的逐字开头 (10-20个英文词，含说话人标签，通常是接线员或IR主持人邀请提问的那段；"
    "不是高管prepared remarks结尾的交接句。从transcript原文逐字复制)"
)

_DEFAULT_HEADINGS = {
    "summary": "Summary",
    "key_points": "Key Points",
    "decisions": "Decisions",
    "action_items": "Action Items",
    "management_remarks": "Management Remarks",
    "qa": "Q&A Highlights",
}


@dataclass(frozen=True)
class Section:
    """One output field. type: "text" (a string), "list" (strings rendered as
    bullets) or "qa" (question/answer pairs)."""

    key: str
    prompt: str
    type: str = "list"
    heading: str = ""

    @property
    def title(self) -> str:
        return self.heading or _DEFAULT_HEADINGS.get(self.key) or self.key.replace("_", " ").title()


_TITLE = Section("title", "简短的中文标题，概括内容主题 (5-15个中文字符，仅中文、数字和基本标点)", "text")

GENERIC_SECTIONS = (
    _TITLE,
    Section("summary", "中文摘要，概括核心内容和关键结论 (不超过300字)", "text"),
    Section("key_points", "中文要点列表，关注可执行事项、决策和重要信息 (最多20条)"),
)

MEETING_SECTIONS = (
    Section("title", "会议标题 (中文，5-20字符)", "text"),
    Section("summary", "会议摘要 (中文，不超过300字)", "text"),
    Section("key_points", "会议要点 (中文，最多15条)"),
    Section("decisions", "会议决策 (中文，如无则输出空数组)"),
    Section("action_items", '待办事项 (中文，每条以"[ ] "开头，如无则输出空数组)'),
)

EARNINGS_SECTIONS = (
    Section("title", '会议标题 (英文，"<Company> <Quarter> <Year> Earnings Call"，'
                     "用公司自己的财季口径，如 Q1 2026 / FQ4 2026 / CQ2 2026)", "text"),
    Section("summary", "会议摘要 (中文，200-350字，概括本季度业绩关键数字、各业务板块表现、"
                       "资本支出、业绩指引与管理层基调)", "text"),
    Section("management_remarks", "管理层prepared remarks要点 (中文，10-15条，按发言顺序，保留具体数字)"),
    Section("qa", 'Q&A问答对，每位分析师提问一条，按提问顺序，覆盖所有提问。q: 分析师提问 (中文，'
                  '以分析师姓名与机构原文开头，格式 "Name (Firm)：问题")；a: 管理层回答要点 '
                  "(中文，2-4句，保留关键数字)", "qa"),
)

_MODE_SECTIONS = {
    "generic": GENERIC_SECTIONS,
    "meeting": MEETING_SECTIONS,
    "earnings": EARNINGS_SECTIONS,
}


def sections_for(mode: str, custom: Sequence[Section] | None = None) -> tuple[Section, ...]:
    """The sections a run produces: the mode's own, or the custom list. A
    custom list without a title gets the mode's title in front, since the
    title is also the note's H1 and its file name."""
    defaults = _MODE_SECTIONS[mode]
    if not custom:
        return defaults
    if any(s.key == "title" for s in custom):
        return tuple(custom)
    return (defaults[0], *custom)


def control_fields(mode: str) -> dict[str, str]:
    """Fields a mode asks for beyond its sections, name -> prompt."""
    return {QA_START_QUOTE: _QA_START_QUOTE_PROMPT} if mode == "earnings" else {}


_TYPE_WORDS = {"text": "string", "list": "array of strings",
               "qa": 'array of objects, each {"q": string, "a": string}'}


def sections_prompt(sections: Sequence[Section], controls: dict[str, str] | None = None) -> str:
    """The field list as the metadata system prompt states it."""
    lines = ["Output a JSON object with these fields:"]
    for s in sections:
        lines.append(f'- "{s.key}": {_TYPE_WORDS[s.type]} - {s.prompt}')
    for name, prompt in (controls or {}).items():
        lines.append(f'- "{name}": string - {prompt}')
    return "\n".join(lines)


def sections_schema(sections: Sequence[Section], controls: dict[str, str] | None = None) -> dict:
    """response_schema for the metadata call. Every field is required, so
    constrained decoding cannot skip one."""
    properties: dict[str, dict] = {}
    for s in sections:
        if s.type == "text":
            properties[s.key] = {"type": "string"}
        elif s.type == "list":
            properties[s.key] = {"type": "array", "items": {"type": "string"}}
        else:
            properties[s.key] = {"type": "array", "items": {
                "type": "object",
                "properties": {"q": {"type": "string"}, "a": {"type": "string"}},
                "required": ["q", "a"],
            }}
    for name in controls or {}:
        properties[name] = {"type": "string"}
    return {"type": "object", "properties": properties, "required": list(properties)}


def render_sections(sections: Sequence[Section], values: dict) -> list[str]:
    """Markdown lines for every section but the title, in section order. A
    section the model left empty (no decisions, say) is left out."""
    lines: list[str] = []
    for s in sections:
        if s.key == "title":
            continue
        value = values.get(s.key)
        if not value:
            continue
        lines += [f"## {s.title}", ""]
        if s.type == "text":
            lines.append(str(value).strip())
        elif s.type == "list":
            lines += [f"- {item}" for item in value if str(item).strip()]
        else:
            for pair in value:
                # Rendered as the notes written before sections were data.
                lines.append(f"- **Q**: {pair.get('q', '')}<br>  **A**: {pair.get('a', '')}")
        lines.append("")
    return lines
