"""Sections as data: one list drives the prompt, the schema and the Markdown."""

import json
import unittest
from unittest import mock

from src.services import editor
from src.services.sections import (
    EARNINGS_SECTIONS,
    GENERIC_SECTIONS,
    MEETING_SECTIONS,
    QA_START_QUOTE,
    Section,
    control_fields,
    render_sections,
    sections_for,
    sections_prompt,
    sections_schema,
)

from .support import run_edit

CUSTOM = (
    Section("summary", "摘要", "text"),
    Section("data_points", "量化数据"),
    Section("follow_ups", "待跟进问题", heading="Follow-ups"),
)


class SectionListTests(unittest.TestCase):
    def test_mode_defaults(self):
        self.assertIs(sections_for("generic"), GENERIC_SECTIONS)
        self.assertIs(sections_for("meeting"), MEETING_SECTIONS)
        self.assertIs(sections_for("earnings"), EARNINGS_SECTIONS)

    def test_custom_list_without_a_title_gets_the_modes_title(self):
        sections = sections_for("meeting", CUSTOM)
        self.assertEqual([s.key for s in sections], ["title", "summary", "data_points", "follow_ups"])
        self.assertEqual(sections[0], MEETING_SECTIONS[0])

    def test_custom_title_is_kept(self):
        own = (Section("title", "英文标题", "text"), *CUSTOM)
        self.assertEqual(sections_for("meeting", own), own)

    def test_only_earnings_asks_for_the_qa_start(self):
        self.assertEqual(list(control_fields("earnings")), [QA_START_QUOTE])
        self.assertEqual(control_fields("meeting"), {})

    def test_prompt_lists_every_field_with_its_type(self):
        prompt = sections_prompt(EARNINGS_SECTIONS, control_fields("earnings"))
        self.assertIn('- "summary": string - ', prompt)
        self.assertIn('- "management_remarks": array of strings - ', prompt)
        self.assertIn('- "qa": array of objects, each {"q": string, "a": string} - ', prompt)
        self.assertIn(f'- "{QA_START_QUOTE}": string - ', prompt)

    def test_schema_covers_question_answer_pairs(self):
        # The old free-text definitions fell back to unconstrained JSON here.
        schema = sections_schema(EARNINGS_SECTIONS, control_fields("earnings"))
        self.assertEqual(schema["required"], ["title", "summary", "management_remarks", "qa", QA_START_QUOTE])
        self.assertEqual(schema["properties"]["qa"]["items"]["required"], ["q", "a"])
        self.assertEqual(schema["properties"]["management_remarks"],
                         {"type": "array", "items": {"type": "string"}})

    def test_rendering_follows_the_list(self):
        lines = render_sections(sections_for("meeting", CUSTOM), {
            "title": "t", "follow_ups": ["c"], "summary": "s", "data_points": ["a", "b"]})
        self.assertEqual(lines, ["## Summary", "", "s", "",
                                 "## Data Points", "", "- a", "- b", "",
                                 "## Follow-ups", "", "- c", ""])

    def test_empty_section_is_left_out(self):
        lines = render_sections(MEETING_SECTIONS, {
            "title": "t", "summary": "s", "key_points": ["k"], "decisions": [], "action_items": []})
        self.assertEqual(lines, ["## Summary", "", "s", "", "## Key Points", "", "- k", ""])

    def test_question_answer_rendering(self):
        lines = render_sections(EARNINGS_SECTIONS, {"qa": [{"q": "Jane Doe (Firm)：问题", "a": "回答"}]})
        self.assertEqual(lines, ["## Q&A Highlights", "", "- **Q**: Jane Doe (Firm)：问题<br>  **A**: 回答", ""])


RAW = ("**Operator:** " + "Welcome to the second quarter earnings call, all lines are muted. " * 3 + "\n\n"
       "**Jane Doe:** " + "Revenue grew sixteen percent and margins expanded this quarter. " * 3 + "\n\n"
       "**Operator:** We will now open the call up for questions. Our first question comes from Sam Roe.\n\n"
       "**Sam Roe:** " + "Could you talk about capital expenditure for the rest of the year? " * 2)


class EditWithSectionsTests(unittest.TestCase):
    def test_earnings_mode_splits_the_transcript_at_the_qa_start(self):
        meta = {"title": "Acme Q2 2026 Earnings Call", "summary": "摘要", "management_remarks": ["要点"],
                "qa": [{"q": "Sam Roe (Firm)：资本支出？", "a": "回答"}],
                QA_START_QUOTE: "Operator: We will now open the call up for questions."}
        out = run_edit(RAW, lambda chunk: chunk, meta, mode="earnings")
        self.assertTrue(out.startswith("# Acme Q2 2026 Earnings Call\n"))
        self.assertLess(out.index("## Management Remarks"), out.index("## Q&A Highlights"))
        self.assertNotIn(QA_START_QUOTE, out)
        management, qa = out.split("### Q&A Section")
        self.assertIn("### Management Discussion Section", management)
        self.assertIn("Revenue grew sixteen percent", management)
        self.assertTrue(qa.lstrip().startswith("**Operator:** We will now open the call up"))

    def test_metadata_call_is_constrained_by_the_sections(self):
        seen = {}

        def capture(system, schema):
            seen["system"], seen["schema"] = system, schema

        meta = {"title": "t", "summary": "s", "data_points": ["a"], "follow_ups": []}
        out = run_edit(RAW, lambda chunk: chunk, meta, mode="meeting", sections=CUSTOM, on_metadata=capture)
        self.assertEqual(list(seen["schema"]["properties"]), ["title", "summary", "data_points", "follow_ups"])
        self.assertIn('- "data_points": array of strings - 量化数据', seen["system"])
        self.assertIn("## Data Points", out)
        self.assertNotIn("## Follow-ups", out)   # empty, left out

    def test_telegram_default_is_unchanged(self):
        out = run_edit(RAW, lambda chunk: chunk, {"title": "标题", "summary": "摘要", "key_points": ["要点"]})
        self.assertTrue(out.startswith("# 标题\n\n## Summary\n\n摘要\n\n## Key Points\n\n- 要点\n\n## Transcript\n"))


if __name__ == "__main__":
    unittest.main()
