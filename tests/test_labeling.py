"""CONTEXT.md settings reach the transcription and edit prompts."""

import logging
import unittest

from src.services import editor, transcriber
from src.services.transcriber import (
    CODE_SWITCH_RULE,
    EARNINGS_SPEAKER_GUIDELINE,
    GENERIC_SPEAKER_GUIDELINE,
    MEETING_SPEAKER_GUIDELINE,
    PODCAST_SPEAKER_GUIDELINE,
    Labeling,
    _build_transcription_prompt,
    _fix_role_swap,
)

from .support import TranscribeRun, run_edit


def prompt(labeling=None, podcast_mode=False) -> str:
    return _build_transcription_prompt(None, podcast_mode, labeling)


class TranscriptionPromptTests(unittest.TestCase):
    def test_modes_pick_their_guideline(self):
        self.assertIn(MEETING_SPEAKER_GUIDELINE + CODE_SWITCH_RULE, prompt(Labeling("meeting")))
        self.assertIn(EARNINGS_SPEAKER_GUIDELINE, prompt(Labeling("earnings")))
        self.assertNotIn(CODE_SWITCH_RULE, prompt(Labeling("earnings")))
        self.assertIn(GENERIC_SPEAKER_GUIDELINE, prompt(Labeling("generic")))

    def test_without_labeling_the_bot_paths_are_unchanged(self):
        self.assertIn(PODCAST_SPEAKER_GUIDELINE, prompt(podcast_mode=True))
        self.assertEqual(prompt(), transcriber.TRANSCRIPTION_PROMPT)

    def test_roles_replace_the_mode_guideline(self):
        text = prompt(Labeling("meeting", roles=("BDA", "专家")))
        self.assertIn("exactly one of these labels and no other: **BDA:**, **专家:**", text)
        self.assertIn("**BDA:** is the side asking the questions", text)
        self.assertIn("**专家:** is the side answering them", text)
        self.assertNotIn("This is a recording of a meeting or a call", text)
        self.assertIn(CODE_SWITCH_RULE, text)

    def test_three_roles_put_the_interviewee_last(self):
        text = prompt(Labeling("meeting", roles=("BDA", "客户", "专家")))
        self.assertIn("**BDA:**, **客户:** are the interviewer side", text)
        self.assertIn("**专家:** is the interviewee", text)

    def test_participants_are_added_to_the_mode_guideline(self):
        text = prompt(Labeling("meeting", participants=(("Jane", "the client"), ("Sam", ""))))
        self.assertIn(MEETING_SPEAKER_GUIDELINE, text)
        self.assertIn("Expected participants and their labels: **Jane:** (the client); **Sam:**.", text)
        self.assertIn("Not everyone listed has to be present", text)


class RoleSwapTests(unittest.TestCase):
    ASK = "**{0}:** 这个市场今年大概多大？\n\n"
    ANSWER = "**{0}:** 大概两百亿，增速在百分之二十左右。\n\n"

    def chunk(self, asker: str, answerer: str, turns: int = 4) -> str:
        return (self.ASK.format(asker) + self.ANSWER.format(answerer)) * turns

    def test_inverted_chunk_is_swapped_back(self):
        text, swapped = _fix_role_swap(self.chunk("专家", "BDA"), "BDA", "专家")
        self.assertTrue(swapped)
        self.assertEqual(text, self.chunk("BDA", "专家"))

    def test_correct_chunk_and_short_chunk_are_left_alone(self):
        good = self.chunk("BDA", "专家")
        self.assertEqual(_fix_role_swap(good, "BDA", "专家"), (good, False))
        short = self.chunk("专家", "BDA", turns=2)
        self.assertEqual(_fix_role_swap(short, "BDA", "专家"), (short, False))

    def test_transcribe_swaps_the_inverted_chunk_of_a_roster_run(self):
        good, bad = self.chunk("BDA", "专家", 6), self.chunk("专家", "BDA", 6)
        answers = {"chunk_000.mp3": good, "chunk_001.mp3": bad}
        run = TranscribeRun(lambda source, model: answers[source.inline_data.data.decode()], chunks=2)
        with self.assertLogs(transcriber.logger, level=logging.WARNING) as captured:
            text = run.run(labeling=Labeling("meeting", roles=("BDA", "专家")))
        self.assertEqual(text.count("**BDA:** 这个市场"), 12)
        self.assertNotIn("**专家:** 这个市场", text)
        self.assertIn("chunk 2/2 had BDA and 专家 swapped", "\n".join(captured.output))


RAW = ("**Speaker 2:** " + "我们今年的收入增长主要来自企业客户的续约和扩容。" * 3 + "\n\n"
       "**Speaker 1:** " + "那明年的指引大概是什么水平，能不能展开讲一下。" * 3)
META = {"title": "t", "summary": "s", "key_points": ["k"], "decisions": [], "action_items": [],
        "management_remarks": ["m"], "qa": [], "_qa_start_quote": ""}


class EditPromptTests(unittest.TestCase):
    def edit(self, raw=RAW, **kwargs):
        calls = []
        out = run_edit(raw, lambda chunk: chunk, META,
                       on_edit=lambda system, contents: calls.append((system, contents)), **kwargs)
        return out, calls[0]

    def test_modes_pick_their_edit_prompt(self):
        self.assertEqual(self.edit(mode="meeting")[1][0], editor.MEETING_RAW_EDIT_PROMPT)
        self.assertEqual(self.edit(mode="earnings")[1][0], editor.EARNINGS_RAW_EDIT_PROMPT)
        self.assertEqual(self.edit()[1][0], editor.RAW_EDIT_SYSTEM_PROMPT)
        self.assertEqual(self.edit(background="ctx")[1][0], editor.RAW_EDIT_WITH_CONTEXT_SYSTEM_PROMPT)

    def test_roles_are_fixed_in_the_prompt_and_not_renumbered(self):
        raw = RAW.replace("Speaker 2", "专家").replace("Speaker 1", "BDA")
        out, (system, _) = self.edit(raw, mode="meeting", roles=("BDA", "专家"))
        self.assertTrue(system.startswith(editor.MEETING_RAW_EDIT_PROMPT))
        self.assertIn("fixed roster: **BDA:** / **专家:**", system)
        self.assertLess(out.index("**专家:**"), out.index("**BDA:**"))

    def test_without_roles_numbered_labels_are_renumbered(self):
        out, _ = self.edit(mode="meeting")
        self.assertLess(out.index("**Speaker 1:** 我们今年"), out.index("**Speaker 2:** 那明年"))

    def test_folder_glossary_reaches_the_editor(self):
        _, (_, contents) = self.edit(mode="meeting", glossary=("CoWoS", "CCL"))
        self.assertIn("## Terminology (use these exact spellings consistently)\nCoWoS, CCL\n", contents)


if __name__ == "__main__":
    unittest.main()
