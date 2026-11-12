"""Number guard on the edit pass. The NumberGuardTests are ported verbatim
from the transcribe skill's test_number_guard.py; the last class checks that
the edit paths report what the guard finds."""

import asyncio
import logging
import unittest
from unittest import mock

from src.services import editor
from src.services.edit_guard import _number_changes


class NumberGuardTests(unittest.TestCase):
    """The post-edit guard must flag a changed figure and stay quiet on the
    formatting and filler edits the editor is supposed to make."""

    def assertUnchanged(self, raw: str, edited: str):
        self.assertEqual(_number_changes(raw, edited), ([], []),
                         f"{raw!r} -> {edited!r} should not count as a number change")

    def test_changed_valuation_is_flagged(self):
        raw = "之前1000万美金融完的时候呢，在硅谷的这个估值是40 亿美金。"
        edited = "之前 1000 万美金融完的时候，在硅谷估值是 4000 万美金。"
        self.assertEqual(_number_changes(raw, edited), (["40亿"], ["4000万"]))

    def test_changed_percentage_is_flagged(self):
        dropped, added = _number_changes("占整个市场0.5%", "占整个市场0.05%")
        self.assertEqual((dropped, added), (["0.5%"], ["0.05%"]))

    def test_spoken_and_digit_forms_are_one_value(self):
        self.assertUnchanged("占整个市场百分之零点五", "占整个市场 0.5%")
        self.assertUnchanged("估值四十亿美金", "估值 40 亿美金")
        self.assertUnchanged("大概四千三百万", "大概 4300 万")
        self.assertUnchanged("ARR is 4,000 million", "ARR is 4000 million")
        self.assertUnchanged("revenue was $5.2B", "revenue was $5.2 billion")
        self.assertUnchanged("3M users, 20k seats", "3 million users, 20 thousand seats")

    def test_measure_word_between_value_and_scale(self):
        self.assertUnchanged("可能 2.8 个亿美金", "可能 2.8 亿美金")
        self.assertUnchanged("13到14个 million 的 ARR", "13 到 14 million ARR")

    def test_stutter_and_filler_removal_is_quiet(self):
        self.assertUnchanged("从25年年初开始的。啊25年年初，然后呃一个一个公司",
                             "从 25 年年初开始的。25 年年初，然后一个公司")

    def test_timestamps_and_labels_are_ignored(self):
        self.assertUnchanged("**12:34** **Speaker 1:** 收入是3亿", "**Speaker 1:** 收入是 3 亿")


class GuardEditReportsNumbersTests(unittest.TestCase):
    """_guard_edit is the one hook both edit paths run on every edited chunk."""

    def guard(self, raw: str, edited: str) -> str:
        # The fidelity guard is taken out so the number check sees the edit as is.
        with mock.patch.object(editor, "_restore_edit_losses", lambda r, e: (e, 0)):
            return asyncio.run(editor._guard_edit(raw, edited, 2, []))

    def test_changed_number_is_logged_with_its_chunk(self):
        with self.assertLogs(editor.logger, level=logging.WARNING) as captured:
            out = self.guard("估值是40亿美金。", "估值是4000万美金。")
        self.assertEqual(out, "估值是4000万美金。")
        self.assertIn("Editor changed numbers in chunk 3: dropped 40亿; introduced 4000万",
                      "\n".join(captured.output))

    def test_reformatted_number_is_quiet(self):
        with self.assertNoLogs(editor.logger, level=logging.WARNING):
            self.guard("估值四十亿美金", "估值 40 亿美金")

    def test_every_edit_prompt_forbids_changing_numbers(self):
        for prompt in (editor.RAW_EDIT_SYSTEM_PROMPT,
                       editor.RAW_EDIT_WITH_CONTEXT_SYSTEM_PROMPT,
                       editor.PODCAST_RAW_EDIT_SYSTEM_PROMPT):
            self.assertIn("绝对不要改动数字及其量级", prompt)


if __name__ == "__main__":
    unittest.main()
