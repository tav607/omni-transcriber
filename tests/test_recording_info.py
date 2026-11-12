"""The summary step is told the file's name and recording date (folder mode)."""

import unittest

from src.services.editor import RecordingInfo
from tests.support import run_edit

RAW = "**Dad:** 这是一个小屁孩去魔法森林的故事。\n\n**Chuck:** 那是彩色的魔法苹果吗？"
META = {"title": "t", "summary": "s", "key_points": []}


def summary_contents(**edit_kwargs) -> str:
    """The user content of the metadata call edit() makes."""
    seen = []
    run_edit(RAW, lambda chunk: chunk, META, on_summary=seen.append, **edit_kwargs)
    return seen[0]


class RecordingInfoTest(unittest.TestCase):
    def test_file_name_and_exact_date_reach_the_summary(self):
        contents = summary_contents(recording=RecordingInfo("魔法森林第一集：苹果篇", "2026-11-12"))
        self.assertIn("- File name: 魔法森林第一集：苹果篇", contents)
        self.assertIn("Recorded on 2026-11-12.", contents)

    def test_upper_bound_date_is_worded_as_one(self):
        contents = summary_contents(
            recording=RecordingInfo("Audio", "2026-08-12", date_is_upper_bound=True))
        self.assertIn("Recorded on or before 2026-08-12.", contents)

    def test_background_is_kept_alongside(self):
        contents = summary_contents(
            recording=RecordingInfo("Audio", "2026-08-12"), background="这是 Dad 给 Chuck 讲的故事")
        self.assertIn("## Recording", contents)
        self.assertIn("这是 Dad 给 Chuck 讲的故事", contents)
        self.assertLess(contents.index("## Recording"), contents.index("## Transcript"))

    def test_no_date_no_date_line(self):
        contents = summary_contents(recording=RecordingInfo("Audio"))
        self.assertIn("- File name: Audio", contents)
        self.assertNotIn("Recorded", contents)

    def test_absent_by_default(self):
        self.assertNotIn("## Recording", summary_contents())


if __name__ == "__main__":
    unittest.main()
