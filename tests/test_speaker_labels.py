import logging
import unittest

from src.services import editor, transcriber

from .support import run_edit, run_edit_podcast


def unify(text: str) -> str:
    return transcriber._unify_prefix_name_labels(transcriber._unify_short_name_labels(text))


class PrefixLabelMergeTests(unittest.TestCase):
    def test_shortened_first_name_merges_into_the_majority_form(self):
        text = ("**Spence Neumann:** a\n**Spence Neumann:** b\n"
                "**Spencer Neumann:** c\n**Spencer Wang:** d")
        out = unify(text)
        self.assertNotIn("Spencer Neumann", out)
        self.assertEqual(out.count("**Spence Neumann:**"), 3)
        self.assertIn("**Spencer Wang:** d", out)

    def test_different_surnames_are_not_merged(self):
        text = "**Spence Neumann:** a\n**Spencer Wang:** b"
        self.assertEqual(unify(text), text)

    def test_a_third_label_with_the_surname_is_ambiguous(self):
        text = "**Spence Neumann:** a\n**Spencer Neumann:** b\n**Sp Neumann:** c"
        self.assertEqual(unify(text), text)

    def test_numbered_speakers_are_left_alone(self):
        text = "**Speaker 1:** a\n**Speaker 1:** b\n**Speaker 2:** c"
        self.assertEqual(unify(text), text)

    def test_merge_runs_after_stitching_chunks(self):
        first = ("**Spencer Neumann:** Revenue grew sixteen percent this quarter.\n\n"
                 "**Spencer Neumann:** Operating margin expanded as planned.")
        second = "**Spence Neumann:** And we raised the full-year outlook for the third time."
        with self.assertLogs(transcriber.logger, level=logging.WARNING):  # no overlap
            merged = transcriber._merge_transcriptions([first, second])
        self.assertEqual(merged.count("**Spencer Neumann:**"), 3)
        self.assertNotIn("Spence Neumann", merged)


if __name__ == "__main__":
    unittest.main()


class NormalizeAfterEditTests(unittest.TestCase):
    """Edit chunks are edited independently, so numbered labels are renumbered
    by first appearance once the edited chunks are joined."""

    def test_gaps_are_closed_in_order_of_first_appearance(self):
        text = "**Speaker 3:** a\n**Speaker 1:** b\n**Speaker 3:** c"
        self.assertEqual(editor._normalize_speaker_labels(text),
                         "**Speaker 1:** a\n**Speaker 2:** b\n**Speaker 1:** c")

    def test_a_swap_does_not_collapse_two_speakers(self):
        text = "**Speaker 2:** a\n**Speaker 1:** b\n**Speaker 2:** c"
        self.assertEqual(editor._normalize_speaker_labels(text),
                         "**Speaker 1:** a\n**Speaker 2:** b\n**Speaker 1:** c")

    def test_named_and_role_labels_are_left_alone(self):
        text = "**Host:** a\n**张三:** b\n**Guest:** c\n**Speaker 1:** d"
        self.assertEqual(editor._normalize_speaker_labels(text), text)

    def test_both_edit_paths_normalize(self):
        raw = ("**Speaker 2:** " + "我们今年的收入增长主要来自企业客户的续约和扩容。" * 3 + "\n\n"
               "**Speaker 1:** " + "那明年的指引大概是什么水平，能不能展开讲一下。" * 3)
        meta = {"title": "t", "summary": "s", "key_points": ["k"], "takeaways": ["k"],
                "qa_pairs": [], "highlights": []}
        generic = run_edit(raw, lambda chunk: chunk, meta)
        podcast = run_edit_podcast(raw, lambda chunk: chunk, meta)
        for out in (generic, podcast.full_transcript):
            self.assertLess(out.index("**Speaker 1:** 我们今年"), out.index("**Speaker 2:** 那明年"))
