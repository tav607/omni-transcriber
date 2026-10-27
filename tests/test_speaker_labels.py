import logging
import unittest

from src.services import transcriber


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
