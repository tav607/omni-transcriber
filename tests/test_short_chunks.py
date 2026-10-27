"""Short edit chunks are kept, and only pure markup is dropped (the skill's
edit_one rule), in both edit paths."""

import unittest
from unittest import mock

from src.services import editor

from .support import run_edit, run_edit_podcast

LONG = ("**Host:** So, uh, welcome back to the show. Today we are talking about how "
        "cloud providers plan data center capacity two to three years ahead of demand.")
EDITED_START = "**Host:** Welcome back to the show."  # the filler cut the guard accepts
SHORT = "**Guest:** Thanks for having me."  # real speech, under the edit threshold
MARKUP = "**00:42** [MUSIC PLAYING]\n\n**Host:**"  # nothing spoken
GENERIC_META = {"title": "测试", "summary": "摘要", "key_points": ["要点"]}
PODCAST_META = {"summary": "摘要", "takeaways": [], "qa_pairs": [], "highlights": []}


class _Editor:
    """Stands in for the edit call: records what it was asked to edit."""

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, chunk: str) -> str:
        self.calls.append(chunk)
        return chunk.replace("So, uh, welcome", "Welcome")


def _chunks(*parts):
    return mock.patch.object(editor, "_split_into_chunks", lambda text, *a, **k: list(parts))


class ResidualContentTests(unittest.TestCase):
    def test_markup_only_is_empty(self):
        self.assertEqual(editor._residual_content(MARKUP), "")
        self.assertEqual(editor._residual_content("[00:01:02] **Speaker 1:** [APPLAUSE]"), "")

    def test_labels_are_not_content(self):
        self.assertEqual(editor._residual_content(SHORT), "Thanks for having me.")

    def test_lowercase_tags_are_content(self):
        self.assertEqual(editor._residual_content("**Host:** [inaudible]"), "[inaudible]")


class PodcastPathTests(unittest.TestCase):
    def test_short_chunk_is_kept_raw_without_an_edit_call(self):
        edit = _Editor()
        with _chunks(LONG, SHORT):
            result = run_edit_podcast(LONG + "\n\n" + SHORT, edit, PODCAST_META)
        self.assertEqual(edit.calls, [LONG])
        self.assertTrue(result.full_transcript.endswith(SHORT))

    def test_only_pure_markup_is_dropped(self):
        edit = _Editor()
        with _chunks(LONG, MARKUP):
            result = run_edit_podcast(LONG + "\n\n" + MARKUP, edit, PODCAST_META)
        self.assertEqual(edit.calls, [LONG])
        self.assertNotIn("MUSIC", result.full_transcript)
        self.assertTrue(result.full_transcript.startswith(EDITED_START))

    def test_a_short_whole_transcript_survives(self):
        edit = _Editor()
        result = run_edit_podcast(SHORT, edit, PODCAST_META)
        self.assertEqual(edit.calls, [])
        self.assertEqual(result.full_transcript, SHORT)


class GenericPathTests(unittest.TestCase):
    def test_short_chunk_is_kept_raw_without_an_edit_call(self):
        edit = _Editor()
        with _chunks(LONG, SHORT):
            out = run_edit(LONG + "\n\n" + SHORT, edit, GENERIC_META)
        self.assertEqual(edit.calls, [LONG])
        self.assertIn(EDITED_START, out)
        self.assertIn(SHORT, out)

    def test_pure_markup_is_dropped_without_an_edit_call(self):
        # The guard has nothing to align in a markup-only chunk, so whatever the
        # editor said about it would pass straight through.
        edit = _Editor()
        with _chunks(LONG, MARKUP, SHORT):
            out = run_edit(LONG + "\n\n" + MARKUP + "\n\n" + SHORT, edit, GENERIC_META)
        self.assertEqual(edit.calls, [LONG])
        self.assertNotIn("MUSIC", out)
        self.assertIn(EDITED_START, out)
        self.assertIn("\n\n" + SHORT, out)
        self.assertNotIn("\n\n\n", out)


if __name__ == "__main__":
    unittest.main()
