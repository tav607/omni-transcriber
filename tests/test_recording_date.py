"""Uploaded and watched audio files get a date for the year anchor."""

import datetime
import unittest

from src.services.transcriber import (
    TranscriptionMetadata,
    _build_transcription_prompt,
    recording_date,
)

TODAY = datetime.date(2026, 11, 12)


class RecordingDateTests(unittest.TestCase):
    def test_date_in_the_filename_is_the_recording_date(self):
        name = "Amazon.com, Inc._Earnings Call_2026-08-11T00_00_00_English.mp3"
        self.assertEqual(recording_date(name, TODAY), ("2026-08-11", False))
        self.assertEqual(recording_date("20261105 expert call.m4a", TODAY), ("2026-11-05", False))

    def test_no_date_falls_back_to_the_arrival_day_as_an_upper_bound(self):
        self.assertEqual(recording_date("voice memo.m4a", TODAY), ("2026-11-12", True))

    def test_filename_date_a_day_ahead_of_a_utc_arrival_is_kept(self):
        self.assertEqual(recording_date("call 2026-11-13.m4a", TODAY), ("2026-11-13", False))

    def test_future_or_impossible_filename_date_is_ignored(self):
        self.assertEqual(recording_date("plan 2027-01-15.mp3", TODAY), ("2026-11-12", True))
        self.assertEqual(recording_date("id 20260231.mp3", TODAY), ("2026-11-12", True))


class DateAnchorPromptTests(unittest.TestCase):
    def prompt(self, upper_bound: bool) -> str:
        return _build_transcription_prompt(
            TranscriptionMetadata(source_name="Direct Upload", title="call",
                                  publish_date="2026-11-12", date_is_upper_bound=upper_bound),
            podcast_mode=True,
        )

    def test_exact_date_wording(self):
        prompt = self.prompt(False)
        self.assertIn("This recording is dated 2026-11-12.", prompt)
        self.assertNotIn("on or before", prompt)

    def test_upper_bound_wording(self):
        prompt = self.prompt(True)
        self.assertIn("This recording was made on or before 2026-11-12.", prompt)
        self.assertIn("may be newer than your knowledge", prompt)
        self.assertNotIn("is dated", prompt)


if __name__ == "__main__":
    unittest.main()
