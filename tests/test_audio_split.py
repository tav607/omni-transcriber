"""Audio is cut in pauses. Ported verbatim from the transcribe skill's
test_audio_split.py (its TRANSCRIBE module handle is `transcriber` here); the
audio is synthesized with ffmpeg's lavfi sources, so no fixture files."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.services import transcriber


def synth(path: Path, parts: list[tuple[str, float]]) -> None:
    """Write mono 16 kHz mp3: ("tone", s) is steady speech-level sound, ("pause", s) faint noise."""
    cmd, labels = ["ffmpeg", "-y", "-v", "error"], []
    for i, (kind, seconds) in enumerate(parts):
        src = (f"sine=frequency=220:duration={seconds}:sample_rate=16000" if kind == "tone"
               else f"anoisesrc=duration={seconds}:amplitude=0.002:sample_rate=16000")
        cmd += ["-f", "lavfi", "-i", src]
        labels.append(f"[{i}]")
    cmd += ["-filter_complex", "".join(labels) + f"concat=n={len(parts)}:v=0:a=1", "-ac", "1", str(path)]
    subprocess.run(cmd, check=True, capture_output=True)


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
class PauseCutTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="split_test_"))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_cut_goes_through_the_longest_pause(self):
        audio = self.dir / "a.mp3"
        synth(audio, [("tone", 3), ("pause", 0.4), ("tone", 2), ("pause", 1.2), ("tone", 3)])
        cut = transcriber._pause_cut(audio, 0, 9.6)
        self.assertAlmostEqual(cut, 5.4 + 0.6, delta=0.15)

    def test_no_pause_means_no_cut(self):
        audio = self.dir / "b.mp3"
        synth(audio, [("tone", 8)])
        self.assertIsNone(transcriber._pause_cut(audio, 0, 8))

    def test_chunks_meet_in_pauses_without_shared_audio(self):
        audio = self.dir / "c.mp3"
        synth(audio, [("tone", 12), ("pause", 1.0), ("tone", 14), ("pause", 0.6), ("tone", 12.4)])
        paths, durations, total, starts = transcriber._split_audio(audio, self.dir, max_duration_ms=15_000)
        self.assertEqual(len(paths), 3)
        self.assertEqual(transcriber._seam_overlaps_s(durations, starts), [0.0, 0.0])
        self.assertAlmostEqual(starts[1] / 1000, 12.5, delta=0.15)
        self.assertAlmostEqual(starts[2] / 1000, 27.3, delta=0.15)
        self.assertEqual(starts[-1] + durations[-1], total)

    def test_without_a_pause_chunks_overlap_as_before(self):
        audio = self.dir / "d.mp3"
        synth(audio, [("tone", 25)])
        paths, durations, total, starts = transcriber._split_audio(audio, self.dir, max_duration_ms=15_000)
        overlaps = transcriber._seam_overlaps_s(durations, starts)
        self.assertTrue(overlaps)
        self.assertTrue(all(o == transcriber.OVERLAP_SECONDS for o in overlaps))

    def test_short_audio_is_one_chunk(self):
        audio = self.dir / "e.mp3"
        synth(audio, [("tone", 5)])
        paths, durations, total, starts = transcriber._split_audio(audio, self.dir, max_duration_ms=15_000)
        self.assertEqual((paths, starts), ([audio], [0]))


if __name__ == "__main__":
    unittest.main()
