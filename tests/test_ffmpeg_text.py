"""Tags that are not UTF-8 must not kill a run. Chinese podcast mp3s routinely
carry GBK-encoded ID3 frames, and ffmpeg echoes tags into stderr at its
default log level, so every ffmpeg/ffprobe call decodes with _FFMPEG_TEXT."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.services import transcriber

# A Chinese podcast title as GBK bytes, which are not valid UTF-8.
GBK_TITLE = "商业访谈录 第12期：半导体".encode("gbk")


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg not installed")
class NonUtf8TagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp(prefix="ffmpeg_text_test_"))
        cls.audio = cls.dir / "gbk_title.mp3"
        # The tag goes in as raw bytes, the way a GBK-tagged podcast file has it.
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
             "-i", "sine=frequency=220:duration=25:sample_rate=16000", "-ac", "1",
             b"-metadata", b"title=" + GBK_TITLE, str(cls.audio)],
            check=True, capture_output=True,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_the_file_puts_invalid_utf8_in_ffmpeg_stderr(self):
        # Keeps the next test honest: an ffmpeg that cleaned the tag up would
        # make it pass without exercising anything.
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(self.audio), "-f", "null", "-"],
            capture_output=True,
        )
        self.assertIn(GBK_TITLE, result.stderr)
        with self.assertRaises(UnicodeDecodeError):
            result.stderr.decode("utf-8")

    def test_split_survives_a_gbk_title_tag(self):
        # The stream copy runs at ffmpeg's default log level, which prints the
        # tags; with a strict text=True this raised UnicodeDecodeError.
        out = self.dir / "chunks"
        out.mkdir()
        paths, durations, total, starts = transcriber._split_audio(
            self.audio, out, max_duration_ms=15_000)
        self.assertGreater(len(paths), 1)
        self.assertTrue(all(p.exists() for p in paths))
        self.assertEqual(starts[-1] + durations[-1], total)

    def test_duration_probe_reads_the_file(self):
        self.assertAlmostEqual(transcriber._get_audio_duration(self.audio), 25_000, delta=100)


if __name__ == "__main__":
    unittest.main()
