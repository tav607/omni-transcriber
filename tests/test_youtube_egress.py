"""The network exit yt-dlp uses for YouTube (YOUTUBE_SOURCE_ADDRESS)."""

import asyncio
import os
import tempfile
import unittest
from unittest import mock

from src.services import downloader

YOUTUBE = "https://youtu.be/dQw4w9WgXcQ"
BILIBILI = "https://www.bilibili.com/video/BV1xx411c7mD"


class EgressOptsTest(unittest.TestCase):
    def test_unset_leaves_default_route(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("YOUTUBE_SOURCE_ADDRESS", None)
            self.assertEqual(downloader._youtube_egress_opts(YOUTUBE), {})

    def test_blank_counts_as_unset(self):
        with mock.patch.dict(os.environ, {"YOUTUBE_SOURCE_ADDRESS": "  "}):
            self.assertEqual(downloader._youtube_egress_opts(YOUTUBE), {})

    def test_youtube_gets_source_address(self):
        with mock.patch.dict(os.environ, {"YOUTUBE_SOURCE_ADDRESS": "172.16.0.2"}):
            self.assertEqual(
                downloader._youtube_egress_opts(YOUTUBE), {"source_address": "172.16.0.2"}
            )

    def test_other_sites_keep_default_route(self):
        with mock.patch.dict(os.environ, {"YOUTUBE_SOURCE_ADDRESS": "172.16.0.2"}):
            self.assertEqual(downloader._youtube_egress_opts(BILIBILI), {})


class DownloadOptsTest(unittest.TestCase):
    def _opts_for(self, url: str) -> dict:
        seen = {}

        def fake(url_, ydl_opts, video_id, output_dir):
            seen.update(ydl_opts)
            path = os.path.join(output_dir, f"{video_id}.mp3")
            open(path, "wb").close()
            return path, None

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            downloader, "_download_with_ytdlp", fake
        ), mock.patch.dict(os.environ, {"YOUTUBE_SOURCE_ADDRESS": "172.16.0.2"}):
            asyncio.run(downloader.download_audio(url, output_dir=tmp))
        return seen

    def test_download_passes_source_address_for_youtube(self):
        self.assertEqual(self._opts_for(YOUTUBE).get("source_address"), "172.16.0.2")

    def test_download_leaves_bilibili_alone(self):
        self.assertNotIn("source_address", self._opts_for(BILIBILI))


if __name__ == "__main__":
    unittest.main()
