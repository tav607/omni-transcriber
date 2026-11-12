"""The Dropbox watcher: what counts as work, what is held back, and the
note the processor writes."""

import asyncio
import datetime
import unittest
from unittest import mock

import dropbox
import yaml
from dropbox.files import FileMetadata, GetMetadataError, LookupError as DropboxLookupError

from src.config import DropboxWatcherConfig
from src.watcher import dropbox_watcher
from src.watcher.context_loader import ContextConfig, ContextError
from src.watcher.dropbox_watcher import DropboxWatcher
from src.watcher.processor import is_audio_file, note_front_matter, unique_name


def meta(path: str) -> FileMetadata:
    return FileMetadata(name=path.rsplit("/", 1)[-1], path_display=path, path_lower=path.lower(),
                        size=1000, client_modified=datetime.datetime(2026, 11, 5, 9, 30))


class FakeBot:
    def __init__(self):
        self.messages: list[str] = []

    async def send_message(self, chat_id, text, parse_mode=None):
        self.messages.append(text)


class WatcherCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.files: dict[str, str] = {}        # CONTEXT.md path -> content
        self.processed: list[tuple[str, ContextConfig, datetime.date]] = []
        self.bot = FakeBot()
        with mock.patch.object(dropbox_watcher.dropbox, "Dropbox", lambda **kw: mock.Mock()):
            self.watcher = DropboxWatcher(
                DropboxWatcherConfig(app_key="k", app_secret="s", refresh_token="t",
                                     watch_folder="", telegram_chat_id=1), self.bot)

        async def read_file(path):
            value = self.files.get(path)
            if isinstance(value, Exception):
                raise value
            return value

        async def process(dropbox_path, context, bot, config, file_size_bytes=0, client_modified=None):
            self.processed.append((dropbox_path, context, client_modified))
            return True

        self.watcher._read_file = read_file
        patcher = mock.patch.object(dropbox_watcher, "process_audio_file", process)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def settle(self):
        while self.watcher._tasks:
            await asyncio.gather(*list(self.watcher._tasks))


class WorkSelectionTests(WatcherCase):
    def test_what_counts_as_work(self):
        should = self.watcher._should_process_file
        self.assertTrue(should(meta("/1 - Calls/call.m4a")))
        self.assertTrue(should(meta("/1 - Calls/call.opus")))
        self.assertFalse(should(meta("/1 - Calls/_processed/call.m4a")))
        self.assertFalse(should(meta("/1 - Calls/_failed/call.m4a")))
        self.assertFalse(should(meta("/1 - Calls/CONTEXT.md")))
        self.assertTrue(is_audio_file("x.MP3"))

    async def test_files_already_waiting_at_startup_are_processed(self):
        # The old watcher took a cursor past them: a file dropped while the
        # service was down was never seen.
        listing = [meta("/1 - Calls/waiting.m4a"), meta("/1 - Calls/_processed/done.m4a"),
                   meta("/1 - Calls/CONTEXT.md")]

        async def list_folder():
            return listing, "cursor-1"

        self.watcher._list_folder = list_folder
        await self.watcher._start_from_listing()
        await self.settle()
        self.assertEqual(self.watcher.cursor, "cursor-1")
        self.assertEqual([p for p, _, _ in self.processed], ["/1 - Calls/waiting.m4a"])
        self.assertEqual(self.processed[0][2], datetime.date(2026, 11, 5))

    async def test_a_file_is_not_started_twice(self):
        f = meta("/1 - Calls/call.m4a")
        self.assertEqual(self.watcher._schedule([f, f]), 1)
        await self.settle()
        self.assertEqual(len(self.processed), 1)


class ContextProblemTests(WatcherCase):
    AUDIO = "/1 - Calls/call.m4a"

    async def test_bad_context_holds_the_file_back_and_says_why(self):
        self.files["/1 - Calls/CONTEXT.md"] = "---\nmode: podcast\n---\n"
        self.watcher._schedule([meta(self.AUDIO)])
        await self.settle()
        self.assertEqual(self.processed, [])
        self.assertIn(self.AUDIO, self.watcher.blocked)
        self.assertEqual(len(self.bot.messages), 1)
        self.assertIn("CONTEXT.md needs fixing", self.bot.messages[0])
        self.assertIn("mode &#x27;podcast&#x27; is not one of", self.bot.messages[0])

    async def test_unreadable_context_holds_the_file_back(self):
        self.files["/CONTEXT.md"] = ContextError("/CONTEXT.md: could not be read from Dropbox (timeout)")
        self.watcher._schedule([meta(self.AUDIO)])
        await self.settle()
        self.assertEqual(self.processed, [])
        self.assertIn(self.AUDIO, self.watcher.blocked)

    async def test_fixed_context_lets_the_file_through_with_its_settings(self):
        self.files["/1 - Calls/CONTEXT.md"] = "---\nmode: podcast\n---\n"
        self.watcher._schedule([meta(self.AUDIO)])
        await self.settle()
        self.files["/1 - Calls/CONTEXT.md"] = "---\nmode: earnings\n---\n"
        retry = list(self.watcher.blocked.values())
        self.watcher.blocked.clear()
        self.watcher._schedule(retry)
        await self.settle()
        self.assertEqual([(p, c.mode) for p, c, _ in self.processed], [(self.AUDIO, "earnings")])
        self.assertEqual(self.watcher.blocked, {})


class ReadFileTests(unittest.IsolatedAsyncioTestCase):
    def watcher(self, get_metadata) -> DropboxWatcher:
        dbx = mock.Mock()
        dbx.files_get_metadata = get_metadata
        with mock.patch.object(dropbox_watcher.dropbox, "Dropbox", lambda **kw: dbx):
            return DropboxWatcher(DropboxWatcherConfig(
                app_key="k", app_secret="s", refresh_token="t", watch_folder="",
                rclone_base_path="/Apps/omni"), FakeBot())

    @staticmethod
    def not_found(path):
        raise dropbox.exceptions.ApiError(
            "req", GetMetadataError.path(DropboxLookupError.not_found), "not found", "en")

    async def test_missing_file_is_none(self):
        self.assertIsNone(await self.watcher(self.not_found)._read_file("/a/CONTEXT.md"))

    async def test_existing_file_comes_through_rclone(self):
        seen = []

        async def cat(path):
            seen.append(path)
            return "---\nmode: generic\n---\n"

        with mock.patch.object(dropbox_watcher, "rclone_cat", cat):
            text = await self.watcher(lambda path: object())._read_file("/a/CONTEXT.md")
        self.assertEqual(text, "---\nmode: generic\n---\n")
        self.assertEqual(seen, ["dropbox:/Apps/omni/a/CONTEXT.md"])

    async def test_file_that_exists_but_cannot_be_fetched_is_an_error(self):
        async def cat(path):
            return None

        with mock.patch.object(dropbox_watcher, "rclone_cat", cat), \
                mock.patch.object(dropbox_watcher.asyncio, "sleep", mock.AsyncMock()):
            with self.assertRaisesRegex(ContextError, "could not be read from Dropbox"):
                await self.watcher(lambda path: object())._read_file("/a/CONTEXT.md")


class NoteTests(unittest.TestCase):
    def test_same_title_on_the_same_day_does_not_overwrite(self):
        taken = {"周会 - 20261105.md", "周会 - 20261105 (2).md", "其他.md"}
        self.assertEqual(unique_name("周会 - 20261105", taken), "周会 - 20261105 (3)")
        self.assertEqual(unique_name("新标题 - 20261105", taken), "新标题 - 20261105")

    def test_front_matter_names_the_source(self):
        text = note_front_matter("1 - Calls/Call: with Dai 20261105.m4a", "2026-11-05", "meeting")
        self.assertTrue(text.startswith("---\n") and text.endswith("---\n\n"))
        self.assertIn("\nrecorded: 2026-11-05\n", text)   # bare, a date to Obsidian
        self.assertEqual(yaml.safe_load(text.strip().strip("-")), {
            "source": "1 - Calls/Call: with Dai 20261105.m4a",
            "recorded": datetime.date(2026, 11, 5), "mode": "meeting"})


if __name__ == "__main__":
    unittest.main()
