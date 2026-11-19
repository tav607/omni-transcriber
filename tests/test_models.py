"""Model choices: Argon is the default for both components, and both offer it."""

import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path

from src.config import EditorConfig, TranscriberConfig
from src.utils import settings_store

# handlers loads the settings store when imported; point the store at a scratch
# file first (its import-time init() keeps the path) so the tests never touch
# data/user_settings.json.
_SETTINGS_DIR = Path(tempfile.mkdtemp(prefix="omni_settings_"))
settings_store.init(_SETTINGS_DIR / "user_settings.json")

from src.bot import handlers  # noqa: E402


def tearDownModule():
    shutil.rmtree(_SETTINGS_DIR, ignore_errors=True)


class _Chat:
    def __init__(self, chat_id: int):
        self.id = chat_id

    async def do(self, action):
        pass


class _Message:
    def __init__(self, chat_id: int):
        self.chat = _Chat(chat_id)
        self.answers: list[tuple[str, dict]] = []

    async def answer(self, text, **kwargs):
        self.answers.append((text, kwargs))

    async def delete(self):
        pass


class _Callback:
    def __init__(self, data: str, chat_id: int):
        self.data = data
        self.message = _Message(chat_id)

    async def answer(self, *args, **kwargs):
        pass


class ModelDefaultsTests(unittest.TestCase):
    def test_telegram_defaults(self):
        self.assertEqual(handlers.MODELS[handlers.DEFAULT_EDITOR_MODEL], "gemini-4-argon")
        self.assertEqual(handlers.MODELS[handlers.DEFAULT_TRANSCRIBER_MODEL], "gemini-4-argon")

    def test_config_defaults_for_the_watcher(self):
        self.assertEqual(EditorConfig(api_key="x").model, "gemini-4-argon")
        self.assertEqual(EditorConfig(api_key="x").translation_model, "gemini-4-hydrogen")
        self.assertEqual(TranscriberConfig(api_key="x").model, "gemini-4-argon")


class ModelMenuTests(unittest.TestCase):
    def test_argon_button_in_both_rows(self):
        message = _Message(chat_id=101)
        asyncio.run(handlers.cmd_model(message))
        text, kwargs = message.answers[0]
        rows = [[b.callback_data for b in row] for row in kwargs["reply_markup"].inline_keyboard]
        self.assertEqual(rows[1], ["model_transcriber_argon"])
        self.assertEqual(rows[3], ["model_editor_argon"])
        # A chat that never chose a model reads the defaults.
        self.assertIn("Transcriber: *ARGON* (`gemini-4-argon`)", text)
        self.assertIn("Editor: *ARGON* (`gemini-4-argon`)", text)

    def test_argon_callback_on_a_chat_that_stored_flash(self):
        # Argon is the only choice, so the chat already reads as Argon and the
        # callback just closes the menu.
        for chat_id, component in ((102, "editor"), (103, "transcriber")):
            settings_store.set(chat_id, f"{component}_model", "flash")
            callback = _Callback(f"model_{component}_argon", chat_id=chat_id)
            asyncio.run(handlers.model_callback(callback))
            self.assertEqual(handlers._chosen_model(chat_id, component), "argon")

    def test_callback_rejects_an_unknown_model(self):
        callback = _Callback("model_transcriber_ultra", chat_id=105)
        asyncio.run(handlers.model_callback(callback))
        self.assertIsNone(settings_store.get(105, "transcriber_model"))
        self.assertEqual(callback.message.answers, [])

    def test_flash_is_no_longer_a_choice(self):
        self.assertNotIn("flash", handlers.MODELS)
        for component in ("transcriber", "editor"):
            callback = _Callback(f"model_{component}_flash", chat_id=104)
            asyncio.run(handlers.model_callback(callback))
            self.assertIsNone(settings_store.get(104, f"{component}_model"))
        self.assertEqual(callback.message.answers, [])

    def test_a_stored_flash_choice_reads_as_the_default(self):
        settings_store.set(108, "transcriber_model", "flash")
        settings_store.set(108, "editor_model", "flash")
        self.assertEqual(handlers._chosen_model(108, "transcriber"), "argon")
        self.assertEqual(handlers._chosen_model(108, "editor"), "argon")

    def test_pro_is_no_longer_a_choice(self):
        self.assertNotIn("pro", handlers.MODELS)
        callback = _Callback("model_transcriber_pro", chat_id=106)
        asyncio.run(handlers.model_callback(callback))
        self.assertIsNone(settings_store.get(106, "transcriber_model"))
        self.assertEqual(callback.message.answers, [])

    def test_a_stored_pro_choice_reads_as_the_default(self):
        settings_store.set(107, "transcriber_model", "pro")
        settings_store.set(107, "editor_model", "pro")
        self.assertEqual(handlers._chosen_model(107, "transcriber"), "argon")
        self.assertEqual(handlers._chosen_model(107, "editor"), "argon")
        message = _Message(chat_id=107)
        asyncio.run(handlers.cmd_model(message))
        self.assertIn("Transcriber: *ARGON* (`gemini-4-argon`)", message.answers[0][0])


if __name__ == "__main__":
    unittest.main()
