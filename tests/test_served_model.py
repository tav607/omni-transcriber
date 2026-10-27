import asyncio
import unittest

from google.genai import types

from src.services import editor, transcriber
from src.utils import gemini


def _response(text: str, served: str) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(
            role="model", parts=[types.Part.from_text(text=text)]))],
        model_version=served,
    )


class _Models:
    def __init__(self, served: str):
        self.served = served
        self.requested: list[str] = []

    def generate_content(self, model, contents, config):
        self.requested.append(model)
        return _response("A transcript long enough to pass every check.", self.served)


class _Client:
    def __init__(self, served: str):
        self.models = _Models(served)


class ServedModelTests(unittest.TestCase):
    def setUp(self):
        gemini.SERVED_MODELS.clear()

    def tearDown(self):
        gemini.SERVED_MODELS.clear()

    def test_reroute_is_reported_once(self):
        with self.assertLogs(gemini.logger, level="WARNING") as captured:
            gemini._check_served_model("gemini-3.7-flash", "gemini-3.8-flash")
            gemini._check_served_model("models/gemini-3.7-flash", "gemini-3.8-flash")
        self.assertEqual(len(captured.output), 1)
        self.assertIn("Requested gemini-3.7-flash but Google served gemini-3.8-flash",
                      captured.output[0])
        self.assertEqual(gemini.served_model_mismatches(), ["gemini-3.7-flash -> gemini-3.8-flash"])

    def test_alias_and_version_suffix_are_not_reroutes(self):
        with self.assertNoLogs(gemini.logger, level="WARNING"):
            gemini._check_served_model("gemini-pro-latest", "gemini-3.1-pro-preview")
            gemini._check_served_model("gemini-3.8-flash", "gemini-3.8-flash")
            gemini._check_served_model("gemini-x", "models/gemini-x-001")
            gemini._check_served_model("gemini-3.8-flash", None)
        self.assertEqual(gemini.served_model_mismatches(), [])

    def test_editor_calls_are_checked(self):
        # Every editor-side call (edit, metadata, translation, glossary) goes
        # through this one wrapper.
        client = _Client(served="gemini-3.8-flash")
        with self.assertLogs(gemini.logger, level="WARNING"):
            asyncio.run(editor._gemini_generate(client, "gemini-3.7-flash", "text"))
        self.assertEqual(gemini.served_model_mismatches(), ["gemini-3.7-flash -> gemini-3.8-flash"])

    def test_transcription_calls_are_checked(self):
        client = _Client(served="gemini-3.8-flash")
        audio = types.Part.from_bytes(data=b"\x00", mime_type="audio/mpeg")
        with self.assertLogs(gemini.logger, level="WARNING"):
            text = asyncio.run(transcriber._transcribe_audio(
                client, audio, "gemini-3.7-flash", 1.0, "high"))
        self.assertTrue(text.startswith("A transcript"))
        self.assertEqual(gemini.served_model_mismatches(), ["gemini-3.7-flash -> gemini-3.8-flash"])


if __name__ == "__main__":
    unittest.main()
