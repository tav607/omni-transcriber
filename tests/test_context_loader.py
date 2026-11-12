"""CONTEXT.md: front matter settings, layering from the root down, and
errors instead of silent defaults."""

import asyncio
import unittest

from src.services.sections import Section
from src.watcher.context_loader import (
    ContextConfig,
    ContextError,
    context_paths,
    load_context,
    merge_layers,
    parse_context_md,
)

FULL = """---
mode: meeting
participants:
  - Jane: the client
  - Sam
glossary: [CoWoS, CCL]
translation: true
output_folder: Client Calls
sections:
  - key: summary
    type: text
    prompt: 会议摘要
  - key: risks
    heading: Risks Raised
    prompt: 提到的风险
---
Weekly call with the client.

<!-- a note to self that the model must not see -->
"""


def load(files: dict[str, str], audio: str, root: str = "") -> ContextConfig:
    async def read(path):
        value = files.get(path)
        if isinstance(value, Exception):
            raise value
        return value
    return asyncio.run(load_context(audio, root, read))


class ParseTests(unittest.TestCase):
    def test_front_matter_and_background(self):
        settings, background = parse_context_md(FULL)
        self.assertEqual(settings["mode"], "meeting")
        self.assertEqual(settings["participants"], (("Jane", "the client"), ("Sam", "")))
        self.assertEqual(settings["glossary"], ("CoWoS", "CCL"))
        self.assertIs(settings["translation"], True)
        self.assertEqual(settings["output_folder"], "Client Calls")
        self.assertEqual(settings["sections"], (
            Section("summary", "会议摘要", "text"),
            Section("risks", "提到的风险", "list", "Risks Raised"),
        ))
        self.assertEqual(background, "Weekly call with the client.")

    def test_only_stated_keys_are_returned(self):
        settings, background = parse_context_md("---\nmode: earnings\n---\n")
        self.assertEqual(settings, {"mode": "earnings"})
        self.assertEqual(background, "")

    def test_file_without_front_matter_is_all_background(self):
        settings, background = parse_context_md("Recordings of the Friday seminar.\n")
        self.assertEqual((settings, background), ({}, "Recordings of the Friday seminar."))

    def test_unfilled_template_comments_are_not_background(self):
        # The old loader sent the example participants inside this comment to
        # the model as the folder's background.
        settings, background = parse_context_md(
            "---\nmode: generic\n---\n<!--\n示例：参与者包括 陈明 (产品经理)\n-->\n")
        self.assertEqual(background, "")

    def test_old_format_is_refused(self):
        with self.assertRaisesRegex(ContextError, "old format"):
            parse_context_md("# Context\n\n## Background\n\nx\n\n## Sections\n\ny\n", "/a/CONTEXT.md")

    def assertRefused(self, front_matter: str, pattern: str):
        with self.assertRaisesRegex(ContextError, pattern):
            parse_context_md(f"---\n{front_matter}\n---\n", "/a/CONTEXT.md")

    def test_mistakes_are_named(self):
        self.assertRefused("mood: meeting", "unknown setting.*mood.*allowed: mode")
        self.assertRefused("mode: podcast", "mode 'podcast' is not one of generic, meeting, earnings")
        self.assertRefused("translation: yes please", "`translation` must be true or false")
        self.assertRefused("mode: [unclosed", "not valid YAML")
        self.assertRefused("roles: [BDA]", "at least two different labels")
        self.assertRefused("roles: [BDA, 专家]\nparticipants: [Jane]", "cannot both be set")
        self.assertRefused("participants: ['Jane: CEO*']", "cannot be a speaker label")
        self.assertRefused("output_folder: ../elsewhere", "plain subfolder path")
        self.assertRefused("sections:\n  - key: Summary\n    prompt: x", "section key 'Summary'")
        self.assertRefused("sections:\n  - key: summary", "needs a `prompt`")
        self.assertRefused("sections:\n  - key: a\n    prompt: x\n    type: table", "type 'table'")
        self.assertRefused("sections:\n  - key: a\n    prompt: x\n    title: A", "unknown field.*title")
        self.assertRefused("sections:\n  - key: a\n    prompt: x\n  - key: a\n    prompt: y", "appears twice")

    def test_error_names_the_file(self):
        with self.assertRaisesRegex(ContextError, "^/1 - Calls/CONTEXT.md: "):
            parse_context_md("---\nmode: nope\n---\n", "/1 - Calls/CONTEXT.md")


class LayeringTests(unittest.TestCase):
    ROOT = "---\nmode: meeting\nglossary: [BDA]\nparticipants: [Lingqi]\n---\nI am Lingqi, a consultant."
    AUDIO = "/7 - Expert Call/call 2026-11-05.m4a"

    def test_paths_run_from_the_root_down(self):
        self.assertEqual(context_paths("/a/b/x.m4a", ""), ["/CONTEXT.md", "/a/CONTEXT.md", "/a/b/CONTEXT.md"])
        self.assertEqual(context_paths("/rec/a/x.m4a", "/rec"), ["/rec/CONTEXT.md", "/rec/a/CONTEXT.md"])
        self.assertEqual(context_paths("/x.m4a", ""), ["/CONTEXT.md"])
        with self.assertRaises(ContextError):
            context_paths("/other/x.m4a", "/rec")
        with self.assertRaises(ContextError):
            context_paths("/rec/../x.m4a", "/rec")

    def test_no_context_anywhere_gives_the_defaults(self):
        config = load({}, self.AUDIO)
        self.assertEqual(config, ContextConfig())
        self.assertEqual(config.mode, "meeting")
        self.assertIn("config: none (defaults)", config.describe())

    def test_folder_overrides_and_adds_to_the_root(self):
        folder = "---\nroles: [BDA, 专家]\nglossary: [GLG, BDA]\ntranslation: true\n---\nExpert network calls."
        config = load({"/CONTEXT.md": self.ROOT, "/7 - Expert Call/CONTEXT.md": folder}, self.AUDIO)
        self.assertEqual(config.mode, "meeting")                       # inherited
        self.assertEqual(config.roles, ("BDA", "专家"))                 # stated here
        self.assertEqual(config.participants, ())                      # dropped: roles took over
        self.assertEqual(config.glossary, ("BDA", "GLG"))              # added up, no repeats
        self.assertEqual(config.background, "I am Lingqi, a consultant.\n\nExpert network calls.")
        self.assertIs(config.translation, True)
        self.assertEqual(config.sources, ("/CONTEXT.md", "/7 - Expert Call/CONTEXT.md"))

    def test_folder_without_its_own_file_gets_the_root(self):
        config = load({"/CONTEXT.md": self.ROOT}, "/2 - Client Call - General/x.m4a")
        self.assertEqual(config.participants, (("Lingqi", ""),))
        self.assertEqual(config.sources, ("/CONTEXT.md",))

    def test_unreadable_file_stops_the_run(self):
        # The old loader took a failed download for "no CONTEXT.md here".
        files = {"/CONTEXT.md": self.ROOT,
                 "/7 - Expert Call/CONTEXT.md": ContextError("could not be read from Dropbox")}
        with self.assertRaises(ContextError):
            load(files, self.AUDIO)

    def test_describe_states_what_applies(self):
        merged = merge_layers([("/CONTEXT.md", {"mode": "earnings", "translation": True}, "")])
        self.assertEqual(merged.describe(), "mode: earnings · translation: on · config: /CONTEXT.md")


if __name__ == "__main__":
    unittest.main()
