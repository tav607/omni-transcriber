"""Edit chunking: cut at speaker turns, pieces sized evenly, and a smaller cap
for CJK-heavy transcripts because Argon's output cap also holds its thinking."""

import math
import unittest
from unittest import mock

from src.services import editor, transcriber

from .support import run_edit, run_edit_podcast

ZH = ["我们在新加坡的数据中心今年扩容了一倍。", "主要是客户的向量检索负载增长很快。",
      "明年的资本开支会维持在现有水准。", "库存调整也需要一些时间来消化。"]
EN = ["We doubled the Singapore data center this year.", "Vector search load grew fast.",
      "Capital spending stays flat next year.", "Inventory needs a few quarters to clear."]


def _dialogue(sentences: list[str], turns: int, per_turn: int = 6) -> str:
    out = []
    for t in range(turns):
        label = ("**BDA:**", "**专家:**")[t % 2]
        body = "".join(sentences[(t + k) % len(sentences)] for k in range(per_turn))
        out.append(f"{label} {body} ({t})")
    return "\n\n".join(out)


class SplitTests(unittest.TestCase):
    def test_cjk_transcript_splits_evenly_under_the_cjk_cap(self):
        text = _dialogue(ZH, 400)
        self.assertTrue(transcriber._is_cjk_heavy(text))
        self.assertGreater(len(text), 2 * editor.CJK_CHUNK_SIZE)
        chunks = editor._split_into_chunks(text, editor.CJK_CHUNK_SIZE)
        self.assertEqual("".join(chunks), text)  # lossless
        self.assertTrue(all(len(c) <= editor.CJK_CHUNK_SIZE for c in chunks))
        # As many pieces as the length needs at 90% of the cap, and no sliver.
        self.assertEqual(len(chunks), math.ceil(len(text) / (0.9 * editor.CJK_CHUNK_SIZE)))
        self.assertGreater(min(map(len, chunks)), 0.5 * max(map(len, chunks)))
        self.assertTrue(all(c.startswith(("**BDA:**", "**专家:**")) for c in chunks))

    def test_just_over_the_cap_leaves_no_sliver(self):
        # The previous splitter filled 32,000 characters and left the rest, a
        # few hundred, as one more chunk that still cost a whole edit call.
        text = "**BDA:** " + EN[0]
        t = 1
        while len(text) <= editor.CHUNK_SIZE + 500:
            text += "\n\n" + ("**BDA:** ", "**专家:** ")[t % 2] + EN[t % 4] + f" ({t})"
            t += 1
        chunks = editor._split_into_chunks(text, editor.CHUNK_SIZE)
        self.assertEqual(len(chunks), 2)
        self.assertEqual("".join(chunks), text)
        self.assertGreater(min(map(len, chunks)), 0.6 * max(map(len, chunks)))

    def test_unlabeled_paragraphs_are_lossless(self):
        text = "\n\n".join(" ".join(EN[(p + k) % 4] for k in range(8)) for p in range(300))
        chunks = editor._split_into_chunks(text, editor.CJK_CHUNK_SIZE)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("\n\n".join(chunks), text)
        self.assertTrue(all(len(c) <= editor.CJK_CHUNK_SIZE for c in chunks))

    def test_bold_labels_are_turn_boundaries(self):
        # omni's previous splitter only knew "Name: " / "**Name**: " labels.
        text = "**Host:** a\n\n**Jane Doe:** b\n\nHost: c\n\n**Bob**: d"
        starts = [m.start() + 1 for m in editor._EDIT_SPLIT_BOUNDARY_RE.finditer(text)]
        self.assertEqual([text[s:s + 4] for s in starts], ["**Ja", "Host", "**Bo"])

    def test_a_turn_longer_than_the_cap_keeps_its_label_on_every_piece(self):
        paragraphs = ["".join(ZH[(p + k) % 4] for k in range(10)) + f"（{p}）" for p in range(200)]
        text = "**BDA:** 请介绍一下。\n\n**专家:** " + "\n\n".join(paragraphs) + "\n\n**BDA:** 谢谢。"
        chunks = editor._split_into_chunks(text, editor.CJK_CHUNK_SIZE)
        # A forced cut snaps to a paragraph break up to CHUNK_OVERLAP past the
        # cap (the skill's behavior), so only these pieces may run a little over.
        self.assertTrue(all(len(c) <= editor.CJK_CHUNK_SIZE + editor.CHUNK_OVERLAP for c in chunks))
        expert = [c for c in chunks if any(p in c for p in paragraphs)]
        self.assertGreater(len(expert), 1)
        self.assertTrue(all(c.startswith("**专家:** ") for c in expert))
        for p in paragraphs:  # every paragraph exactly once
            self.assertEqual(sum(c.count(p) for c in chunks), 1)

class EditorsPickTheCapTests(unittest.TestCase):
    """Both editors pass CJK_CHUNK_SIZE for CJK-heavy text, CHUNK_SIZE otherwise."""

    META = {"title": "测试", "summary": "摘要", "key_points": ["要点"]}
    PODCAST_META = {"summary": "摘要", "takeaways": [], "qa_pairs": [], "highlights": []}

    def _caps(self, run, text, meta):
        caps = []

        def split(transcript, max_size=editor.CHUNK_SIZE):
            caps.append(max_size)
            return [transcript]

        with mock.patch.object(editor, "_split_into_chunks", split):
            run(text, lambda chunk: chunk, meta)
        return caps

    def test_generic_editor(self):
        self.assertEqual(self._caps(run_edit, _dialogue(ZH, 2), self.META), [editor.CJK_CHUNK_SIZE])
        self.assertEqual(self._caps(run_edit, _dialogue(EN, 2), self.META), [editor.CHUNK_SIZE])

    def test_podcast_editor(self):
        self.assertEqual(self._caps(run_edit_podcast, _dialogue(ZH, 2), self.PODCAST_META),
                         [editor.CJK_CHUNK_SIZE])
        self.assertEqual(self._caps(run_edit_podcast, _dialogue(EN, 2), self.PODCAST_META),
                         [editor.CHUNK_SIZE])


if __name__ == "__main__":
    unittest.main()
