"""Edit fidelity guard. The first 14 tests are ported verbatim from the
transcribe skill's test_edit_guard.py; the rest check that both edit paths
run the guard."""

import unittest
from unittest import mock

from src.services import editor
from src.services.edit_guard import _restore_edit_losses

from .support import run_edit, run_edit_podcast


def guard(raw: str, edited: str) -> str:
    return _restore_edit_losses(raw, edited)[0]


class CleanupSurvivesTests(unittest.TestCase):
    """Edits the prompts ask for are kept."""

    def test_fillers_and_stutters(self):
        raw = "**专家:** 嗯，那个，我们我们早在二零二二年底就，就是先试点了向量检索。"
        edited = "**专家:** 我们早在二零二二年底就先试点了向量检索。"
        self.assertEqual(guard(raw, edited), edited)

    def test_english_fillers_and_repeat(self):
        raw = "**Expert:** So, uh, you know, we we moved the workload to Pinecone."
        edited = "**Expert:** We moved the workload to Pinecone."
        self.assertEqual(guard(raw, edited), edited)

    def test_restated_false_start(self):
        raw = "第二，我感觉在美国市场价格战的这个，就是大家打价格战相对比中国要好吧。"
        edited = "第二，我感觉在美国市场，大家打价格战相对比中国要好吧。"
        self.assertEqual(guard(raw, edited), edited)

    def test_number_formatting_variant(self):
        raw = "We grew five percent and hired twenty people."
        edited = "We grew 5 percent and hired 20 people."
        self.assertEqual(guard(raw, edited), edited)

    def test_timestamps_removed_on_purpose(self):
        raw = "**00:00** **Host:** Welcome back. **01:23** Today we talk about vector databases."
        edited = "**Host:** Welcome back. Today we talk about vector databases."
        self.assertEqual(guard(raw, edited), edited)


class LossesAreRestoredTests(unittest.TestCase):
    """Content the editor dropped or reworded comes back from the raw text."""

    def test_dropped_sentence(self):
        raw = ("**专家:** 可以。那我就不以我离职之后来说了，就以我按那个来说。"
               "我们早在2022年底先在商品图搜场景试点了向量检索。")
        edited = "**专家:** 可以。我们早在2022年底先在商品图搜场景试点了向量检索。"
        out = guard(raw, edited)
        self.assertIn("离职之后来说了", out)
        self.assertIn("商品图搜场景", out)

    def test_reworded_claim(self):
        raw = ("**专家:** When I did the costing, the cost difference only because of the "
               "database is 45%, it's higher.")
        edited = "**专家:** When I did the costing, Oracle was 45% higher because of the database."
        out = guard(raw, edited)
        self.assertNotIn("Oracle was", out)
        self.assertIn("only because of the database is 45%", out)

    def test_negation_is_never_cleanup(self):
        raw = "The driver is not the costing part, it is the Solaris end of life."
        edited = "The driver is the costing part, it is the Solaris end of life."
        self.assertIn("not the costing part", guard(raw, edited))

    def test_unitless_number_stays_unitless(self):
        raw = "**专家:** 能到一个七八这样子都是很好的。"
        edited = "**专家:** 能到7%~8%都是很好的。"
        out = guard(raw, edited)
        self.assertIn("七八", out)
        self.assertNotIn("7%", out)

    def test_inserted_words_at_turn_start_keep_the_label(self):
        raw = "**BDA:** 好的。\n\n**专家:** 我们用的是自建的Milvus。"
        edited = "**BDA:** 好的。\n\n**专家:** 早期我们用的是自建的Milvus。"
        out = guard(raw, edited)
        self.assertIn("**专家:**", out)
        self.assertNotIn("早期", out)

    def test_numbers_fused_by_dropped_punctuation(self):
        raw = "**专家:** 大概加起来可能得六……600多左右了吧，七、八个人。"
        edited = "**专家:** 大概加起来可能得六600多左右了吧，七八个人。"
        out = guard(raw, edited)
        self.assertIn("六……600多", out)
        self.assertIn("七、八个", out)

    def test_numbers_left_with_only_a_space_between(self):
        self.assertEqual(guard("We sold 5, 6 units.", "We sold 5 6 units."), "We sold 5, 6 units.")

    def test_kept_number_separator_is_not_touched(self):
        raw = "Q3, 4 and 5 were flat."
        self.assertEqual(guard(raw, "Q3, 4 and 5 were flat."), raw)

    def test_restored_count(self):
        raw = "Revenue grew 12% and margins expanded to 40% in the quarter."
        edited = "Revenue grew and margins expanded in the quarter."
        out, n = _restore_edit_losses(raw, edited)
        self.assertEqual(out, raw)
        self.assertEqual(n, 2)


RAW = ("**专家:** 可以。那我就不以我离职之后来说了，就以我按那个来说。"
       "我们早在2022年底先在商品图搜场景试点了向量检索，后来又扩展到了推荐和广告。")
LOSSY = "**专家:** 可以。我们早在2022年底先在商品图搜场景试点了向量检索，后来又扩展到了推荐和广告。"


class EditPathsRunTheGuardTests(unittest.TestCase):
    def test_generic_edit(self):
        with self.assertLogs(editor.logger, level="INFO") as captured:
            out = run_edit(RAW, lambda chunk: LOSSY, {
                "title": "向量检索", "summary": "摘要", "key_points": ["要点"]})
        self.assertIn("离职之后来说了", out)
        self.assertIn("Edit chunk 1: put back 1 span(s)", "\n".join(captured.output))
        self.assertIn("Edit guard: put back 1 span(s)", "\n".join(captured.output))

    def test_podcast_edit(self):
        with self.assertLogs(editor.logger, level="INFO") as captured:
            result = run_edit_podcast(RAW, lambda chunk: LOSSY, {
                "summary": "摘要", "takeaways": [], "qa_pairs": [], "highlights": []})
        self.assertIn("离职之后来说了", result.full_transcript)
        self.assertIn("离职之后来说了", result.markdown)
        self.assertIn("Edit guard: put back 1 span(s)", "\n".join(captured.output))

    def test_failed_guard_keeps_the_raw_chunk(self):
        # A guard failure must not drop content: the chunk falls back to raw,
        # like an edit that failed after retries.
        def broken(raw, edited):
            raise RuntimeError("guard bug")

        with mock.patch.object(editor, "_restore_edit_losses", broken), \
                self.assertLogs(editor.logger, level="WARNING"):
            out = run_edit(RAW, lambda chunk: LOSSY, {
                "title": "向量检索", "summary": "摘要", "key_points": ["要点"]})
        self.assertIn(RAW.removeprefix("**专家:** "), out)


if __name__ == "__main__":
    unittest.main()
