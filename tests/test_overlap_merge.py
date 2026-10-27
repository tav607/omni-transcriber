import logging
import unittest
from unittest import mock

from src.services import transcriber

TRADITIONAL = (
    "這個問題會對整體營運產生壓力，貿易戰導致關稅與匯率變動難以預測，我們預計明年的資本開支"
    "會維持在現有水準，但是產能擴張的節奏會放緩，因為終端需求還沒有完全恢復，庫存調整也需要時間。"
)
SIMPLIFIED = (
    "这个问题会对整体营运产生压力，贸易战导致关税与汇率变动难以预测，我们预计明年的资本开支"
    "会维持在现有水准，但是产能扩张的节奏会放缓，因为终端需求还没有完全恢复，库存调整也需要时间。"
)


class SeamScriptFoldingTests(unittest.TestCase):
    def setUp(self):
        # One chunk came back traditional and the next simplified (two October
        # expert calls in the skill). Most characters of the passage differ
        # between the two scripts.
        self.previous = "前面的讨论到这里。\n\n**专家:** " + TRADITIONAL
        self.current = "**专家:** " + SIMPLIFIED + "\n\n**BDA:** 那性能呢？这是新的问题。"

    def test_script_flip_between_chunks_still_matches(self):
        merged = transcriber._merge_transcriptions([self.previous, self.current])
        self.assertEqual(merged.count("貿易戰") + merged.count("贸易战"), 1)
        self.assertTrue(merged.endswith("**BDA:** 那性能呢？这是新的问题。"))

    def test_without_folding_the_overlap_is_duplicated(self):
        # Control: the fold is what finds this seam.
        with mock.patch.object(transcriber, "_zh_fold", None), \
                self.assertLogs(transcriber.logger, level=logging.WARNING):
            merged = transcriber._merge_transcriptions([self.previous, self.current])
        self.assertEqual(merged.count("貿易戰") + merged.count("贸易战"), 2)

    def test_mapping_points_at_the_original_characters(self):
        clean, mapping = transcriber._strip_for_matching_with_map("**专家:** 這個，問題")
        self.assertEqual(clean, "这个问题")
        self.assertEqual([("**专家:** 這個，問題")[i] for i in mapping], list("這個問題"))


if __name__ == "__main__":
    unittest.main()
