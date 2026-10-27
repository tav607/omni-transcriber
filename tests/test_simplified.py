import unittest

from src.services.editor import PodcastEpisodeMetadata
from src.utils.chinese import _to_simplified

from .support import TranscribeRun, run_edit, run_edit_podcast


class SimplifiedScriptTests(unittest.TestCase):
    def test_traditional_folds_to_simplified(self):
        self.assertEqual(_to_simplified("**張三：** 台積電的產能"), "**张三：** 台积电的产能")

    def test_english_and_numbers_are_untouched(self):
        text = "**Jane Doe:** CoWoS capacity reached 1,200k in 2026."
        self.assertEqual(_to_simplified(text), text)

    def test_japanese_is_left_alone(self):
        text = "客户：日本の開発チームは藝術的なゲームを作っています。"
        self.assertEqual(_to_simplified(text), text)


class PipelineFoldingTests(unittest.TestCase):
    def test_traditional_chunk_is_folded_before_merging(self):
        run = TranscribeRun(lambda source, model: "**張三:** 台積電今年的產能會繼續擴張，這點我們在法說會上已經講得很清楚了。")
        self.assertEqual(run.run(podcast_mode=True),
                         "**张三:** 台积电今年的产能会继续扩张，这点我们在法说会上已经讲得很清楚了。")

    def test_generic_edit_output_and_title_are_folded(self):
        raw = "**專家:** 我們早在二零二二年底就先試點了向量檢索，「這是」很早的決定。"
        out = run_edit(raw, lambda chunk: chunk, {
            "title": "台積電法說會", "summary": "營收成長", "key_points": ["產能擴張"]})
        self.assertTrue(out.startswith("# 台积电法说会"))
        self.assertIn("营收成长", out)
        self.assertIn("我们早在二零二二年底就先试点了向量检索", out)
        # zhconv turns 「」 into curly quotes; they still end up straight.
        self.assertIn('"这是"', out)

    def test_podcast_edit_markdown_and_title_are_folded(self):
        raw = "**主持人:** 歡迎收聽本期節目，今天我們聊聊半導體產業的最新變化和長期趨勢。"
        episode = PodcastEpisodeMetadata(podcast_name="科技島讀", episode_title="晶片戰爭")
        # zh-cn also localizes vocabulary (晶片 -> 芯片), as in the skill.
        result = run_edit_podcast(raw, lambda chunk: chunk, {
            "summary": "討論半導體", "takeaways": ["產能"], "qa_pairs": [], "highlights": []},
            episode=episode)
        self.assertEqual(result.title, "科技岛读 - 芯片战争")
        self.assertTrue(result.markdown.startswith("# 科技岛读 - 芯片战争"))
        self.assertIn("欢迎收听本期节目", result.markdown)
        self.assertIn("讨论半导体", result.markdown)


if __name__ == "__main__":
    unittest.main()
