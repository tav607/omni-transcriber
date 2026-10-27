"""Chunk seam merge. OverlapMergeTests is ported verbatim from the transcribe
skill's test_overlap_merge.py (its TRANSCRIBE module handle is `transcriber`
here); ScriptFoldingTests are omni's own."""

import logging
import unittest
import unittest.mock

from src.services import transcriber


class OverlapMergeTests(unittest.TestCase):
    def test_script_flip_between_chunks_still_matches(self):
        # One chunk came back traditional and the next simplified (two October
        # expert calls); the seam must still be found or both copies survive.
        # Most characters of this sentence differ between the two scripts, so
        # without folding the seam falls below the identity threshold.
        previous = "前面的讨论。\n\n**专家:** 這個問題會對整體營運產生壓力，貿易戰導致關稅與匯率變動難以預測。"
        current = (
            "**专家:** 这个问题会对整体营运产生压力，贸易战导致关税与汇率变动难以预测。"
            "\n\n**BDA:** 那性能呢？"
        )
        merged = transcriber._merge_transcriptions([previous, current])
        self.assertEqual(merged.count("貿易戰") + merged.count("贸易战"), 1)
        self.assertTrue(merged.rstrip().endswith("**BDA:** 那性能呢？"))

    def test_exact_english_suffix_prefix_overlap(self):
        overlap = (
            "**Operator:** The next question comes from the line of Jason Helfstein "
            "with Oppenheimer & Company. Please proceed with your question."
        )
        previous = "Earlier unique discussion about revenue and margins.\n\n" + overlap
        current = overlap + "\n\n**Jason Helfstein:** Thanks. My question is about AWS demand."

        skip = transcriber._find_overlap_length(previous, current)

        self.assertEqual(current[skip:].lstrip(), "**Jason Helfstein:** Thanks. My question is about AWS demand.")

    def test_operator_template_after_overlap_is_preserved(self):
        overlap = (
            "**Operator:** The next question comes from the line of Jason Helfstein "
            "with Oppenheimer & Company. Please proceed with your question."
        )
        next_operator = (
            "**Operator:** Okay, the next question comes from the line of Ken Gawrelski "
            "with Wells Fargo. Please proceed with your question."
        )
        new_question = "**Ken Gawrelski:** My first question is about advertising trends."
        previous = "Earlier unique discussion.\n\n" + overlap
        current = overlap + "\n\n" + next_operator + "\n\n" + new_question

        merged = transcriber._merge_transcriptions([previous, current])

        self.assertEqual(merged.count(overlap), 1)
        self.assertIn(next_operator, merged)
        self.assertIn(new_question, merged)

    def test_repeated_template_earlier_in_previous_is_not_overlap(self):
        operator = (
            "**Operator:** The next question comes from the line of Ken Gawrelski "
            "with Wells Fargo. Please proceed with your question."
        )
        previous = operator + "\n\n**Executive:** We are now concluding the prepared remarks."
        current = operator + "\n\n**Ken Gawrelski:** My question is new content."

        self.assertEqual(transcriber._find_overlap_length(previous, current), 0)

    def test_minor_transcription_variation_is_accepted(self):
        previous = (
            "The quarter was driven by higher advertising demand, especially among "
            "retail and travel customers."
        )
        current = (
            "This quarter was driven by stronger advertising demand, particularly from "
            "retail and travel clients. The next topic is capital expenditure."
        )

        skip = transcriber._find_overlap_length(previous, current)

        self.assertGreater(skip, 0)
        self.assertIn("The next topic is capital expenditure.", current[skip:])

    def test_chinese_short_overlap(self):
        overlap = "感谢大家参加今天的电话会议，下面有请管理层介绍本季度业绩。"
        previous = "欢迎各位投资者。" + overlap
        current = overlap + "公司本季度收入保持稳健增长。"

        skip = transcriber._find_overlap_length(previous, current)

        self.assertEqual(current[skip:], "公司本季度收入保持稳健增长。")

    def test_chinese_overlap_before_long_english_text(self):
        overlap = "感谢大家参加今天的电话会议下面有请管理层介绍本季度业绩"
        previous = "欢迎各位投资者" + overlap
        current = overlap + " This is entirely new English content after the seam. " * 8

        skip = transcriber._find_overlap_length(previous, current)

        self.assertTrue(current[skip:].startswith("This is entirely new English content"))

    def test_short_mixed_cjk_overlap_before_english_text(self):
        overlap = "欢迎参加今天的AI电话会讨论GPU和云业务增长"
        previous = "此前内容。" + overlap
        current = overlap + " This is entirely new English material after the audio seam."

        skip = transcriber._find_overlap_length(previous, current)

        self.assertTrue(current[skip:].startswith("This is entirely new English material"))

    def test_very_short_mixed_cjk_overlap_preserves_new_word(self):
        overlap = "欢迎参加AI电话会讨论GPU云增长"
        previous = "此前完全不同的内容。" + overlap
        current = overlap + " New unique English content follows."

        skip = transcriber._find_overlap_length(previous, current)

        self.assertEqual(current[skip:], "New unique English content follows.")

    def test_unicode_casefold_expansion_keeps_mapping_valid(self):
        overlap = "Die Straße und die oﬃzielle İST-Zahl bleiben im Bericht unverändert"
        previous = "Einleitung " + overlap
        current = overlap + ". Neue Inhalte folgen."

        skip = transcriber._find_overlap_length(previous, current)

        self.assertEqual(current[skip:], "Neue Inhalte folgen.")

    def test_new_speaker_turn_is_never_consumed(self):
        overlap = (
            "**Operator:** The next question comes from the line of Jason Helfstein "
            "with Oppenheimer & Company. Please proceed with your question."
        )
        previous = overlap + " Thank you."
        current = overlap + "\n\n**Jason:** New question."

        skip = transcriber._find_overlap_length(previous, current)

        self.assertEqual(current[skip:].lstrip(), "**Jason:** New question.")

    def test_new_speaker_protected_when_prev_tail_has_other_labels(self):
        operator = (
            "**Operator:** The next question comes from the line of Jason Helfstein "
            "with Oppenheimer & Company. Please proceed with your question."
        )
        previous = (
            "**Andy:** We are seeing strong growth in our advertising business "
            "this quarter across every region. " + operator + " Thank you."
        )
        current = operator + "\n\n**Jason:** What is the outlook for AWS demand and capacity?"

        skip = transcriber._find_overlap_length(previous, current)

        self.assertEqual(
            current[skip:].lstrip(),
            "**Jason:** What is the outlook for AWS demand and capacity?",
        )

    def test_new_speaker_protected_when_next_starts_mid_sentence(self):
        overlap = (
            "comes from the line of Jason Helfstein with Oppenheimer & Company. "
            "Please proceed with your question."
        )
        previous = (
            "**Andy:** We keep investing in capacity. The next question "
            + overlap + " Thank you."
        )
        current = overlap + "\n\n**Jason:** My question is about advertising."

        skip = transcriber._find_overlap_length(previous, current)

        self.assertEqual(current[skip:].lstrip(), "**Jason:** My question is about advertising.")

    def test_overlap_can_span_multiple_speaker_turns(self):
        overlap = (
            "**Host:** First speaker overlap sentence contains enough distinctive "
            "English words for reliable matching before turn. "
            "**Guest:** Second speaker overlap sentence also contains enough "
            "distinctive English words for matching."
        )
        current = overlap + "\n\n**Host:** New material."

        skip = transcriber._find_overlap_length(overlap, current)

        self.assertEqual(current[skip:].lstrip(), "**Host:** New material.")

    def test_different_speaker_labels_do_not_break_overlap(self):
        previous = "**Host:** Welcome back. Today we are discussing cloud infrastructure demand."
        current = (
            "**Speaker 1:** Welcome back. Today we are discussing cloud infrastructure demand.\n\n"
            "**Guest:** Demand remains strong."
        )

        skip = transcriber._find_overlap_length(previous, current)

        self.assertEqual(current[skip:].lstrip(), "**Guest:** Demand remains strong.")

    def test_boundary_does_not_snap_into_new_sentence(self):
        overlap = "This is the exact shared boundary phrase for both adjacent audio chunks"
        current = overlap + " New material ends here. Another sentence follows."

        skip = transcriber._find_overlap_length("Unique prefix. " + overlap, current)

        self.assertEqual(current[skip:], "New material ends here. Another sentence follows.")

    def test_closing_quote_at_boundary_is_consumed(self):
        overlap = "The shared quotation reaches the exact end of the previous audio chunk"
        current = overlap + ".” New material follows."

        skip = transcriber._find_overlap_length("He said “" + overlap, current)

        self.assertEqual(current[skip:], "New material follows.")

    def test_opening_straight_quote_of_new_content_is_preserved(self):
        overlap = "The shared boundary phrase reaches exactly the end of the previous chunk here"
        current = overlap + '. "New quoted material follows," she said.'

        skip = transcriber._find_overlap_length("Intro. " + overlap, current)

        self.assertEqual(current[skip:], '"New quoted material follows," she said.')

    def test_safety_limit_prefers_duplication_over_data_loss(self):
        current = "x" * (transcriber.MAX_OVERLAP_SKIP + 1) + "new content"
        with unittest.mock.patch.object(
            transcriber, "_find_overlap_length", return_value=transcriber.MAX_OVERLAP_SKIP + 1
        ):
            merged = transcriber._merge_transcriptions(["previous", current])

        self.assertTrue(merged.endswith(current))

    def test_large_accepted_skip_logs_warning(self):
        current = "x" * transcriber.OVERLAP_WARNING_SKIP + "new content"
        with unittest.mock.patch.object(
            transcriber, "_find_overlap_length", return_value=transcriber.OVERLAP_WARNING_SKIP
        ):
            with self.assertLogs(transcriber.logger, logging.WARNING) as captured:
                transcriber._merge_transcriptions(["previous", current])

        self.assertTrue(any("Removed" in message for message in captured.output))


    def test_seam_cut_in_a_pause_is_joined_without_alignment(self):
        prev = "**Host:** We moved the workload to Pinecone last year."
        nxt = "**Host:** We moved the workload to Pinecone last year, and costs fell."
        with unittest.mock.patch.object(transcriber, "_find_overlap_length") as align:
            merged = transcriber._merge_transcriptions([prev, nxt], [0.0])
        align.assert_not_called()
        self.assertEqual(merged, prev + "\n\n" + nxt)

    def test_overlap_seams_still_align(self):
        with unittest.mock.patch.object(transcriber, "_find_overlap_length", return_value=5) as align:
            merged = transcriber._merge_transcriptions(["previous", "dupe new"], [10.0])
        align.assert_called_once()
        self.assertEqual(merged, "previous\n\nnew")

    def test_sessions_without_seam_overlaps_align_every_seam(self):
        with unittest.mock.patch.object(transcriber, "_find_overlap_length", return_value=0) as align:
            transcriber._merge_transcriptions(["a", "b", "c"])
        self.assertEqual(align.call_count, 2)

    def test_seam_overlaps_from_the_split(self):
        self.assertEqual(transcriber._seam_overlaps_s([15_000, 15_000, 9_000], [0, 12_500, 17_500]),
                         [2.5, 10.0])


class ScriptFoldingTests(unittest.TestCase):
    PREVIOUS = "前面的讨论。\n\n**专家:** 這個問題會對整體營運產生壓力，貿易戰導致關稅與匯率變動難以預測。"
    CURRENT = ("**专家:** 这个问题会对整体营运产生压力，贸易战导致关税与汇率变动难以预测。"
               "\n\n**BDA:** 那性能呢？")

    def test_without_folding_the_overlap_is_duplicated(self):
        # Control for test_script_flip_between_chunks_still_matches: the fold
        # is what finds this seam.
        with unittest.mock.patch.object(transcriber, "_zh_fold", None), \
                self.assertLogs(transcriber.logger, level=logging.WARNING):
            merged = transcriber._merge_transcriptions([self.PREVIOUS, self.CURRENT])
        self.assertEqual(merged.count("貿易戰") + merged.count("贸易战"), 2)

    def test_mapping_points_at_the_original_characters(self):
        text = "**专家:** 這個，問題"
        clean, mapping = transcriber._strip_for_matching_with_map(text)
        self.assertEqual(clean, transcriber._SPEAKER_BOUNDARY + "这个问题")
        self.assertEqual([text[i] for i in mapping[1:]], list("這個問題"))


if __name__ == "__main__":
    unittest.main()
