import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from inference import TTSClient
from text_processing import split_speech_text

STORY = "直到全球气温断崖式暴跌，城市一夜冰封，我才明白那不是bug"


class TextTests(unittest.TestCase):
    def test_story_keeps_context_through_all_commas(self):
        self.assertEqual(split_speech_text(STORY), [STORY])
        self.assertEqual(split_speech_text(STORY, "punctuation"), [
            "直到全球气温断崖式暴跌，", "城市一夜冰封，", "我才明白那不是bug"])

    def test_quotes_newlines_and_sentence_endings(self):
        self.assertEqual(split_speech_text('“真的，没错！”\n我才明白那不是bug。继续说。'),
                         ['“真的，没错！”', "我才明白那不是bug。", "继续说。"])

    def test_decimal_versions_urls_and_english_words_are_not_cut_at_dots(self):
        text = "版本v1.2.3，温度-3.5度，打开https://example.com/a，检查bug。"
        self.assertEqual(split_speech_text(text), [text])
        self.assertIn("版本v1.2.3，", split_speech_text(text, "punctuation"))

    def test_long_sentences_are_bounded_without_losing_text(self):
        text = ("城市突然冰封，" * 50) + "我才明白那不是bug。"
        chunks = split_speech_text(text)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))
        self.assertEqual("".join(chunks), text)
        text = "我" * 97 + "bugfix" + "之后" * 70
        chunks = split_speech_text(text)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(any("bugfix" in chunk for chunk in chunks))
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))

    def test_only_punctuation_is_rejected_and_invalid_mode_errors(self):
        self.assertEqual(split_speech_text(" ……！！\n，。"), [])
        with self.assertRaises(ValueError):
            split_speech_text(STORY, "unknown")

    def test_http_payload_does_not_cut_again_and_uses_stable_sampling(self):
        output = io.BytesIO()
        with wave.open(output, "wb") as audio:
            audio.setparams((1, 2, 32000, 0, "NONE", "not compressed"))
            audio.writeframes(b"\x00\x10" * 32)
        payloads = []

        class Response:
            status_code = 200
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def iter_content(self, chunk_size):
                yield output.getvalue()

        def post(url, **kwargs):
            payloads.append(kwargs["json"])
            return Response()

        client = TTSClient()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(client, "is_api_running", return_value=True), \
                patch("inference.requests.post", side_effect=post):
            path = str(Path(directory) / "result.wav")
            self.assertTrue(client.generate_speech(STORY, path)[0])
            self.assertTrue(client.generate_speech(STORY, path, segmentation="punctuation")[0])
        self.assertEqual(payloads[0]["text"], STORY)
        self.assertEqual(payloads[1]["text"].count("\n"), 2)
        self.assertTrue(all(p["cut_punc"] == "" for p in payloads))
        self.assertEqual(payloads[0]["top_p"], 0.6)
        self.assertEqual(payloads[0]["temperature"], 0.6)


if __name__ == "__main__":
    unittest.main()
