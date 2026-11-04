"""用计数式替身校验缓存失效和语言边界，不加载模型或启动服务。"""
from collections import Counter
from contextvars import Context
from pathlib import Path
import tempfile
from types import SimpleNamespace as Namespace
import unittest

from engine_runtime import BoundedCache, install_optimizations


def fake_api():
    counts = Counter()
    hps = Namespace(data=Namespace(sampling_rate=32000, filter_length=2048,
                                   hop_length=640, win_length=2048))

    def ssl_forward(value):
        counts["ssl"] += 1
        return object()

    def spec(config, path):
        counts["spec"] += 1
        return object()

    def clean(text, language, version):
        if language == "en":
            return [",", "B", "AH1", "G"], None, text
        return [text], [1], text

    api = Namespace(counts=counts, get_spepc=spec, clean_text=clean,
                    ssl_model=Namespace(model=Namespace(forward=ssl_forward)),
                    speaker_list={"default": Namespace(sovits=Namespace(vq_model=object(), hps=hps))})

    def phones(text, language, version, final=False):
        counts[("phones", text)] += 1
        if language == "zh" and "bug" in text:
            # 中英片段在一次 get_phones_and_bert 调用中拼接。
            first, _, _ = api.clean_text(text.split("bug")[0], "zh", version)
            english, _, _ = api.clean_text("bug", "en", version)
            return first + english, object(), text
        result, _, norm = api.clean_text(text, language, version)
        return result, object(), norm

    def tts(path, prompt="参考音频", text="目标文案"):
        api.ssl_model.model.forward(path)
        api.get_spepc(api.speaker_list["default"].sovits.hps, path)
        api.get_phones_and_bert(prompt, "zh", "v2")
        api.get_phones_and_bert(text, "zh", "v2")
        yield b"wav"

    api.get_phones_and_bert = phones
    api.get_tts_wav = tts
    install_optimizations(api)
    return api


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "reference.wav"
        self.path.write_bytes(b"reference")
        self.api = fake_api()

    def tearDown(self):
        self.temp.cleanup()

    def test_different_texts_reuse_reference_features_and_reference_text(self):
        list(self.api.get_tts_wav(self.path, text="第一条文案"))
        list(self.api.get_tts_wav(self.path, text="第二条文案"))
        self.assertEqual(self.api.counts["ssl"], 1)
        self.assertEqual(self.api.counts["spec"], 1)
        self.assertEqual(self.api.counts[("phones", "参考音频")], 1)
        self.assertEqual(self.api.counts[("phones", "第二条文案")], 1)

    def test_file_change_and_model_change_invalidate_reference_cache(self):
        list(self.api.get_tts_wav(self.path))
        self.path.write_bytes(b"different reference")
        list(self.api.get_tts_wav(self.path))
        self.assertEqual(self.api.counts["ssl"], 2)
        self.assertEqual(self.api.counts["spec"], 2)
        self.api.speaker_list["default"].sovits = Namespace(
            vq_model=object(), hps=Namespace(data=Namespace(sampling_rate=16000,
                filter_length=1024, hop_length=320, win_length=1024)))
        list(self.api.get_tts_wav(self.path))
        self.assertEqual(self.api.counts["ssl"], 3)
        self.assertEqual(self.api.counts["spec"], 3)

    def test_short_english_in_mixed_text_drops_only_injected_comma(self):
        mixed, _, _ = self.api.get_phones_and_bert("这是bug", "zh", "v2")
        self.assertEqual(mixed, ["这是", "B", "AH1", "G"])
        english, _, _ = self.api.get_phones_and_bert("bug", "en", "v2")
        self.assertEqual(english[0], ",")
        actual, _, _ = self.api.get_phones_and_bert("这是，bug", "zh", "v2")
        self.assertEqual(actual[0], "这是，")

    def test_streaming_generator_does_not_leak_context_across_yields(self):
        iterator = self.api.get_tts_wav(self.path)
        self.assertEqual(Context().run(next, iterator), b"wav")
        with self.assertRaises(StopIteration):
            Context().run(next, iterator)
        # 上下文之外的模型调用不能错误命中参考音频缓存。
        self.api.ssl_model.model.forward("unrelated audio")
        self.assertEqual(self.api.counts["ssl"], 2)

    def test_cache_is_bounded_and_failed_values_are_not_cached(self):
        cache = BoundedCache(2)
        self.assertEqual(cache.get("a", lambda: 1), 1)
        cache.get("b", lambda: 2)
        cache.get("a", lambda: self.fail("expected cache hit"))
        cache.get("c", lambda: 3)
        self.assertEqual(list(cache.values), ["a", "c"])
        with self.assertRaises(ZeroDivisionError):
            cache.get("bad", lambda: 1 / 0)
        self.assertNotIn("bad", cache.values)


if __name__ == "__main__":
    unittest.main()
