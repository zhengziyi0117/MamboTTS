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

    def latent(value):
        counts["latent"] += 1
        return object()

    def decode(value):
        counts["decode"] += 1
        return value

    def predict():
        return None

    predictor = Namespace(forward=predict)

    def infer():
        for _ in range(33):
            predictor.forward()
        return object()

    def speaker():
        return Namespace(sovits=Namespace(
            vq_model=Namespace(extract_latent=latent, decode=decode), hps=hps),
            gpt=Namespace(t2s_model=Namespace(model=Namespace(
                infer_panel=infer, ar_predict_layer=predictor))))

    api = Namespace(counts=counts, get_spepc=spec, clean_text=clean,
                    pack_audio=lambda audio_bytes, data, rate: audio_bytes,
                    ssl_model=Namespace(model=Namespace(forward=ssl_forward)),
                    speaker_list={"default": speaker()})

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
        content = api.ssl_model.model.forward(path)
        api.speaker_list["default"].sovits.vq_model.extract_latent(content)
        api.get_spepc(api.speaker_list["default"].sovits.hps, path)
        api.get_phones_and_bert(prompt, "zh", "v2")
        for segment in text.split("\n"):
            api.get_phones_and_bert(segment, "zh", "v2")
            semantic = api.speaker_list["default"].gpt.t2s_model.model.infer_panel()
            audio = api.speaker_list["default"].sovits.vq_model.decode(semantic)
            api.pack_audio(b"", [0] * 16000, 32000)
        yield b"wav"

    api.get_phones_and_bert = phones
    api.get_tts_wav = tts
    api.progress = install_optimizations(api)
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
        self.assertEqual(self.api.counts["latent"], 1)
        self.assertEqual(self.api.counts[("phones", "参考音频")], 1)
        self.assertEqual(self.api.counts[("phones", "第二条文案")], 1)

    def test_file_change_and_model_change_invalidate_reference_cache(self):
        list(self.api.get_tts_wav(self.path))
        self.path.write_bytes(b"different reference")
        list(self.api.get_tts_wav(self.path))
        self.assertEqual(self.api.counts["ssl"], 2)
        self.assertEqual(self.api.counts["spec"], 2)
        previous = self.api.speaker_list["default"].sovits.vq_model
        self.api.speaker_list["default"].sovits = Namespace(
            vq_model=Namespace(extract_latent=previous.extract_latent.__wrapped__,
                               decode=previous.decode.__wrapped__),
            hps=Namespace(data=Namespace(sampling_rate=16000,
                filter_length=1024, hop_length=320, win_length=1024)))
        list(self.api.get_tts_wav(self.path))
        self.assertEqual(self.api.counts["ssl"], 3)
        self.assertEqual(self.api.counts["spec"], 3)
        self.assertEqual(self.api.counts["latent"], 3)

    def test_progress_counts_actual_segments_and_autoregressive_steps(self):
        job_id = "a" * 32
        self.api.progress.start(job_id)
        token = self.api.progress.current.set(job_id)
        try:
            list(self.api.get_tts_wav(self.path, text="第一句\n第二句"))
        finally:
            self.api.progress.current.reset(token)
        progress = self.api.progress.get(job_id)
        self.assertEqual(progress["completed_segments"], 2)
        self.assertEqual(progress["segment_seconds"], (0.5, 0.5))
        self.assertEqual(progress["total_segments"], 2)
        self.assertEqual(progress["current_tokens"], 33)
        self.assertIn("semantic", progress["timings"])

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
