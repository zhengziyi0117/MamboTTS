"""加载上游旧 API 并安装小范围推理优化，不改下载的引擎源码。"""
from collections import OrderedDict
from contextvars import ContextVar
from functools import wraps
import importlib.util
from pathlib import Path
import re
import sys
import threading


def file_signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class BoundedCache:
    def __init__(self, size):
        self.size = size
        self.values = OrderedDict()

    def get(self, key, compute):
        if key in self.values:
            value = self.values.pop(key)
        else:
            value = compute()
        self.values[key] = value
        while len(self.values) > self.size:
            self.values.popitem(last=False)
        return value


def install_optimizations(api):
    """限定缓存大小；参考文件或模型变化后自动失效。"""
    reference = ContextVar("mambo_reference", default=None)
    mixed_text = ContextVar("mambo_mixed_text", default=False)
    lock = threading.RLock()
    feature_cache = BoundedCache(1)
    spec_cache = BoundedCache(2)
    phone_cache = BoundedCache(8)

    original_clean = api.clean_text

    @wraps(original_clean)
    def clean_text(text, language, version=None):
        phones, word2ph, normalized = original_clean(text, language, version)
        # cleaner.py 会给不足 4 个音素的英文补逗号。中英混读时去掉这个
        # 人工前缀，保留文案中真正存在的逗号；纯英文短句继续走原逻辑。
        if (mixed_text.get() and language == "en" and 1 < len(phones) <= 4
                and phones[0] == "," and re.match(r"[A-Za-z0-9]", normalized)):
            phones = phones[1:]
        return phones, word2ph, normalized

    api.clean_text = clean_text
    original_phones = api.get_phones_and_bert

    @wraps(original_phones)
    def get_phones_and_bert(text, language, version, final=False):
        def compute():
            mixed = bool(re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", text)
                         and re.search(r"[A-Za-z]", text))
            token = mixed_text.set(mixed)
            try:
                return original_phones(text, language, version, final=final)
            finally:
                mixed_text.reset(token)
        with lock:
            # 避免缓存用户提交的超长文案，常用参考文本和重试短句可命中。
            if len(text) > 240:
                return compute()
            return phone_cache.get((text, language, version, final), compute)

    api.get_phones_and_bert = get_phones_and_bert
    original_spec = api.get_spepc

    @wraps(original_spec)
    def get_spepc(hps, filename):
        with lock:
            config = hps.data
            key = (file_signature(filename), config.sampling_rate, config.filter_length,
                   config.hop_length, config.win_length)
            return spec_cache.get(key, lambda: original_spec(hps, filename))

    api.get_spepc = get_spepc
    original_ssl = api.ssl_model.model.forward

    @wraps(original_ssl)
    def ssl_forward(*args, **kwargs):
        key = reference.get()
        if key is None:
            return original_ssl(*args, **kwargs)
        with lock:
            return feature_cache.get(key, lambda: original_ssl(*args, **kwargs))

    api.ssl_model.model.forward = ssl_forward
    original_tts = api.get_tts_wav

    @wraps(original_tts)
    def get_tts_wav(ref_wav_path, *args, **kwargs):
        iterator = original_tts(ref_wav_path, *args, **kwargs)
        while True:
            # StreamingResponse 可能在不同线程里逐次 next()。ContextVar 必须
            # 在同一次 next() 内 set/reset，不能跨 yield 留下线程上下文。
            with lock:
                speaker = api.speaker_list[kwargs.get("spk", "default")]
                key = (file_signature(ref_wav_path), id(speaker.sovits.vq_model),
                       speaker.sovits.hps.data.sampling_rate)
                token = reference.set(key)
                try:
                    chunk = next(iterator)
                except StopIteration:
                    return
                finally:
                    reference.reset(token)
            yield chunk

    api.get_tts_wav = get_tts_wav


def main():
    from run_engine import engine_paths
    paths = engine_paths()
    spec = importlib.util.spec_from_file_location("mambo_gsv_api", paths["api_py"])
    api = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = api
    spec.loader.exec_module(api)
    install_optimizations(api)
    print("[优化] 已启用参考 HuBERT / 频谱 / 文本特征缓存及中英短词停顿修正。", flush=True)
    import uvicorn
    uvicorn.run(api.app, host=api.host, port=api.port, workers=1)


if __name__ == "__main__":
    main()
