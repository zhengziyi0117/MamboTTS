"""加载上游旧 API 并安装小范围推理优化，不改下载的引擎源码。"""
from collections import OrderedDict
from contextvars import ContextVar
from functools import wraps
import importlib.util
import inspect
from pathlib import Path
import re
import sys
import threading

from engine_progress import EngineProgress, install_progress_api


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


def optimize_bert(api):
    """只取编码层特征，跳过未使用的 MLM 头，并在原设备展开音素。"""
    if not hasattr(api, "bert_model"):
        return
    api.bert_model = api.bert_model.base_model
    original_feature = api.get_bert_feature

    @wraps(original_feature)
    def get_bert_feature(text, word2ph):
        torch = api.torch
        with torch.no_grad():
            inputs = {key: value.to(api.device)
                      for key, value in api.tokenizer(text, return_tensors="pt").items()}
            output = api.bert_model(**inputs, output_hidden_states=True)
            hidden = output["hidden_states"][-3][0, 1:-1]
            assert len(word2ph) == len(text) == hidden.shape[0]
            repeats = torch.tensor(word2ph, device=hidden.device, dtype=torch.long)
            return torch.repeat_interleave(hidden, repeats, dim=0, output_size=sum(word2ph)).T

    api.get_bert_feature = get_bert_feature


def install_optimizations(api):
    """限定缓存大小；参考文件或模型变化后自动失效。"""
    reference = ContextVar("mambo_reference", default=None)
    mixed_text = ContextVar("mambo_mixed_text", default=False)
    lock = threading.RLock()
    progress = EngineProgress()
    token_counter = ContextVar("mambo_token_counter", default=None)
    feature_cache = BoundedCache(1)
    semantic_cache = BoundedCache(1)
    spec_cache = BoundedCache(2)
    phone_cache = BoundedCache(8)
    optimize_bert(api)

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
        progress.update("text")
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
    tts_signature = inspect.signature(original_tts)

    def hook_speaker(speaker):
        vq = speaker.sovits.vq_model
        if not getattr(vq, "_mambo_hooked", False):
            original_latent = vq.extract_latent
            original_decode = getattr(vq, "decode", None)

            @wraps(original_latent)
            def extract_latent(*args, **kwargs):
                key = reference.get()
                if key is None:
                    return original_latent(*args, **kwargs)
                return semantic_cache.get(key, lambda: original_latent(*args, **kwargs))

            @wraps(original_decode)
            def decode(*args, **kwargs):
                progress.update("vocoder")
                return original_decode(*args, **kwargs)

            vq.extract_latent = extract_latent
            if original_decode is not None:
                vq.decode = decode
            vq._mambo_hooked = True

        model = speaker.gpt.t2s_model.model
        if not getattr(model, "_mambo_hooked", False):
            original_infer = model.infer_panel

            @wraps(original_infer)
            def infer_panel(*args, **kwargs):
                if progress.current.get() is None:
                    return original_infer(*args, **kwargs)
                counter = [0]
                token = token_counter.set(counter)
                progress.update("semantic", current_tokens=0)
                try:
                    return original_infer(*args, **kwargs)
                finally:
                    progress.update(current_tokens=counter[0])
                    token_counter.reset(token)

            model.infer_panel = infer_panel
            predictor = getattr(model, "ar_predict_layer", None)
            if predictor is not None:
                original_predict = predictor.forward

                @wraps(original_predict)
                def predict(*args, **kwargs):
                    result = original_predict(*args, **kwargs)
                    counter = token_counter.get()
                    if counter is not None:
                        counter[0] += 1
                        if counter[0] % 16 == 0:
                            progress.update(current_tokens=counter[0])
                    return result

                predictor.forward = predict
            model._mambo_hooked = True

    if hasattr(api, "pack_audio"):
        original_audio = api.pack_audio

        @wraps(original_audio)
        def pack_audio(*args, **kwargs):
            result = original_audio(*args, **kwargs)
            # 已经过 .cpu() 同步和 PCM 转换，才算这段音频真正完成。
            progress.segment_done()
            return result

        api.pack_audio = pack_audio

    if hasattr(api, "pack_wav"):
        original_pack = api.pack_wav

        @wraps(original_pack)
        def pack_wav(*args, **kwargs):
            progress.update("packing")
            return original_pack(*args, **kwargs)

        api.pack_wav = pack_wav

    @wraps(original_tts)
    def get_tts_wav(ref_wav_path, *args, **kwargs):
        bound = tts_signature.bind(ref_wav_path, *args, **kwargs)
        text = bound.arguments.get("text", "")
        progress.update(total_segments=sum(any(c.isalnum() for c in line) for line in text.split("\n")))
        iterator = original_tts(ref_wav_path, *args, **kwargs)
        while True:
            # StreamingResponse 可能在不同线程里逐次 next()。ContextVar 必须
            # 在同一次 next() 内 set/reset，不能跨 yield 留下线程上下文。
            with lock:
                speaker = api.speaker_list[bound.arguments.get("spk", "default")]
                hook_speaker(speaker)
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
    return progress


def main():
    from run_engine import engine_paths
    paths = engine_paths()
    spec = importlib.util.spec_from_file_location("mambo_gsv_api", paths["api_py"])
    api = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = api
    spec.loader.exec_module(api)
    progress = install_optimizations(api)
    install_progress_api(api.app, progress)
    print("[优化] 已启用 BERT 编码优化、参考特征缓存、中英短词修正及实时进度。", flush=True)
    import uvicorn
    uvicorn.run(api.app, host=api.host, port=api.port, workers=1)


if __name__ == "__main__":
    main()
