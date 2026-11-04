"""在送入引擎前规划合成段落，标点仍由模型决定韵律。"""
import re


SEGMENTATION_MODES = {"sentence", "punctuation"}
MAX_SEGMENT_CHARS = 100
_CLOSERS = '”’」』）》】\"\''


def _is_boundary(text, index, mode):
    char = text[index]
    if char in "。！？!?；;":
        return True
    if mode == "punctuation" and char in "，,：:、…":
        return True
    if char == ".":
        # 小数、版本号、域名和英文单词中的点不作为句号。
        following = text[index + 1:index + 2]
        return not following or following.isspace() or following in _CLOSERS or bool(
            re.match(r"[\u3400-\u9fff]", following))
    return False


def _shorten(text, max_chars):
    """过长的句子优先在逗号切开，其次空格，最后才按字数兜底。"""
    while len(text) > max_chars:
        candidates = [i + 1 for i, c in enumerate(text[:max_chars]) if c in "，,：:、…"]
        spaces = [m.end() for m in re.finditer(r"\s+", text[:max_chars])]
        end = next((p for p in reversed(candidates) if p >= max_chars // 3), None)
        if end is None:
            end = next((p for p in reversed(spaces) if p >= max_chars // 3), max_chars)
        # 不把 bug、版本号或 URL 的 ASCII 连续部分切成两半。
        for word in re.finditer(r"[A-Za-z0-9_./:+-]+", text):
            if word.start() < end < word.end():
                end = word.start() if word.start() else word.end()
                break
            if word.start() >= end:
                break
        if text[:end].strip():
            yield text[:end].strip()
        text = text[end:].strip()
    if text:
        yield text


def split_speech_text(text, mode="sentence", max_chars=MAX_SEGMENT_CHARS):
    """换行与句末分段；短句保留逗号上下文，避免每个逗号都重启推理。"""
    if mode not in SEGMENTATION_MODES:
        raise ValueError("不支持的断句方式")
    if max_chars < 2:
        raise ValueError("合成段落长度须大于 1")
    result = []
    for paragraph in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        paragraph = paragraph.strip()
        start = index = 0
        while index < len(paragraph):
            if _is_boundary(paragraph, index, mode):
                end = index + 1
                while end < len(paragraph) and (paragraph[end] in _CLOSERS or
                                               paragraph[end] in "。！？!?"):
                    end += 1
                result.extend(_shorten(paragraph[start:end].strip(), max_chars))
                start = end
                index = end
            else:
                index += 1
        result.extend(_shorten(paragraph[start:].strip(), max_chars))
    # 纯标点会被引擎忽略，提前过滤，避免返回空音频。
    return [part for part in result if any(c.isalnum() for c in part)]
